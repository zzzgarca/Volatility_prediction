import numpy as np
import warnings
import torch
import torch.nn as nn
from kan import SimpleKAN, KerasSimpleKAN
from kan import HierarchicalMultiTaskKAN, KerasHierarchicalMultiTaskKAN
from kan import TKANHierarchicalSeq2Seq

try:
    import tensorflow as tf
    from tensorflow import keras
    class TFScheduler(keras.callbacks.Callback):
        def __init__(self, total_epochs, start=1.0, end=0.0, verbose=0):
            super().__init__()
            self.total_epochs = max(1, int(total_epochs))
            self.start = float(start)
            self.end = float(end)
            self.verbose = int(verbose)
        def on_epoch_begin(self, epoch, logs=None):
            p = epoch / max(1, self.total_epochs - 1)
            ratio = float(np.clip(self.start + p * (self.end - self.start), 0.0, 1.0))
            if hasattr(self.model, "tf_ratio"):
                self.model.tf_ratio = ratio
            if hasattr(self.model, "teacher_forcing"):
                self.model.teacher_forcing = (ratio > 0.0)
            if self.verbose:
                print(f"[TF] epoch={epoch+1} tf_ratio={ratio:.3f}")
    class KerasProgressLogger(keras.callbacks.Callback):
        def __init__(self, log_fn=None):
            super().__init__()
            self.log_fn = log_fn
        def on_epoch_end(self, epoch, logs=None):
            if self.log_fn:
                logs = logs or {}
                txt = ", ".join([f"{k}={float(v):.6f}" for k, v in logs.items() if v is not None])
                self.log_fn(f"Epoch {epoch+1}: {txt}")
except Exception:
    TFScheduler = None
    KerasProgressLogger = None
    tf = None
    keras = None

class Wrapper:
    def __init__(
        self,
        arch: str = "kan",
        input_dim=None,
        output_dim=None,
        no_tasks: int = 1,
        pred_len: int | None = None,
        hidden_layers: int = 3,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        knots: int = 8,
        spline_power: int = 3,
        lr: float = 1e-3,
        l2_weight: float = 1e-4,
        epochs: int = 50,
        batch_size: int = 512,
        patience: int = 10,
        min_epochs: int = 20,
        min_delta: float = 1e-4,
        warmup_aux_epochs: int | None = 10,
        joint_epochs: int | None = 10,
        loss_type: str = "gauss_nll_var",
        target_is_logvar: bool = True,
        device: str | None = None,
        verbose: bool = True,
        **kwargs
    ):
        aliases = {
            "weight_decay": "l2_weight",
            "wd": "l2_weight",
            "learning_rate": "lr",
            "bs": "batch_size",
            "batchsize": "batch_size",
            "grid_size": "knots",
            "noduri_kan": "knots",
            "spline_order": "spline_power",
            "n_layers": "hidden_layers",
            "layers": "hidden_layers",
            "dim_hidden": "hidden_dim",
            "dropout_rate": "dropout",
            "tasks": "no_tasks",
            "num_tasks": "no_tasks",
            "num_epochs": "epochs",
            "pred_len_out": "pred_len",
        }
        for k, v in list(kwargs.items()):
            if k in aliases:
                mapped = aliases[k]
                if verbose and mapped in ["l2_weight", "lr", "batch_size", "knots", "spline_power"]:
                    print(f"[Wrapper] mapping alias '{k}' -> '{mapped}' ({kwargs[k]})")
                setattr(self, mapped, kwargs.pop(k))
        self._extra_params = kwargs
        if verbose and kwargs:
            print(f"[Wrapper] ignoring unknown params: {sorted(kwargs.keys())}")
        self.arch = arch
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.no_tasks = int(no_tasks)
        self.pred_len = pred_len
        self.hidden_layers = int(hidden_layers)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.knots = int(knots)
        self.spline_power = int(spline_power)
        self.lr = float(lr)
        self.l2_weight = float(l2_weight)
        self.epochs = int(epochs)
        self.batch_size = int(batch_size)
        self.patience = int(patience)
        self.min_epochs = int(min_epochs)
        self.min_delta = float(min_delta)
        self.warmup_aux_epochs = None if warmup_aux_epochs is None else int(warmup_aux_epochs)
        self.joint_epochs = None if joint_epochs is None else int(joint_epochs)
        self.loss_type = str(loss_type)
        self.target_is_logvar = bool(target_is_logvar)
        self.verbose = bool(verbose)
        self.device = device or ("mps" if torch.backends.mps.is_available()
                                 else ("cuda" if torch.cuda.is_available() else "cpu"))
        self._backend = None
        self._model = None
        self._optimizer = None
        self._keras_callbacks = None
        if self.arch not in ("kan", "kan_hier", "kan_lstm"):
            raise ValueError("arch must be one of {'kan','kan_hier','kan_lstm'}")

    def _build_torch_kan(self):
        if self.input_dim is None or self.output_dim is None:
            raise ValueError("input_dim and output_dim are required for 'kan'")
        model = SimpleKAN(
            input_dim=int(self.input_dim),
            output_dim=int(self.output_dim),
            no_tasks=max(1, int(self.no_tasks)),
            hidden_layers=self.hidden_layers,
            dropout=self.dropout,
            knots=self.knots,
            spline_power=self.spline_power,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            name_l = name.lower()
            if name_l.endswith("bias") or "norm" in name_l:
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.l2_weight},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.lr
        )
        return model, opt

    def _build_torch_kan_hier(self):
        if self.input_dim is None or self.output_dim is None:
            raise ValueError("input_dim and output_dim are required for 'kan_hier'")
        if self.no_tasks < 2:
            raise ValueError("kan_hier requires no_tasks >= 2")
        model = HierarchicalMultiTaskKAN(
            input_dim=int(self.input_dim),
            output_dim=int(self.output_dim),
            no_tasks=int(self.no_tasks),
            knots=self.knots,
            spline_power=self.spline_power,
            dropout=self.dropout,
            hidden_layers=self.hidden_layers,
            hidden_dim=self.hidden_dim,
        ).to(self.device)
        decay, no_decay = [], []
        for name, p in model.named_parameters():
            if not p.requires_grad:
                continue
            n = name.lower()
            if n.endswith("bias") or "norm" in n or "grid" in n or "knot" in n or "spline" in n:
                no_decay.append(p)
            else:
                decay.append(p)
        opt = torch.optim.AdamW(
            [{"params": decay, "weight_decay": self.l2_weight},
             {"params": no_decay, "weight_decay": 0.0}],
            lr=self.lr
        )
        return model, opt

    def _keras_make_loss(self):
        import tensorflow as tf
        if self.loss_type == "mse":
            def _mse(y_true, y_pred):
                return tf.reduce_mean(tf.square(y_true - y_pred))
            return _mse
        def _nll_loss(y_true, y_pred):
            z = tf.clip_by_value(y_pred, -20.0, 20.0)
            v = tf.exp(y_true) if self.target_is_logvar else y_true
            v = tf.maximum(v, 1e-12)
            if self.loss_type == "gauss_nll_var":
                return tf.reduce_mean(v * tf.exp(-z) + z)
            if self.loss_type == "student_t_nll_var":
                nu = tf.cast(5.0, z.dtype)
                return tf.reduce_mean(0.5 * (nu + 1.0) * tf.math.log1p(v / (nu * tf.exp(z))) + 0.5 * z)
            return tf.reduce_mean(tf.square(y_true - y_pred))
        return _nll_loss

    def _build_keras_kan_lstm(self):
        from tensorflow import keras
        if self.input_dim is None or self.output_dim is None or self.pred_len is None:
            raise ValueError("kan_lstm requires input_dim, output_dim and pred_len")
        task_out = max(1, int(self.output_dim // max(1, self.no_tasks)))
        model = TKANHierarchicalSeq2Seq(
            input_dim=int(self.input_dim),
            output_dim=int(self.output_dim),
            no_tasks=int(self.no_tasks),
            task_output_dim=task_out,
            pred_len=int(self.pred_len),
            hidden_layers=self.hidden_layers,
            hidden_dim=self.hidden_dim,
            dropout=self.dropout,
            knots=self.knots,
            spline_power=self.spline_power,
            teacher_forcing=True,
        )
        opt = keras.optimizers.AdamW(learning_rate=self.lr, weight_decay=self.l2_weight)
        loss_fn = self._keras_make_loss()
        mets = [keras.metrics.MeanSquaredError(name="mse")]
        model.compile(optimizer=opt, loss=loss_fn, metrics=mets, run_eagerly=False, jit_compile=False)
        callbacks = [
            keras.callbacks.ReduceLROnPlateau(monitor="val_loss", mode="min", factor=0.5,
                                              patience=max(1, self.patience//2), verbose=1 if self.verbose else 0, min_lr=1e-7),
            keras.callbacks.EarlyStopping(monitor="val_loss", mode="min", patience=self.patience,
                                          min_delta=self.min_delta, restore_best_weights=True, verbose=1 if self.verbose else 0),
            keras.callbacks.TerminateOnNaN(),
        ]
        if TFScheduler is not None:
            callbacks.append(TFScheduler(total_epochs=self.epochs, start=1.0, end=0.0, verbose=1 if self.verbose else 0))
        self._keras_callbacks = callbacks
        return model

    def _build(self):
        if self.arch == "kan":
            self._backend = "torch"
            self._model, self._optimizer = self._build_torch_kan()
        elif self.arch == "kan_hier":
            self._backend = "torch"
            self._model, self._optimizer = self._build_torch_kan_hier()
        else:
            self._backend = "keras"
            self._model = self._build_keras_kan_lstm()

    def _torch_loss(self, y_hat, y_true):
        if self.loss_type == "mse":
            return nn.functional.mse_loss(y_hat, y_true, reduction="mean")
        z = torch.clamp(y_hat, -20.0, 20.0)
        v = torch.exp(y_true) if self.target_is_logvar else y_true
        v = torch.clamp(v, min=1e-12)
        if self.loss_type == "gauss_nll_var":
            return (v * torch.exp(-z) + z).mean()
        if self.loss_type == "student_t_nll_var":
            nu = torch.tensor(5.0, device=z.device, dtype=z.dtype)
            return (0.5 * (nu + 1.0) * torch.log1p(v / (nu * torch.exp(z))) + 0.5 * z).mean()
        return nn.functional.mse_loss(y_hat, y_true, reduction="mean")

    def fit(self, X, y, X_val, y_val, log_callback=None):
        if self._model is None:
            self._build()
        if self._backend == "keras":
            import numpy as np
            X = np.asarray(X, dtype=np.float32)
            y = np.asarray(y, dtype=np.float32)
            Xv = np.asarray(X_val, dtype=np.float32)
            yv = np.asarray(y_val, dtype=np.float32)
            if log_callback:
                log_callback(f"Shapes train X={X.shape} y={y.shape} | val X={Xv.shape} y={yv.shape}")
            else:
                print(f"Shapes train X={X.shape} y={y.shape} | val X={Xv.shape} y={yv.shape}")
            callbacks = list(self._keras_callbacks or [])
            if KerasProgressLogger is not None and log_callback is not None:
                callbacks.append(KerasProgressLogger(log_callback))
            self._model.fit(
                x=X, y=y,
                batch_size=self.batch_size,
                epochs=self.epochs,
                verbose=0,
                validation_data=(Xv, yv),
                shuffle=True,
                callbacks=callbacks
            )
            return self
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=self.device)
        Xv_t = torch.as_tensor(X_val, dtype=torch.float32, device=self.device)
        yv_t = torch.as_tensor(y_val, dtype=torch.float32, device=self.device)
        if log_callback:
            log_callback(f"Shapes train X={tuple(X_t.shape)} y={tuple(y_t.shape)} | val X={tuple(Xv_t.shape)} y={tuple(yv_t.shape)}")
        else:
            print(f"Shapes train X={tuple(X_t.shape)} y={tuple(y_t.shape)} | val X={tuple(Xv_t.shape)} y={tuple(yv_t.shape)}")
        loader = torch.utils.data.DataLoader(
            torch.utils.data.TensorDataset(X_t, y_t),
            batch_size=self.batch_size,
            shuffle=True
        )
        if self.arch == "kan":
            best_val = float("inf")
            best_state = None
            no_imp = 0
            for epoch in range(1, self.epochs + 1):
                self._model.train()
                running, nb = 0.0, 0
                for xb, yb in loader:
                    self._optimizer.zero_grad(set_to_none=True)
                    preds = self._model(xb)
                    loss = self._torch_loss(preds, yb)
                    loss.backward()
                    self._optimizer.step()
                    running += float(loss.detach().cpu()); nb += 1
                train_loss = running / max(1, nb)
                self._model.eval()
                with torch.no_grad():
                    vpred = self._model(Xv_t)
                    vloss = float(self._torch_loss(vpred, yv_t).detach().cpu())
                if log_callback:
                    log_callback(f"Epoch {epoch}: train={train_loss:.6f}, val={vloss:.6f}")
                elif self.verbose:
                    print(f"Epoch {epoch}: Train {self.loss_type}={train_loss:.4f} | Val {self.loss_type}={vloss:.4f}")
                if (best_val - vloss) > self.min_delta:
                    best_val = vloss; no_imp = 0
                    best_state = {k: v.detach().cpu() for k, v in self._model.state_dict().items()}
                else:
                    no_imp += 1
                if no_imp >= self.patience and epoch >= self.min_epochs:
                    if log_callback:
                        log_callback("Early stopping.")
                    elif self.verbose:
                        print("Early stopping.")
                    break
            if best_state is not None:
                self._model.load_state_dict(best_state)
            return self
        task_out = int(self.output_dim // max(1, self.no_tasks))
        aux_end = min(self.warmup_aux_epochs or 0, self.epochs)
        joint_end = min(aux_end + (self.joint_epochs or 0), self.epochs)
        best_val = float("inf")
        best_state = None
        no_imp = 0
        for epoch in range(self.epochs):
            if epoch < aux_end:
                phase = 1
            elif epoch < joint_end:
                phase = 0
            else:
                phase = 2
            self._model.set_trainable_parts_KAN(phase)
            self._model.train()
            running, nb = 0.0, 0
            for xb, yb in loader:
                self._optimizer.zero_grad(set_to_none=True)
                preds = self._model(xb).contiguous()
                total = 0.0
                if phase == 1:
                    for i in range(1, self.no_tasks):
                        s = i * task_out; e = s + task_out
                        total = total + self._torch_loss(preds[:, s:e], yb[:, s:e])
                elif phase == 0:
                    for i in range(self.no_tasks):
                        s = i * task_out; e = s + task_out
                        total = total + self._torch_loss(preds[:, s:e], yb[:, s:e])
                else:
                    total = total + self._torch_loss(preds[:, 0:task_out], yb[:, 0:task_out])
                total.backward()
                self._optimizer.step()
                running += float(total.detach().cpu()); nb += 1
            train_loss = running / max(1, nb)
            self._model.eval()
            with torch.no_grad():
                vpred = self._model(Xv_t).contiguous()
                vloss = float(self._torch_loss(vpred[:, 0:task_out], yv_t[:, 0:task_out]).detach().cpu())
            if log_callback:
                log_callback(f"Epoch {epoch+1:03d} | phase={phase} | train={train_loss:.6f} | val_main={vloss:.6f}")
            elif self.verbose:
                print(f"Epoch {epoch+1:03d} | phase={phase} | train={train_loss:.4f} | val_main={vloss:.6f}")
            if (best_val - vloss) > self.min_delta:
                best_val = vloss
                no_imp = 0
                best_state = {k: v.detach().cpu() for k, v in self._model.state_dict().items()}
            else:
                no_imp += 1
                if no_imp >= self.patience and (epoch + 1) >= self.min_epochs:
                    if log_callback:
                        log_callback("Early stopping.")
                    elif self.verbose:
                        print("Early stopping.")
                    break
        if best_state is not None:
            self._model.load_state_dict(best_state)
        return self

    def predict(self, X):
        if self._model is None:
            raise RuntimeError("Model not fitted")
        if self._backend == "keras":
            import numpy as np
            X = np.asarray(X, dtype=np.float32)
            return self._model.predict(X, batch_size=self.batch_size, verbose=0)
        X_t = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        self._model.eval()
        with torch.no_grad():
            if self.arch == "kan":
                preds = self._model(X_t)
                return preds.detach().cpu().numpy()
            else:
                preds = self._model(X_t).contiguous()
                return preds.detach().cpu().numpy()

    def get_params(self, deep=True):
        d = dict(
            arch=self.arch, input_dim=self.input_dim, output_dim=self.output_dim,
            no_tasks=self.no_tasks, pred_len=self.pred_len,
            hidden_layers=self.hidden_layers, hidden_dim=self.hidden_dim, dropout=self.dropout,
            knots=self.knots, spline_power=self.spline_power,
            lr=self.lr, l2_weight=self.l2_weight, epochs=self.epochs, batch_size=self.batch_size,
            patience=self.patience, min_epochs=self.min_epochs, min_delta=self.min_delta,
            warmup_aux_epochs=self.warmup_aux_epochs, joint_epochs=self.joint_epochs,
            loss_type=self.loss_type, target_is_logvar=self.target_is_logvar,
            device=self.device, verbose=self.verbose
        )
        if self._extra_params:
            d.update(self._extra_params)
        return d

    def set_params(self, **params):
        for k, v in params.items():
            setattr(self, k, v)
        self._model = None
        self._optimizer = None
        self._keras_callbacks = None
        return self
