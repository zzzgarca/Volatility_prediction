import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from arch import arch_model
from metrics import Metrics
from model_wrapper import Wrapper

class DataUtils:
    @staticmethod
    def load_two_series_csv(path, log_col=None, rv_col=None, orientation="auto"):
        try:
            df = pd.read_csv(path, engine="python", sep=None)
        except Exception:
            df = pd.read_csv(path)
        df_cols_norm = {c: str(c).strip().lower() for c in df.columns}
        inv_norm = {v: k for k, v in df_cols_norm.items()}

        def _to_numeric_frame(frame: pd.DataFrame) -> pd.DataFrame:
            out = frame.copy()
            for c in out.columns:
                out[c] = pd.to_numeric(out[c], errors="coerce")
            return out

        if log_col or rv_col:
            def _pick(name):
                key = str(name).strip().lower()
                if key in inv_norm:
                    return inv_norm[key]
                for k, v in df_cols_norm.items():
                    if key == v or key in v:
                        return k
                raise ValueError(f"Column '{name}' not found in CSV.")
            log_c = _pick(log_col) if log_col else None
            rv_c  = _pick(rv_col) if rv_col  else None
            if log_c is None or rv_c is None:
                raise ValueError("When specifying columns, provide both log_col and rv_col.")
            num = _to_numeric_frame(df[[log_c, rv_c]]).dropna(how="any")
            if num.empty:
                raise ValueError("After numeric coercion, selected columns are empty.")
            r_log = num.iloc[:, 0].to_numpy(dtype=float)
            r_var = num.iloc[:, 1].to_numpy(dtype=float)
            n = min(len(r_log), len(r_var))
            return r_log[:n], r_var[:n]

        if orientation == "auto":
            if df.shape[0] == 2 and df.shape[1] >= 2:
                r_log = pd.to_numeric(df.iloc[0], errors="coerce").to_numpy(dtype=float)
                r_var = pd.to_numeric(df.iloc[1], errors="coerce").to_numpy(dtype=float)
                n = min(len(r_log), len(r_var))
                m = np.isfinite(r_log[:n]) & np.isfinite(r_var[:n])
                return r_log[:n][m], r_var[:n][m]
            df_try = df.copy()
            if df_try.shape[1] >= 3:
                first_is_mostly_nonnum = pd.to_numeric(df_try.iloc[:, 0], errors="coerce").isna().mean() > 0.5
                if first_is_mostly_nonnum:
                    df_try = df_try.iloc[:, 1:]
            num = _to_numeric_frame(df_try)
            valid_cols = [c for c in num.columns if num[c].notna().sum() > 0]
            num = num[valid_cols].dropna(how="all")
            if num.shape[1] < 2:
                raise ValueError("Could not find two numeric columns in CSV.")
            col_scores = sorted([(c, num[c].notna().sum()) for c in num.columns], key=lambda x: x[1], reverse=True)
            pick_cols = [col_scores[0][0], col_scores[1][0]]
            num2 = num[pick_cols].dropna(how="any")
            if num2.empty:
                raise ValueError("After dropping NaNs, no aligned data for two numeric columns.")
            r_log = num2.iloc[:, 0].to_numpy(dtype=float)
            r_var = num2.iloc[:, 1].to_numpy(dtype=float)
            n = min(len(r_log), len(r_var))
            return r_log[:n], r_var[:n]
        raise ValueError("Unsupported orientation.")

    @staticmethod
    def fill_forward(arr):
        a = np.asarray(arr, dtype=float)
        n = len(a)
        if n == 0:
            return a
        out = a.copy()
        last = np.nan
        for i in range(n):
            if np.isfinite(out[i]):
                last = out[i]
            else:
                out[i] = last
        if not np.isfinite(out[0]):
            finite_vals = out[np.isfinite(out)]
            fill = float(np.nanmean(finite_vals)) if finite_vals.size else 0.0
            out[~np.isfinite(out)] = fill
        return out

    @staticmethod
    def compute_garch_rolling(log_returns, start_index=100, window=None, verbose=True, save_path=None):
        lr = np.asarray(log_returns, dtype=float)
        T = len(lr)
        preds = np.full(T, np.nan, dtype=float)
        if verbose:
            print(f"[GARCH] computing rolling GARCH(1,1) with start_index={start_index}, window={'full' if window is None else window}")
        for t in range(start_index, T):
            s = max(0, t - window) if window is not None else 0
            x = lr[s:t]
            x = x[np.isfinite(x)]
            if len(x) < 50:
                continue
            try:
                am = arch_model(x, vol="GARCH", p=1, q=1, dist="t", rescale=False)
                res = am.fit(disp="off", options={"maxiter": 500})
                f = res.forecast(horizon=1, reindex=False)
                preds[t] = float(f.variance.iloc[-1, 0])
            except Exception:
                preds[t] = np.nan
        if save_path:
            gp = DataUtils.fill_forward(preds)
            df = pd.DataFrame({"t": np.arange(T, dtype=int), "lr": lr, "garch_var": gp})
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            df.to_csv(save_path, index=False)
            if verbose:
                print(f"[GARCH] saved garch series to {save_path}")
        return preds

    @staticmethod
    def make_windows_from_series(series, win: int, horizon: int):
        s = np.asarray(series, dtype=float).reshape(-1)
        n = len(s)
        if n < win + horizon:
            raise ValueError("Series too short for given win and horizon")
        X = np.empty((n - win - horizon + 1, win, 1), dtype=float)
        Y = np.empty((n - win - horizon + 1, horizon, 1), dtype=float)
        for i, t in enumerate(range(win, n - horizon + 1)):
            X[i, :, 0] = s[t - win:t]
            Y[i, :, 0] = s[t:t + horizon]
        return X, Y

    @staticmethod
    def align_aux_for_horizon(aux_series, win: int, horizon: int):
        _, Y = DataUtils.make_windows_from_series(aux_series, win=win, horizon=horizon)
        return Y

    @staticmethod
    def row_mask_all_finite(*arrays):
        mask = None
        for arr in arrays:
            a = np.asarray(arr, dtype=float)
            dims = tuple(range(1, a.ndim))
            m = np.isfinite(a).all(axis=dims)
            mask = m if mask is None else (mask & m)
        return mask

    @staticmethod
    def split_time(X, y, test_frac=0.2):
        n = X.shape[0]
        cut = max(1, int(round((1 - test_frac) * n)))
        cut = min(max(1, cut), n - 1)
        return X[:cut], X[cut:], y[:cut], y[cut:]

    @staticmethod
    def scale_X_3d(X_train, X_test):
        Btr, T, F = X_train.shape
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_train.reshape(-1, F)).reshape(Btr, T, F)
        if X_test is None or len(X_test) == 0:
            Xte = None
        else:
            Bte = X_test.shape[0]
            Xte = scaler.transform(X_test.reshape(-1, F)).reshape(Bte, T, F)
        return Xtr, Xte, scaler

    @staticmethod
    def scale_y(y_train, y_test, transform="log", eps=1e-12, for_seq=False):
        if transform == "log":
            ytr = np.log(np.clip(y_train, eps, None))
            yte = None if y_test is None else np.log(np.clip(y_test, eps, None))
            return ytr, yte, {"mode": "log", "eps": eps}
        if transform == "identity":
            return y_train, y_test, {"mode": "identity", "eps": None}
        raise ValueError("transform must be 'log' or 'identity'.")

    @staticmethod
    def scale_y_2d(y_train, y_test, transform="log", eps=1e-12):
        return DataUtils.scale_y(y_train, y_test, transform=transform, eps=eps, for_seq=True)

    @staticmethod
    def flatten_for_mlp(X_3d, y_3d):
        Xf = X_3d.reshape(X_3d.shape[0], -1)
        Yf = y_3d.reshape(y_3d.shape[0], -1)
        return Xf, Yf

class TrainEvalModel:
    def __init__(
        self,
        model_kind="kan",
        loss_type="gauss_nll_var",
        target_transform="log",
        no_tasks=1,
        hidden_layers=3,
        hidden_dim=128,
        knots=10,
        spline_power=3,
        dropout=0.0,
        lr=1e-3,
        l2_weight=1e-5,
        epochs=50,
        batch_size=256,
        patience=10,
        min_epochs=50,
        min_delta=1e-4,
        warmup_aux_epochs=15,
        joint_epochs=15,
        verbose=True,
        seed=42
    ):
        self.model_kind = model_kind
        self.loss_type = loss_type
        self.target_transform = target_transform
        self.no_tasks = no_tasks
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.knots = knots
        self.spline_power = spline_power
        self.dropout = dropout
        self.lr = lr
        self.l2_weight = l2_weight
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.min_epochs = min_epochs
        self.min_delta = min_delta
        self.warmup_aux_epochs = warmup_aux_epochs
        self.joint_epochs = joint_epochs
        self.verbose = verbose
        self.seed = seed
        self.wrapper = None
        self.scaler_X = None
        self.y_ctx = None
        self.shapes = {}

    def prepare_from_csv(
        self,
        csv_path,
        input_len=60,
        pred_len=28,
        start_day=100,
        use_garch=True,
        garch_window=None,
        test_frac=0.2,
        log_col=None,
        rv_col=None,
        garch_save_path=None
    ):
        r_log, r_var = DataUtils.load_two_series_csv(csv_path, log_col=log_col, rv_col=rv_col, orientation="auto")
        X, y_rv = DataUtils.make_windows_from_series(r_var, win=input_len, horizon=pred_len)
        if self.model_kind in ("kan_hier", "kan_lstm") and use_garch:
            g = DataUtils.compute_garch_rolling(r_log, start_index=start_day, window=garch_window, verbose=self.verbose, save_path=garch_save_path)
            y_g = DataUtils.align_aux_for_horizon(g, win=input_len, horizon=pred_len)
            y = np.concatenate([y_rv, y_g], axis=2)
            self.shapes["no_tasks"] = 2
        else:
            y = y_rv
            self.shapes["no_tasks"] = 1
        Xtr, Xte, ytr, yte = DataUtils.split_time(X, y, test_frac=test_frac)
        Xtr_s, Xte_s, x_scaler = DataUtils.scale_X_3d(Xtr, Xte)
        ytr_t, yte_t, y_ctx = DataUtils.scale_y_2d(ytr, yte, transform=self.target_transform)
        self.scaler_X = x_scaler
        self.y_ctx = y_ctx
        self.shapes["input_len"] = input_len
        self.shapes["pred_len"] = pred_len
        self.shapes["features"] = Xtr_s.shape[2]
        if self.model_kind in ("kan", "kan_hier"):
            Xtr_f, ytr_f = DataUtils.flatten_for_mlp(Xtr_s, ytr_t)
            Xte_f, yte_f = DataUtils.flatten_for_mlp(Xte_s, yte_t) if Xte_s is not None else (None, None)
            return (Xtr_f, ytr_f), (Xte_f, yte_f)
        else:
            return (Xtr_s, ytr_t), (Xte_s, yte_t)

    def _build_wrapper(self, input_dim_or_shape, output_dim):
        if self.model_kind == "kan":
            return Wrapper(
                arch="kan",
                input_dim=input_dim_or_shape,
                output_dim=output_dim,
                no_tasks=1,
                hidden_layers=self.hidden_layers,
                hidden_dim=self.hidden_dim,
                knots=self.knots,
                spline_power=self.spline_power,
                dropout=self.dropout,
                lr=self.lr,
                l2_weight=self.l2_weight,
                epochs=self.epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                min_epochs=self.min_epochs,
                min_delta=self.min_delta,
                loss_type=self.loss_type,
                target_is_logvar=(self.target_transform == "log"),
                verbose=self.verbose
            )
        if self.model_kind == "kan_hier":
            nt = self.shapes.get("no_tasks", max(2, self.no_tasks))
            return Wrapper(
                arch="kan_hier",
                input_dim=input_dim_or_shape,
                output_dim=output_dim,
                no_tasks=nt,
                hidden_layers=self.hidden_layers,
                hidden_dim=self.hidden_dim,
                knots=self.knots,
                spline_power=self.spline_power,
                dropout=self.dropout,
                lr=self.lr,
                l2_weight=self.l2_weight,
                epochs=self.epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                min_epochs=self.min_epochs,
                min_delta=self.min_delta,
                loss_type=self.loss_type,
                target_is_logvar=(self.target_transform == "log"),
                warmup_aux_epochs=self.warmup_aux_epochs,
                joint_epochs=self.joint_epochs,
                verbose=self.verbose
            )
        if self.model_kind == "kan_lstm":
            return Wrapper(
                arch="kan_lstm",
                input_dim=input_dim_or_shape[1],
                output_dim=output_dim,
                no_tasks=max(1, self.no_tasks),
                pred_len=self.shapes["pred_len"],
                hidden_layers=self.hidden_layers,
                hidden_dim=self.hidden_dim,
                knots=self.knots,
                spline_power=self.spline_power,
                dropout=self.dropout,
                lr=self.lr,
                l2_weight=self.l2_weight,
                epochs=self.epochs,
                batch_size=self.batch_size,
                patience=self.patience,
                min_delta=self.min_delta,
                loss_type=("gauss_nll_var" if self.loss_type == "gauss_nll_var" else "mse"),
                target_is_logvar=(self.target_transform == "log"),
                verbose=self.verbose
            )
        raise ValueError("Unknown model_kind")

    def fit_from_arrays(self, train_pack, test_pack):
        Xtr, ytr = train_pack
        Xte, yte = test_pack
        if self.model_kind in ("kan", "kan_hier"):
            input_dim = Xtr.shape[1]
            output_dim = ytr.shape[1]
        else:
            input_dim = (Xtr.shape[1], Xtr.shape[2])
            output_dim = ytr.shape[2]
        self.wrapper = self._build_wrapper(input_dim, output_dim)
        self.wrapper.fit(Xtr, ytr, Xte, yte)
        return self

    def evaluate(self, test_pack, horizons=(1, 3, 5, 10, 20, -1)):
        Xte, yte_t = test_pack
        if Xte is None or yte_t is None or len(Xte) == 0:
            return {}
        if self.model_kind == "kan":
            y_pred_t = self.wrapper.predict(Xte)
            T = self.shapes["pred_len"]
            y_pred_2d = y_pred_t.reshape(-1, T)
            y_true_2d = yte_t.reshape(-1, T)
        elif self.model_kind == "kan_hier":
            y_pred_t = self.wrapper.predict(Xte)
            T = self.shapes["pred_len"]
            y_pred_2d = y_pred_t[:, :T]
            y_true_2d = yte_t.reshape(-1, T * self.shapes.get("no_tasks", 2))[:, :T]
        else:
            y_pred_t = self.wrapper.predict(Xte)
            y_pred_2d = y_pred_t[:, :, 0]
            y_true_2d = yte_t[:, :, 0]
        return Metrics.evaluate(y_true_2d, y_pred_2d, horizons=list(horizons))

    def predict_next(self, last_window_returns, last_window_garch=None):
        L = self.shapes["input_len"]
        F = self.shapes["features"]
        r = np.asarray(last_window_returns, dtype=float).reshape(-1)
        if len(r) != L:
            raise ValueError("last_window_returns length must equal input_len.")
        if F == 2:
            if last_window_garch is None:
                raise ValueError("garch window required for 2-feature models.")
            g = np.asarray(last_window_garch, dtype=float).reshape(-1)
            if len(g) != L:
                raise ValueError("last_window_garch length must equal input_len.")
            X = np.stack([r, g], axis=-1)[None, ...]
        else:
            X = r[:, None][None, ...]
        B, T, F = X.shape
        Xs = self.scaler_X.transform(X.reshape(-1, F)).reshape(B, T, F)
        if self.model_kind == "kan":
            Xf = Xs.reshape(1, -1)
            yp_t = self.wrapper.predict(Xf)
            Tpred = self.shapes["pred_len"]
            yp = yp_t.reshape(1, Tpred)
        elif self.model_kind == "kan_hier":
            Xf = Xs.reshape(1, -1)
            yp_t = self.wrapper.predict(Xf)
            Tpred = self.shapes["pred_len"]
            yp = yp_t[:, :Tpred]
        else:
            yp = self.wrapper.predict(Xs)
            yp = yp[:, :, 0]
        if self.y_ctx and self.y_ctx.get("mode") == "log":
            return np.exp(yp)
        return yp

    def save_bundle(self, path):
        import joblib
        bundle = {
            "model_kind": self.model_kind,
            "wrapper": self.wrapper,
            "scaler_X": self.scaler_X,
            "y_ctx": self.y_ctx,
            "shapes": self.shapes
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump(bundle, path)
        return path

    def load_bundle(self, path):
        import joblib
        b = joblib.load(path)
        self.model_kind = b["model_kind"]
        self.wrapper = b["wrapper"]
        self.scaler_X = b["scaler_X"]
        self.y_ctx = b["y_ctx"]
        self.shapes = b["shapes"]
        return self
