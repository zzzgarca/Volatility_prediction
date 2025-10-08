import tensorflow as tf
import keras
from keras import layers
from tkan import TKAN

@keras.saving.register_keras_serializable(package="tkan_seq2seq")
class TKANHierarchicalSeq2Seq(keras.Model):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        no_tasks: int = 3,
        task_output_dim: int = 1,
        hidden_layers: int = 2,
        pred_len: int = 28,
        hidden_dim: int = 32,
        dropout: float = 0.2,
        teacher_forcing: bool = True,
        sub_kan_configs=None,
        knots: int = 8,
        spline_power: int = 3,
        name: str = "TKANHierarchicalSeq2Seq",
    ):
        super().__init__(name=name)
        if sub_kan_configs is None:
            sub_kan_configs = [{
                "spline_order": spline_power,
                "grid_size": knots,
                "base_activation": "silu",
                "grid_range": (-1.0, 1.0),
            }]
        assert output_dim == no_tasks * task_output_dim
        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.no_tasks = int(no_tasks)
        self.task_output_dim = int(task_output_dim)
        self.pred_len = int(pred_len)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.teacher_forcing = bool(teacher_forcing)
        self.sub_kan_configs = sub_kan_configs
        self.tf_ratio = 1.0
        enc_layers = int(hidden_layers)
        dec_layers = int(hidden_layers)
        self.enc_blocks = []
        for _ in range(enc_layers):
            self.enc_blocks.append(TKAN(
                units=self.hidden_dim,
                return_sequences=True,
                return_state=False,
                dropout=0.0,
                recurrent_dropout=0.0,
                sub_kan_configs=self.sub_kan_configs,
            ))
        self.enc_last = TKAN(
            units=self.hidden_dim,
            return_sequences=False,
            return_state=True,
            dropout=0.0,
            recurrent_dropout=0.0,
            sub_kan_configs=self.sub_kan_configs,
        )
        self.dec_in_proj = []
        self.dec_layers = []
        self.out_proj = []
        self.start_tokens = []
        for t in range(self.no_tasks):
            if t == 0:
                dec_in_dim = 2 * self.task_output_dim
            elif t == 1:
                dec_in_dim = max(1, (self.no_tasks - 2) * self.task_output_dim)
            else:
                dec_in_dim = self.task_output_dim
            self.dec_in_proj.append(layers.Dense(self.hidden_dim, activation=None, name=f"dec{t}_in_proj"))
            task_stack = []
            for _ in range(dec_layers):
                task_stack.append(TKAN(
                    units=self.hidden_dim,
                    return_sequences=True,
                    return_state=True,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    sub_kan_configs=self.sub_kan_configs,
                ))
            self.dec_layers.append(task_stack)
            self.out_proj.append(layers.Dense(self.task_output_dim, name=f"out_proj_task_{t}"))
            if t >= 2:
                self.start_tokens.append(
                    self.add_weight(
                        name=f"start_token_task_{t}",
                        shape=(1, 1, dec_in_dim),
                        initializer="zeros",
                        trainable=True,
                    )
                )
            else:
                self.start_tokens.append(None)
        self.enc_dropout = layers.Dropout(self.dropout, name="enc_dropout")
        self.dec_dropout = layers.Dropout(self.dropout, name="dec_dropout")
        self.enc_to_task0 = layers.Dense(self.task_output_dim, name="enc_to_task0")

    def _encode(self, x, training: bool):
        h = x
        for blk in self.enc_blocks:
            h = blk(h, training=training)
            if training and self.dropout > 0.0:
                h = self.enc_dropout(h)
        enc_out, *enc_states = self.enc_last(h, training=training)
        return enc_out, enc_states

    def _vectorized_decode_task(self, task_idx, y_in_seq, enc_states, training: bool):
        x = self.dec_in_proj[task_idx](y_in_seq)
        if training and self.dropout > 0.0:
            x = self.dec_dropout(x)
        out = x
        states = enc_states
        for i, blk in enumerate(self.dec_layers[task_idx]):
            if i == 0 and states:
                out, *states = blk(out, initial_state=states, training=training)
            else:
                out, *states = blk(out, training=training)
            if training and self.dropout > 0.0:
                out = self.dec_dropout(out)
        return self.out_proj[task_idx](out)

    def _autoregressive_decode_aux(self, task_idx, enc_states, batch_size, dtype):
        assert task_idx >= 2
        if self.start_tokens[task_idx] is not None:
            dec_in = tf.tile(tf.cast(self.start_tokens[task_idx], dtype), [batch_size, 1, 1])
        else:
            dec_in = tf.zeros([batch_size, 1, self.task_output_dim], dtype=dtype)
        num_layers = len(self.dec_layers[task_idx])
        zero_state = tf.nest.map_structure(lambda s: tf.zeros_like(s), enc_states)
        states_per_layer = [enc_states] + [zero_state] * (num_layers - 1)
        ta = tf.TensorArray(dtype=dtype, size=self.pred_len)
        def cond(t, *_):
            return t < self.pred_len
        def body(t, dec_in, states_per_layer, ta):
            x = self.dec_in_proj[task_idx](tf.squeeze(dec_in, axis=1))
            x = tf.expand_dims(x, 1)
            out = x
            new_states = []
            for blk, st in zip(self.dec_layers[task_idx], states_per_layer):
                out, *st_new = blk(out, initial_state=st, training=False)
                new_states.append(st_new)
            y_t = self.out_proj[task_idx](tf.squeeze(out, axis=1))
            ta = ta.write(t, y_t)
            return t + 1, tf.expand_dims(y_t, 1), new_states, ta
        _, _, _, ta = tf.while_loop(
            cond, body,
            loop_vars=[tf.constant(0), dec_in, states_per_layer, ta],
            parallel_iterations=1
        )
        return tf.transpose(ta.stack(), [1, 0, 2])

    def call(self, inputs, targets=None, training=False):
        enc_out, enc_states = self._encode(inputs, training=training)
        batch = tf.shape(inputs)[0]
        dtype = inputs.dtype
        use_vectorized = bool(training and self.teacher_forcing and (targets is not None) and (self.tf_ratio > 0.0))
        aux_preds = []
        if use_vectorized:
            for t in range(2, self.no_tasks):
                if self.start_tokens[t] is not None:
                    start = tf.tile(tf.cast(self.start_tokens[t], dtype), [batch, 1, 1])
                else:
                    start = tf.zeros([batch, 1, self.task_output_dim], dtype=dtype)
                s = t * self.task_output_dim
                e = s + self.task_output_dim
                y_gt = targets[:, :, s:e]
                y_in = tf.concat([start, y_gt[:, :-1, :]], axis=1)
                aux_pred_t = self._vectorized_decode_task(t, y_in, enc_states, training=True)
                aux_preds.append(aux_pred_t)
            if self.no_tasks > 2:
                parts = [targets[:, :, tt*self.task_output_dim:(tt+1)*self.task_output_dim] for tt in range(2, self.no_tasks)]
                y_in1 = tf.concat(parts, axis=-1)
            else:
                y_in1 = tf.zeros([batch, self.pred_len, self.task_output_dim], dtype=dtype)
            task1 = self._vectorized_decode_task(1, y_in1, enc_states, training=True)
            enc_proj = self.enc_to_task0(enc_out)
            enc_seq = tf.tile(tf.expand_dims(enc_proj, 1), [1, self.pred_len, 1])
            s1 = 1 * self.task_output_dim
            e1 = s1 + self.task_output_dim
            t1_seq = targets[:, :, s1:e1]
            y_in0 = tf.concat([t1_seq, enc_seq], axis=-1)
            task0 = self._vectorized_decode_task(0, y_in0, enc_states, training=True)
        else:
            for t in range(2, self.no_tasks):
                aux_pred_t = self._autoregressive_decode_aux(t, enc_states, batch, dtype)
                aux_preds.append(aux_pred_t)
            if self.no_tasks > 2:
                y_in1 = tf.concat(aux_preds, axis=-1)
            else:
                y_in1 = tf.zeros([batch, self.pred_len, self.task_output_dim], dtype=dtype)
            task1 = self._vectorized_decode_task(1, y_in1, enc_states, training=False)
            enc_proj = self.enc_to_task0(enc_out)
            enc_seq = tf.tile(tf.expand_dims(enc_proj, 1), [1, self.pred_len, 1])
            y_in0 = tf.concat([task1, enc_seq], axis=-1)
            task0 = self._vectorized_decode_task(0, y_in0, enc_states, training=False)
        outs = [task0, task1] + aux_preds
        return tf.concat(outs, axis=-1)

    @property
    def metrics(self):
        return [self.loss_tracker]

    def _compiled_metrics_results(self):
        cm = getattr(self, "compiled_metrics", None)
        if cm is None:
            return {}
        try:
            return cm.result_dict()
        except Exception:
            try:
                names = cm.metrics_names
                values = cm.result()
                return {n: v for n, v in zip(names, values)}
            except Exception:
                return {}

    def train_step(self, data):
        x, y, sw = keras.utils.unpack_x_y_sample_weight(data)
        with tf.GradientTape() as tape:
            y_pred = self(x, targets=y, training=True)
            loss = self.compute_loss(x, y, y_pred, sw, training=True)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.loss_tracker.update_state(loss)
        self.compute_metrics(x, y, y_pred, sw)
        logs = {"loss": self.loss_tracker.result()}
        logs.update(self._compiled_metrics_results())
        return logs

    def test_step(self, data):
        x, y, sw = keras.utils.unpack_x_y_sample_weight(data)
        y_pred = self(x, targets=None, training=False)
        s, e = 0, self.task_output_dim
        val_loss = self.compute_loss(x, y[:, :, s:e], y_pred[:, :, s:e], sw, training=False)
        self.loss_tracker.update_state(val_loss)
        self.compute_metrics(x, y, y_pred, sw)
        logs = {"loss": self.loss_tracker.result()}
        logs.update(self._compiled_metrics_results())
        return logs

    def get_config(self):
        return {
            "input_dim": self.input_dim,
            "output_dim": self.output_dim,
            "no_tasks": self.no_tasks,
            "task_output_dim": self.task_output_dim,
            "pred_len": self.pred_len,
            "hidden_dim": self.hidden_dim,
            "dropout": self.dropout,
            "teacher_forcing": self.teacher_forcing,
            "sub_kan_configs": self.sub_kan_configs,
        }
