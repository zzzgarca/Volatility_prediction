import torch
import torch.nn as nn
import efficient_kan as torch_kan

import tensorflow as tf
import keras
from keras import layers
from tkan import TKAN


class HierarchicalMultiTaskKAN(nn.Module):
    def __init__(self, input_dim, output_dim, no_tasks, knots=8, spline_power=4, dropout=0.0, hidden_layers=2, hidden_dim=64, head_hidden_dim=64):
        super().__init__()
        assert output_dim % no_tasks == 0
        assert no_tasks > 1
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.no_tasks = no_tasks
        self.knots = knots
        self.spline_power = spline_power
        self.dropout = dropout
        self.hidden_layers = hidden_layers
        self.hidden_dim = hidden_dim
        self.head_hidden_dim = head_hidden_dim
        self.task_output_dim = self.output_dim // self.no_tasks

        def build_trunk():
            seq = []
            in_f = self.input_dim
            for _ in range(self.hidden_layers):
                seq.append(
                    torch_kan.KANLinear(
                        in_features=in_f,
                        out_features=self.hidden_dim,
                        grid_size=self.knots,
                        spline_order=self.spline_power,
                        scale_noise=0.1,
                        scale_base=1.0,
                        scale_spline=1.0,
                        base_activation=torch.nn.SiLU,
                        grid_eps=0.02,
                        grid_range=[-1, 1],
                    )
                )
                seq.append(nn.Dropout(self.dropout))
                in_f = self.hidden_dim
            return nn.Sequential(*seq)

        self.trunk_aux = build_trunk()
        self.trunk_task1 = build_trunk()
        self.trunk_main = build_trunk()

        def make_head(in_dim):
            return nn.Sequential(
                torch_kan.KANLinear(
                    in_features=in_dim,
                    out_features=self.head_hidden_dim,
                    grid_size=self.knots,
                    spline_order=self.spline_power,
                    scale_noise=0.1,
                    scale_base=1.0,
                    scale_spline=1.0,
                    base_activation=torch.nn.SiLU,
                    grid_eps=0.02,
                    grid_range=[-1, 1],
                ),
                nn.Dropout(self.dropout),
                torch_kan.KANLinear(
                    in_features=self.head_hidden_dim,
                    out_features=self.head_hidden_dim,
                    grid_size=self.knots,
                    spline_order=self.spline_power,
                    scale_noise=0.1,
                    scale_base=1.0,
                    scale_spline=1.0,
                    base_activation=torch.nn.SiLU,
                    grid_eps=0.02,
                    grid_range=[-1, 1],
                ),
                nn.Dropout(self.dropout),
                torch_kan.KANLinear(
                    in_features=self.head_hidden_dim,
                    out_features=self.task_output_dim,
                    grid_size=self.knots,
                    spline_order=self.spline_power,
                    scale_noise=0.1,
                    scale_base=1.0,
                    scale_spline=1.0,
                    base_activation=torch.nn.SiLU,
                    grid_eps=0.02,
                    grid_range=[-1, 1],
                ),
            )

        self.task_layers = nn.ModuleDict()
        for task_id in range(2, self.no_tasks):
            self.task_layers[f"task_{task_id}"] = make_head(self.hidden_dim)
        in_task1 = self.hidden_dim + max(0, (self.no_tasks - 2) * self.head_hidden_dim)
        self.task_layers["task_1"] = make_head(in_task1)
        in_task0 = self.hidden_dim + self.head_hidden_dim
        self.task_layers["task_0"] = make_head(in_task0)

    def forward(self, x, detach_task2plus=False, detach_task1=False):
        xa = self.trunk_aux(x).contiguous()
        x1 = self.trunk_task1(x).contiguous()
        xm = self.trunk_main(x).contiguous()
        outputs = []
        embeds_2plus = []
        for task_id in range(2, self.no_tasks):
            head = self.task_layers[f"task_{task_id}"]
            h = head[0](xa)
            h = head[1](h)
            h = head[2](h)
            h = head[3](h)
            embeds_2plus.append(h)
            y_i = head[4](h)
            outputs.append((task_id, y_i))
        if embeds_2plus:
            H = torch.cat(embeds_2plus, dim=-1).contiguous()
            if detach_task2plus:
                H = H.detach()
            t1_in = torch.cat([x1, H], dim=-1).contiguous()
        else:
            t1_in = x1
        head1 = self.task_layers["task_1"]
        h1 = head1[0](t1_in)
        h1 = head1[1](h1)
        h1 = head1[2](h1)
        h1 = head1[3](h1)
        y1 = head1[4](h1)
        t0_in = torch.cat([xm, (h1.detach() if detach_task1 else h1)], dim=-1).contiguous()
        head0 = self.task_layers["task_0"]
        h0 = head0[0](t0_in)
        h0 = head0[1](h0)
        h0 = head0[2](h0)
        h0 = head0[3](h0)
        y0 = head0[4](h0)
        outputs = [y0, y1] + [y for (_tid, y) in sorted(outputs, key=lambda t: t[0])]
        return torch.cat(outputs, dim=-1).contiguous()

    def set_trainable_parts_KAN(self, phase):
        for p in self.parameters():
            p.requires_grad = False
        if phase == 0:
            for p in self.parameters():
                p.requires_grad = True
            return
        if phase == 1:
            for p in self.trunk_aux.parameters():
                p.requires_grad = True
            for tid in range(2, self.no_tasks):
                for layer in self.task_layers[f"task_{tid}"].parameters():
                    layer.requires_grad = True
            for layer in self.task_layers["task_1"].parameters():
                layer.requires_grad = True
            return
        if phase == 2:
            for p in self.trunk_main.parameters():
                p.requires_grad = True
            for layer in self.task_layers["task_0"].parameters():
                layer.requires_grad = True
            return
        if phase == 3:
            for p in self.trunk_task1.parameters():
                p.requires_grad = True
            for layer in self.task_layers["task_1"].parameters():
                layer.requires_grad = True
            return
        raise ValueError("phase must be one of {0,1,2,3}")


@keras.saving.register_keras_serializable(package="kan_dense_keras")
class KerasHierarchicalMultiTaskKAN(keras.Model):
    def __init__(self, input_dim, output_dim, no_tasks, knots=8, spline_power=4, dropout=0.0, hidden_layers=2, hidden_dim=64, head_hidden_dim=64, name="KerasHierarchicalMultiTaskKAN"):
        super().__init__(name=name)
        assert output_dim % no_tasks == 0
        assert no_tasks > 1
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.no_tasks = int(no_tasks)
        self.knots = int(knots)
        self.spline_power = int(spline_power)
        self.dropout = float(dropout)
        self.hidden_layers = int(hidden_layers)
        self.hidden_dim = int(hidden_dim)
        self.head_hidden_dim = int(head_hidden_dim)
        self.task_output_dim = self.output_dim // self.no_tasks

        def build_trunk():
            blocks = []
            for _ in range(self.hidden_layers):
                blocks.append(
                    TKAN(
                        units=self.hidden_dim,
                        return_sequences=False,
                        return_state=False,
                        dropout=0.0,
                        recurrent_dropout=0.0,
                        sub_kan_configs=[{"spline_order": self.spline_power, "grid_size": self.knots, "base_activation": "silu", "grid_range": (-1.0, 1.0)}],
                    )
                )
            return blocks

        self.trunk_aux = build_trunk()
        self.trunk_task1 = build_trunk()
        self.trunk_main = build_trunk()
        self.dropout_layer = layers.Dropout(self.dropout)

        def make_head(in_dim):
            return [
                layers.Dense(self.head_hidden_dim, activation=None),
                layers.Dropout(self.dropout),
                layers.Dense(self.head_hidden_dim, activation=None),
                layers.Dropout(self.dropout),
                layers.Dense(self.task_output_dim, activation=None),
            ]

        self.task_layers = {}
        for task_id in range(2, self.no_tasks):
            self.task_layers[f"task_{task_id}"] = make_head(self.hidden_dim)
        in_task1 = self.hidden_dim + max(0, (self.no_tasks - 2) * self.head_hidden_dim)
        self.task_layers["task_1"] = make_head(in_task1)
        in_task0 = self.hidden_dim + self.head_hidden_dim
        self.task_layers["task_0"] = make_head(in_task0)

    def _run_trunk(self, blocks, x, training):
        h = tf.expand_dims(x, 1)
        for blk in blocks:
            h1 = blk(h, training=training)
            if training and self.dropout > 0.0:
                h1 = self.dropout_layer(h1)
            h = tf.expand_dims(h1, 1)
        return h1

    def call(self, inputs, training=False, detach_task2plus=False, detach_task1=False):
        xa = self._run_trunk(self.trunk_aux, inputs, training)
        x1 = self._run_trunk(self.trunk_task1, inputs, training)
        xm = self._run_trunk(self.trunk_main, inputs, training)
        outputs = []
        embeds_2plus = []
        for task_id in range(2, self.no_tasks):
            head = self.task_layers[f"task_{task_id}"]
            h = head[0](xa)
            h = head[1](h, training=training)
            h = head[2](h)
            h = head[3](h, training=training)
            embeds_2plus.append(h)
            y_i = head[4](h)
            outputs.append((task_id, y_i))
        if len(embeds_2plus) > 0:
            H = tf.concat(embeds_2plus, axis=-1)
            t1_in = tf.concat([x1, H], axis=-1)
        else:
            t1_in = x1
        head1 = self.task_layers["task_1"]
        h1 = head1[0](t1_in)
        h1 = head1[1](h1, training=training)
        h1 = head1[2](h1)
        h1 = head1[3](h1, training=training)
        y1 = head1[4](h1)
        t0_in = tf.concat([xm, h1], axis=-1)
        head0 = self.task_layers["task_0"]
        h0 = head0[0](t0_in)
        h0 = head0[1](h0, training=training)
        h0 = head0[2](h0)
        h0 = head0[3](h0, training=training)
        y0 = head0[4](h0)
        outputs = [y0, y1] + [y for (_tid, y) in sorted(outputs, key=lambda t: t[0])]
        return tf.concat(outputs, axis=-1)
