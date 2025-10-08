import torch
import torch.nn as nn
from efficient_kan import kan as torch_kan

import tensorflow as tf
import keras
from keras import layers
from tkan import TKAN


class SimpleKAN(nn.Module):
    def __init__(self, input_dim, output_dim, no_tasks, hidden_layers=2, dropout=0.0, hidden_dim=64, knots=8, spline_power=3):
        super().__init__()
        self.no_tasks = no_tasks
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout
        self.knots = knots
        self.spline_power = spline_power
        self.input_dim = input_dim
        self.output_dim = output_dim
        layers_list = []
        layers_list.append(
            torch_kan.KANLinear(
                in_features=self.input_dim,
                out_features=self.hidden_dim,
                grid_size=self.knots,
                spline_order=self.spline_power,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1]
            )
        )
        layers_list.append(nn.Dropout(self.dropout))
        for _ in range(self.hidden_layers):
            layers_list.append(
                torch_kan.KANLinear(
                    in_features=self.hidden_dim,
                    out_features=self.hidden_dim,
                    grid_size=self.knots,
                    spline_order=self.spline_power,
                    scale_noise=0.1,
                    scale_base=1.0,
                    scale_spline=1.0,
                    base_activation=torch.nn.SiLU,
                    grid_eps=0.02,
                    grid_range=[-1, 1]
                )
            )
            layers_list.append(nn.Dropout(self.dropout))
        layers_list.append(
            torch_kan.KANLinear(
                in_features=self.hidden_dim,
                out_features=self.output_dim,
                grid_size=self.knots,
                spline_order=self.spline_power,
                scale_noise=0.1,
                scale_base=1.0,
                scale_spline=1.0,
                base_activation=torch.nn.SiLU,
                grid_eps=0.02,
                grid_range=[-1, 1]
            )
        )
        self.model = nn.Sequential(*layers_list)

    def forward(self, x):
        return self.model(x)


@keras.saving.register_keras_serializable(package="kan_dense_keras")
class KerasSimpleKAN(keras.Model):
    def __init__(self, input_dim, output_dim, hidden_layers=2, dropout=0.0, hidden_dim=64, knots=8, spline_power=3, name="KerasSimpleKAN"):
        super().__init__(name=name)
        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.hidden_layers = int(hidden_layers)
        self.dropout = float(dropout)
        self.hidden_dim = int(hidden_dim)
        self.knots = int(knots)
        self.spline_power = int(spline_power)
        self.enc_blocks = []
        for _ in range(self.hidden_layers):
            self.enc_blocks.append(
                TKAN(
                    units=self.hidden_dim,
                    return_sequences=False,
                    return_state=False,
                    dropout=0.0,
                    recurrent_dropout=0.0,
                    sub_kan_configs=[{"spline_order": self.spline_power, "grid_size": self.knots, "base_activation": "silu", "grid_range": (-1.0, 1.0)}],
                )
            )
        self.dropout_layer = layers.Dropout(self.dropout)
        self.out_proj = layers.Dense(self.output_dim)

    def call(self, inputs, training=False):
        x = tf.expand_dims(inputs, axis=1)
        for blk in self.enc_blocks:
            h = blk(x, training=training)
            if training and self.dropout > 0.0:
                h = self.dropout_layer(h)
            x = tf.expand_dims(h, 1)
        return self.out_proj(h)
