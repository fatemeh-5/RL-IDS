"""Shared network builders for advanced algorithms (Discrete-2 IDS)."""

from __future__ import annotations

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Add,
    Dense,
    Dropout,
    Input,
    LSTM,
    Lambda,
    Layer,
    Reshape,
    Softmax,
    Subtract,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2


def _trunk(inputs, name_prefix: str = ""):
    p = f"{name_prefix}_" if name_prefix else ""
    x = Dense(64, activation="relu", name=f"{p}dense1")(inputs)
    x = Dropout(0.1, name=f"{p}drop1")(x)
    x = Reshape((1, 64), name=f"{p}reshape")(x)
    x = LSTM(64, return_sequences=True, name=f"{p}lstm1")(x)
    x = LSTM(64, return_sequences=True, name=f"{p}lstm2")(x)
    x = LSTM(64, name=f"{p}lstm3")(x)
    x = Dense(32, activation="relu", kernel_regularizer=l2(0.01), name=f"{p}dense2")(x)
    x = Dropout(0.1, name=f"{p}drop2")(x)
    return x


def build_q_network(
    n_features: int,
    n_actions: int = 2,
    learning_rate: float = 3e-4,
    name: str = "QNet",
):
    inputs = Input(shape=(n_features,), name="state")
    x = _trunk(inputs)
    q = Dense(n_actions, activation="linear", name="q_values")(x)
    model = Model(inputs=inputs, outputs=q, name=name)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_dueling_q_network(
    n_features: int,
    n_actions: int = 2,
    learning_rate: float = 3e-4,
    name: str = "DuelingQ",
):
    inputs = Input(shape=(n_features,), name="state")
    shared = _trunk(inputs)
    v = Dense(1, activation="linear", name="value")(Dense(32, activation="relu")(shared))
    a = Dense(n_actions, activation="linear", name="advantage")(
        Dense(32, activation="relu")(shared)
    )
    mean_a = Lambda(lambda t: tf.reduce_mean(t, axis=1, keepdims=True))(a)
    q = Add(name="q_values")([v, Subtract()([a, mean_a])])
    model = Model(inputs=inputs, outputs=q, name=name)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_policy_network(
    n_features: int,
    n_actions: int = 2,
    learning_rate: float = 3e-4,
    name: str = "Policy",
    softmax: bool = False,
):
    inputs = Input(shape=(n_features,), name="state")
    x = _trunk(inputs, name_prefix="pi")
    logits = Dense(n_actions, activation="linear", name="logits")(x)
    outputs = Softmax(name="probs")(logits) if softmax else logits
    model = Model(inputs=inputs, outputs=outputs, name=name)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


def build_value_network(
    n_features: int,
    learning_rate: float = 3e-4,
    name: str = "Value",
):
    inputs = Input(shape=(n_features,), name="state")
    x = _trunk(inputs, name_prefix="v")
    value = Dense(1, activation="linear", name="value")(x)
    model = Model(inputs=inputs, outputs=value, name=name)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return model


class CategoricalExpectation(Layer):
    def __init__(self, n_actions=2, n_atoms=51, v_min=0.0, v_max=10.0, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = int(n_actions)
        self.n_atoms = int(n_atoms)
        self.v_min = float(v_min)
        self.v_max = float(v_max)

    def build(self, input_shape):
        self.support = tf.linspace(self.v_min, self.v_max, self.n_atoms)
        super().build(input_shape)

    def call(self, logits):
        batch = tf.shape(logits)[0]
        reshaped = tf.reshape(logits, (batch, self.n_actions, self.n_atoms))
        probs = tf.nn.softmax(reshaped, axis=-1)
        return tf.reduce_sum(probs * self.support, axis=-1)

    def get_config(self):
        return {
            **super().get_config(),
            "n_actions": self.n_actions,
            "n_atoms": self.n_atoms,
            "v_min": self.v_min,
            "v_max": self.v_max,
        }


class QuantileMean(Layer):
    def __init__(self, n_actions=2, n_quantiles=32, **kwargs):
        super().__init__(**kwargs)
        self.n_actions = int(n_actions)
        self.n_quantiles = int(n_quantiles)

    def call(self, quantiles):
        batch = tf.shape(quantiles)[0]
        reshaped = tf.reshape(quantiles, (batch, self.n_actions, self.n_quantiles))
        return tf.reduce_mean(reshaped, axis=-1)

    def get_config(self):
        return {
            **super().get_config(),
            "n_actions": self.n_actions,
            "n_quantiles": self.n_quantiles,
        }


def build_c51_pair(
    n_features: int,
    n_actions: int = 2,
    n_atoms: int = 51,
    v_min: float = 0.0,
    v_max: float = 10.0,
    learning_rate: float = 3e-4,
    name: str = "C51",
):
    """Returns (eval_q_model, logits_model). eval_q_model.predict -> (N, A)."""
    inputs = Input(shape=(n_features,), name="state")
    x = _trunk(inputs)
    logits = Dense(n_actions * n_atoms, activation="linear", name="atom_logits")(x)
    q = CategoricalExpectation(n_actions, n_atoms, v_min, v_max, name="q_values")(logits)
    eval_model = Model(inputs=inputs, outputs=q, name=name)
    logits_model = Model(inputs=inputs, outputs=logits, name=f"{name}_logits")
    eval_model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return eval_model, logits_model


# Register custom layers for .keras save/load round-trips.
tf.keras.utils.get_custom_objects().update(
    {
        "CategoricalExpectation": CategoricalExpectation,
        "QuantileMean": QuantileMean,
    }
)


def build_qr_pair(
    n_features: int,
    n_actions: int = 2,
    n_quantiles: int = 32,
    learning_rate: float = 3e-4,
    name: str = "QR_DQN",
):
    inputs = Input(shape=(n_features,), name="state")
    x = _trunk(inputs)
    quantiles = Dense(n_actions * n_quantiles, activation="linear", name="quantiles")(x)
    q = QuantileMean(n_actions, n_quantiles, name="q_values")(quantiles)
    eval_model = Model(inputs=inputs, outputs=q, name=name)
    quant_model = Model(inputs=inputs, outputs=quantiles, name=f"{name}_quantiles")
    eval_model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return eval_model, quant_model


def build_c51_dueling_pair(
    n_features: int,
    n_actions: int = 2,
    n_atoms: int = 51,
    v_min: float = 0.0,
    v_max: float = 10.0,
    learning_rate: float = 3e-4,
    name: str = "Rainbow",
):
    """Dueling + categorical (Rainbow-style head)."""
    inputs = Input(shape=(n_features,), name="state")
    shared = _trunk(inputs)
    v_logits = Dense(n_atoms, activation="linear", name="value_atoms")(
        Dense(32, activation="relu")(shared)
    )
    a_logits = Dense(n_actions * n_atoms, activation="linear", name="adv_atoms")(
        Dense(32, activation="relu")(shared)
    )

    def _dueling_atoms(args):
        v, a = args
        batch = tf.shape(v)[0]
        v = tf.reshape(v, (batch, 1, n_atoms))
        a = tf.reshape(a, (batch, n_actions, n_atoms))
        a = a - tf.reduce_mean(a, axis=1, keepdims=True)
        return tf.reshape(v + a, (batch, n_actions * n_atoms))

    logits = Lambda(_dueling_atoms, name="atom_logits")([v_logits, a_logits])
    q = CategoricalExpectation(n_actions, n_atoms, v_min, v_max, name="q_values")(logits)
    eval_model = Model(inputs=inputs, outputs=q, name=name)
    logits_model = Model(inputs=inputs, outputs=logits, name=f"{name}_logits")
    eval_model.compile(optimizer=Adam(learning_rate=learning_rate), loss="mse")
    return eval_model, logits_model
