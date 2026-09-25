"""DQN model builders (same architectures as Untitled.ipynb)."""

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
    Reshape,
    Subtract,
)
from tensorflow.keras.losses import Huber
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.regularizers import l2

from agent.data.constants import BASELINE_FEATURE_COLUMNS


def build_baseline_dqn(
    n_features: int | None = None,
    learning_rate: float = 0.001,
    name: str = "Baseline_Stacked_LSTM_DQN",
):
    """B0 original paper-style softmax stacked-LSTM DQN."""

    n_features = n_features or len(BASELINE_FEATURE_COLUMNS)
    model = Sequential(name=name)
    model.add(Dense(64, activation="relu", input_shape=(n_features,)))
    model.add(Dropout(0.1))
    model.add(Reshape((1, 64)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(32, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(2, activation="softmax"))
    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def build_double_dqn(
    n_features: int | None = None,
    learning_rate: float = 0.001,
    name: str = "Double_DQN_Stacked_LSTM",
    n_actions: int = 2,
):
    """Linear-Q stacked-LSTM Double DQN (B1+ style)."""

    n_features = n_features or len(BASELINE_FEATURE_COLUMNS)

    model = Sequential(name=name)
    model.add(Input(shape=(n_features,)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.1))
    model.add(Reshape((1, 64)))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64, return_sequences=True))
    model.add(LSTM(64))
    model.add(Dense(32, activation="relu", kernel_regularizer=l2(0.01)))
    model.add(Dropout(0.1))
    model.add(Dense(n_actions, activation="linear", name="q_values"))
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber())
    return model


def build_dueling_dqn(
    input_features: int | None = None,
    action_count: int = 2,
    learning_rate: float = 0.001,
    model_name: str = "B6_Dueling_Double_DQN",
):
    """B6 dueling Double DQN: Q(s,a)=V(s)+A(s,a)-mean_a A(s,a)."""

    input_features = input_features or len(BASELINE_FEATURE_COLUMNS)
    if input_features <= 0:
        raise ValueError("input_features must be greater than zero.")
    if action_count <= 1:
        raise ValueError("action_count must be greater than one.")

    inputs = Input(shape=(input_features,), name="state_input")
    shared = Dense(64, activation="relu", name="shared_dense")(inputs)
    shared = Dropout(0.1, name="shared_dropout")(shared)
    shared = Reshape((1, 64), name="shared_reshape")(shared)
    shared = LSTM(64, return_sequences=True, name="shared_lstm_1")(shared)
    shared = LSTM(64, return_sequences=True, name="shared_lstm_2")(shared)
    shared = LSTM(64, name="shared_lstm_3")(shared)
    shared = Dense(
        32, activation="relu", kernel_regularizer=l2(0.01), name="shared_dense_2"
    )(shared)
    shared = Dropout(0.1, name="shared_dropout_2")(shared)

    value_hidden = Dense(32, activation="relu", name="value_hidden")(shared)
    state_value = Dense(1, activation="linear", name="state_value")(value_hidden)

    advantage_hidden = Dense(32, activation="relu", name="advantage_hidden")(shared)
    action_advantages = Dense(
        action_count, activation="linear", name="action_advantages"
    )(advantage_hidden)

    mean_advantage = Lambda(
        lambda advantages: tf.reduce_mean(advantages, axis=1, keepdims=True),
        name="mean_advantage",
    )(action_advantages)
    centered_advantages = Subtract(name="centered_advantages")(
        [action_advantages, mean_advantage]
    )
    q_values = Add(name="dueling_q_values")([state_value, centered_advantages])

    model = Model(inputs=inputs, outputs=q_values, name=model_name)
    model.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber())
    return model


def clone_compiled_model(model, learning_rate: float = 0.001):
    clone = tf.keras.models.clone_model(model)
    n_features = model.input_shape[-1]
    _ = clone(tf.zeros((1, n_features)))
    clone.set_weights(model.get_weights())
    # Prefer Huber for Double/Dueling; baseline uses CCE and should rebuild instead.
    clone.compile(optimizer=Adam(learning_rate=learning_rate), loss=Huber())
    return clone


def synchronize_target_network(online_model, target_model) -> None:
    target_model.set_weights(online_model.get_weights())
