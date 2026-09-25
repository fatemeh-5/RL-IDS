"""Value-based advanced DQN variants: C51, QR-DQN, Rainbow, M-DQN, R2D2, Ape-X."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import tensorflow as tf
from tensorflow.keras import Model

from agent.buffers import ReplayBuffer
from agent.buffers.prioritized import PrioritizedReplayBuffer, calculate_per_beta
from agent.training.advanced.common import (
    cfg_get,
    checkpoint,
    epsilon_action,
    finish,
    hard_update,
    maybe_resume,
)
from agent.training.advanced.networks import (
    build_c51_dueling_pair,
    build_c51_pair,
    build_dueling_q_network,
    build_q_network,
    build_qr_pair,
)


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-8)


def _log_softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    e = np.exp(x)
    return x - np.log(e.sum(axis=1, keepdims=True) + 1e-8)


def _unpack_batch(batch):
    """Uniform or PER batch -> states, actions, rewards, next_states, dones, weights, indices."""
    if len(batch) == 5:
        states, actions, rewards, next_states, dones = batch
        weights = np.ones(len(actions), dtype=np.float32)
        indices = None
    else:
        states, actions, rewards, next_states, dones, indices, weights = batch
    return states, actions, rewards, next_states, dones, weights, indices


def _logits_head(eval_model, layer_name: str = "atom_logits"):
    return Model(eval_model.input, eval_model.get_layer(layer_name).output)


def _dqn_loop(
    env,
    *,
    name: str,
    online,
    target,
    update_fn: Callable,
    episodes: int = 31,
    episode_length: int = 256,
    batch_size: int = 64,
    gamma: float = 0.97,
    seed: int = 42,
    mode: str = "train",
    checkpoint_dir: str | Path | None = None,
    save_every_episodes: int = 5,
    cfg: dict[str, Any] | None = None,
    verbose: bool = True,
    learning_rate: float = 3e-4,
    use_per: bool = False,
    replay_capacity: int = 5000,
    target_update_interval: int = 250,
    **_: Any,
) -> dict[str, Any]:
    checkpoint_dir = Path(checkpoint_dir) if checkpoint_dir else None
    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    epsilon_start = float(cfg_get(cfg, "epsilon_start", 1.0))
    epsilon_min = float(cfg_get(cfg, "epsilon_min", 0.05))
    epsilon_decay = float(cfg_get(cfg, "epsilon_decay", 0.90))

    start_episode, resumed_online, resumed_target, history_rows, extra = maybe_resume(
        checkpoint_dir, mode, verbose
    )
    if resumed_online is not None:
        online = resumed_online
        if resumed_target is not None:
            target = resumed_target
        else:
            hard_update(online, target)

    if use_per:
        buffer: Any = PrioritizedReplayBuffer(
            capacity=int(cfg_get(cfg, "replay_capacity", replay_capacity)),
            alpha=float(cfg_get(cfg, "per_alpha", 0.6)),
            priority_epsilon=float(cfg_get(cfg, "priority_epsilon", 1e-6)),
            random_state=seed,
        )
        beta_start = float(cfg_get(cfg, "beta_start", 0.4))
        beta_end = float(cfg_get(cfg, "beta_end", 1.0))
    else:
        buffer = ReplayBuffer(
            capacity=int(cfg_get(cfg, "replay_capacity", replay_capacity)),
            random_state=seed,
        )
        beta_start = beta_end = 1.0

    epsilon = epsilon_start
    if start_episode > 0:
        epsilon = float(extra.get("epsilon", epsilon_start))
        if history_rows and "Epsilon" in history_rows[-1]:
            epsilon = float(history_rows[-1]["Epsilon"])

    overall_start = time.time()
    global_step = 0
    target_update_interval = int(cfg_get(cfg, "target_update_interval", target_update_interval))

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        correct = 0
        losses: list[float] = []

        for _ in range(episode_length):
            s = np.asarray(state, dtype=np.float32).reshape(1, -1)
            q = online.predict(s, verbose=0)[0]
            action = epsilon_action(rng, q, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state
            global_step += 1

            if len(buffer) >= batch_size:
                if use_per:
                    beta = calculate_per_beta(episode, episodes, beta_start, beta_end)
                    batch = buffer.sample(batch_size, beta=beta)
                    loss, td_errors = update_fn(online, target, batch, gamma)
                    _, _, _, _, _, _, indices = _unpack_batch(batch)
                    buffer.update_priorities(indices, td_errors)
                else:
                    batch = buffer.sample(batch_size)
                    loss, _ = update_fn(online, target, batch, gamma)
                losses.append(float(loss))

            if target is not None and global_step % target_update_interval == 0:
                hard_update(online, target)

            if done:
                break

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(np.mean(losses)) if losses else np.nan,
            "Epsilon": epsilon,
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(
                f"[{name}] Episode {episode:02d}/{episodes - 1} | "
                f"Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f} | "
                f"Eps: {epsilon:.3f}"
            )
        checkpoint(
            checkpoint_dir,
            episode,
            online,
            history_rows,
            save_every_episodes,
            episodes,
            target_model=target,
            extra={"epsilon": epsilon},
        )

    return finish(online, history_rows, overall_start, target_model=target, epsilon=epsilon)


def _double_q_update(online, target, batch, gamma):
    states, actions, rewards, next_states, dones, weights, _ = _unpack_batch(batch)
    current_q = online.predict(states, verbose=0)
    next_online = online.predict(next_states, verbose=0)
    next_target = target.predict(next_states, verbose=0)
    best = np.argmax(next_online, axis=1)
    next_q = next_target[np.arange(len(actions)), best]
    bellman = rewards + (1.0 - dones) * gamma * next_q
    targets = current_q.copy()
    idx = np.arange(len(actions))
    td = bellman - current_q[idx, actions]
    targets[idx, actions] = bellman
    hist = online.fit(
        states, targets, sample_weight=weights, epochs=1, batch_size=len(actions), verbose=0
    )
    return float(hist.history["loss"][0]), np.abs(td)


def train_m_dqn(env, **kwargs) -> dict[str, Any]:
    cfg = kwargs.get("cfg")
    lr = float(kwargs.get("learning_rate", 3e-4))
    tau_m = float(cfg_get(cfg, "munchausen_tau", 0.03))
    alpha_m = float(cfg_get(cfg, "munchausen_alpha", 0.9))
    online = build_q_network(env.n_features, 2, lr, "M_DQN_online")
    target = build_q_network(env.n_features, 2, lr, "M_DQN_target")
    hard_update(online, target)

    def update_fn(online_m, target_m, batch, gamma):
        states, actions, rewards, next_states, dones, weights, _ = _unpack_batch(batch)
        current_q = online_m.predict(states, verbose=0)
        next_q = target_m.predict(next_states, verbose=0)
        log_pi_next = _log_softmax_np(next_q / max(tau_m, 1e-6))
        munch_next = tau_m * log_pi_next
        soft_value = np.sum(np.exp(log_pi_next) * (next_q + munch_next), axis=1)
        log_pi_sa = _log_softmax_np(current_q / max(tau_m, 1e-6))
        munch_sa = tau_m * log_pi_sa[np.arange(len(actions)), actions]
        bellman = rewards + alpha_m * munch_sa + (1.0 - dones) * gamma * soft_value
        targets = current_q.copy()
        idx = np.arange(len(actions))
        td = bellman - current_q[idx, actions]
        targets[idx, actions] = bellman
        hist = online_m.fit(
            states, targets, sample_weight=weights, epochs=1, batch_size=len(actions), verbose=0
        )
        return float(hist.history["loss"][0]), np.abs(td)

    return _dqn_loop(env, name="M-DQN", online=online, target=target, update_fn=update_fn, use_per=True, **kwargs)


def train_c51(env, **kwargs) -> dict[str, Any]:
    cfg = kwargs.get("cfg")
    lr = float(kwargs.get("learning_rate", 3e-4))
    n_atoms = int(cfg_get(cfg, "n_atoms", 51))
    v_min = float(cfg_get(cfg, "v_min", -10.0))
    v_max = float(cfg_get(cfg, "v_max", 20.0))
    online, _ = build_c51_pair(env.n_features, 2, n_atoms, v_min, v_max, lr, "C51_online")
    target, _ = build_c51_pair(env.n_features, 2, n_atoms, v_min, v_max, lr, "C51_target")
    hard_update(online, target)
    support = np.linspace(v_min, v_max, n_atoms, dtype=np.float32)
    delta_z = (v_max - v_min) / max(n_atoms - 1, 1)
    opt = tf.keras.optimizers.Adam(lr)

    # Will be rebound after possible resume inside loop via closure refresh
    def update_fn(online_m, target_m, batch, gamma):
        states, actions, rewards, next_states, dones, weights, _ = _unpack_batch(batch)
        online_logits = _logits_head(online_m, "atom_logits")
        target_logits = _logits_head(target_m, "atom_logits")
        next_actions = np.argmax(online_m.predict(next_states, verbose=0), axis=1)
        next_logits = target_logits.predict(next_states, verbose=0).reshape(-1, 2, n_atoms)
        next_probs = _softmax_np(next_logits[np.arange(len(actions)), next_actions])
        tz = np.clip(
            rewards[:, None] + (1.0 - dones[:, None]) * gamma * support[None, :],
            v_min,
            v_max,
        )
        b = (tz - v_min) / delta_z
        l = np.floor(b).astype(np.int32)
        u = np.ceil(b).astype(np.int32)
        l = np.clip(l, 0, n_atoms - 1)
        u = np.clip(u, 0, n_atoms - 1)
        m = np.zeros((len(actions), n_atoms), dtype=np.float32)
        for i in range(len(actions)):
            for j in range(n_atoms):
                m[i, l[i, j]] += next_probs[i, j] * (u[i, j] - b[i, j])
                m[i, u[i, j]] += next_probs[i, j] * (b[i, j] - l[i, j])

        with tf.GradientTape() as tape:
            logits = online_logits(states, training=True)
            logits_a = tf.reshape(logits, (-1, 2, n_atoms))
            gather = tf.gather(logits_a, tf.convert_to_tensor(actions, tf.int32), batch_dims=1)
            log_p = tf.nn.log_softmax(gather)
            loss = -tf.reduce_mean(
                tf.convert_to_tensor(weights, tf.float32)
                * tf.reduce_sum(tf.convert_to_tensor(m) * log_p, axis=1)
            )
        grads = tape.gradient(loss, online_m.trainable_variables)
        opt.apply_gradients(zip(grads, online_m.trainable_variables))
        td = np.abs(
            online_m.predict(states, verbose=0)[np.arange(len(actions)), actions]
            - (rewards + (1.0 - dones) * gamma * target_m.predict(next_states, verbose=0)[
                np.arange(len(actions)), next_actions
            ])
        )
        return float(loss.numpy()), td

    return _dqn_loop(env, name="C51", online=online, target=target, update_fn=update_fn, use_per=False, **kwargs)


def train_qr_dqn(env, **kwargs) -> dict[str, Any]:
    cfg = kwargs.get("cfg")
    lr = float(kwargs.get("learning_rate", 3e-4))
    n_quantiles = int(cfg_get(cfg, "n_quantiles", 32))
    online, _ = build_qr_pair(env.n_features, 2, n_quantiles, lr, "QR_online")
    target, _ = build_qr_pair(env.n_features, 2, n_quantiles, lr, "QR_target")
    hard_update(online, target)
    taus = (np.arange(n_quantiles, dtype=np.float32) + 0.5) / n_quantiles
    opt = tf.keras.optimizers.Adam(lr)

    def update_fn(online_m, target_m, batch, gamma):
        states, actions, rewards, next_states, dones, weights, _ = _unpack_batch(batch)
        online_q = _logits_head(online_m, "quantiles")
        target_q = _logits_head(target_m, "quantiles")
        next_actions = np.argmax(online_m.predict(next_states, verbose=0), axis=1)
        next_quant = target_q.predict(next_states, verbose=0).reshape(-1, 2, n_quantiles)
        next_q_a = next_quant[np.arange(len(actions)), next_actions]
        target_quant = rewards[:, None] + (1.0 - dones[:, None]) * gamma * next_q_a

        with tf.GradientTape() as tape:
            quant = tf.reshape(online_q(states, training=True), (-1, 2, n_quantiles))
            chosen = tf.gather(quant, tf.convert_to_tensor(actions, tf.int32), batch_dims=1)
            td = tf.expand_dims(tf.convert_to_tensor(target_quant), 1) - tf.expand_dims(chosen, 2)
            huber = tf.where(tf.abs(td) <= 1.0, 0.5 * tf.square(td), tf.abs(td) - 0.5)
            tau = tf.reshape(tf.constant(taus), (1, 1, n_quantiles))
            weight = tf.abs(tau - tf.cast(td < 0, tf.float32))
            loss = tf.reduce_mean(
                tf.convert_to_tensor(weights, tf.float32)[:, None, None] * weight * huber
            )
        grads = tape.gradient(loss, online_m.trainable_variables)
        opt.apply_gradients(zip(grads, online_m.trainable_variables))
        cur = online_q.predict(states, verbose=0).reshape(-1, 2, n_quantiles)[
            np.arange(len(actions)), actions
        ]
        td_abs = np.mean(np.abs(target_quant - cur), axis=1)
        return float(loss.numpy()), td_abs

    return _dqn_loop(env, name="QR-DQN", online=online, target=target, update_fn=update_fn, use_per=True, **kwargs)


def train_rainbow(env, **kwargs) -> dict[str, Any]:
    cfg = kwargs.get("cfg")
    lr = float(kwargs.get("learning_rate", 3e-4))
    n_atoms = int(cfg_get(cfg, "n_atoms", 51))
    v_min = float(cfg_get(cfg, "v_min", -10.0))
    v_max = float(cfg_get(cfg, "v_max", 20.0))
    n_step = int(cfg_get(cfg, "n_step", 3))
    online, _ = build_c51_dueling_pair(env.n_features, 2, n_atoms, v_min, v_max, lr, "Rainbow_online")
    target, _ = build_c51_dueling_pair(env.n_features, 2, n_atoms, v_min, v_max, lr, "Rainbow_target")
    hard_update(online, target)
    support = np.linspace(v_min, v_max, n_atoms, dtype=np.float32)
    delta_z = (v_max - v_min) / max(n_atoms - 1, 1)
    opt = tf.keras.optimizers.Adam(lr)
    gamma = float(kwargs.get("gamma", 0.97))
    gamma_n = gamma ** n_step

    def update_fn(online_m, target_m, batch, _gamma):
        states, actions, rewards, next_states, dones, weights, _ = _unpack_batch(batch)
        online_logits = _logits_head(online_m, "atom_logits")
        target_logits = _logits_head(target_m, "atom_logits")
        next_actions = np.argmax(online_m.predict(next_states, verbose=0), axis=1)
        next_logits = target_logits.predict(next_states, verbose=0).reshape(-1, 2, n_atoms)
        next_probs = _softmax_np(next_logits[np.arange(len(actions)), next_actions])
        tz = np.clip(
            rewards[:, None] + (1.0 - dones[:, None]) * gamma_n * support[None, :],
            v_min,
            v_max,
        )
        b = (tz - v_min) / delta_z
        l = np.clip(np.floor(b).astype(np.int32), 0, n_atoms - 1)
        u = np.clip(np.ceil(b).astype(np.int32), 0, n_atoms - 1)
        m = np.zeros((len(actions), n_atoms), dtype=np.float32)
        for i in range(len(actions)):
            for j in range(n_atoms):
                m[i, l[i, j]] += next_probs[i, j] * (u[i, j] - b[i, j])
                m[i, u[i, j]] += next_probs[i, j] * (b[i, j] - l[i, j])

        with tf.GradientTape() as tape:
            logits = online_logits(states, training=True)
            logits_a = tf.reshape(logits, (-1, 2, n_atoms))
            gather = tf.gather(logits_a, tf.convert_to_tensor(actions, tf.int32), batch_dims=1)
            log_p = tf.nn.log_softmax(gather)
            loss = -tf.reduce_mean(
                tf.convert_to_tensor(weights, tf.float32)
                * tf.reduce_sum(tf.convert_to_tensor(m) * log_p, axis=1)
            )
        grads = tape.gradient(loss, online_m.trainable_variables)
        opt.apply_gradients(zip(grads, online_m.trainable_variables))
        td = np.full(len(actions), float(loss.numpy()), dtype=np.float32)
        return float(loss.numpy()), td

    return _dqn_loop(env, name="RAINBOW", online=online, target=target, update_fn=update_fn, use_per=True, **kwargs)


def train_r2d2(env, **kwargs) -> dict[str, Any]:
    cfg = kwargs.get("cfg")
    lr = float(kwargs.get("learning_rate", 3e-4))
    online = build_dueling_q_network(env.n_features, 2, lr, "R2D2_online")
    target = build_dueling_q_network(env.n_features, 2, lr, "R2D2_target")
    hard_update(online, target)
    n_step = int(cfg_get(cfg, "n_step", 5))
    gamma = float(kwargs.get("gamma", 0.97))
    gamma_n = gamma ** n_step

    def update_fn(online_m, target_m, batch, _gamma):
        states, actions, rewards, next_states, dones, weights, _ = _unpack_batch(batch)
        current_q = online_m.predict(states, verbose=0)
        next_online = online_m.predict(next_states, verbose=0)
        next_target = target_m.predict(next_states, verbose=0)
        best = np.argmax(next_online, axis=1)
        next_q = next_target[np.arange(len(actions)), best]
        bellman = rewards + (1.0 - dones) * gamma_n * next_q
        targets = current_q.copy()
        idx = np.arange(len(actions))
        td = bellman - current_q[idx, actions]
        targets[idx, actions] = bellman
        hist = online_m.fit(
            states, targets, sample_weight=weights, epochs=1, batch_size=len(actions), verbose=0
        )
        return float(hist.history["loss"][0]), np.abs(td)

    cfg = dict(cfg or {})
    cfg.setdefault("replay_capacity", 10000)
    kwargs = {**kwargs, "cfg": cfg}
    return _dqn_loop(env, name="R2D2", online=online, target=target, update_fn=update_fn, use_per=True, **kwargs)


def train_apex(env, **kwargs) -> dict[str, Any]:
    cfg = dict(kwargs.get("cfg") or {})
    lr = float(kwargs.get("learning_rate", 3e-4))
    online = build_q_network(env.n_features, 2, lr, "APEX_online")
    target = build_q_network(env.n_features, 2, lr, "APEX_target")
    hard_update(online, target)
    cfg.setdefault("epsilon_start", 1.0)
    cfg.setdefault("epsilon_min", 0.01)
    cfg.setdefault("epsilon_decay", 0.95)
    cfg.setdefault("replay_capacity", 20000)
    cfg.setdefault("target_update_interval", 100)
    kwargs = {**kwargs, "cfg": cfg}
    return _dqn_loop(
        env,
        name="APEX",
        online=online,
        target=target,
        update_fn=_double_q_update,
        use_per=True,
        **kwargs,
    )
