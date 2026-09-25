"""TRPO, ACER, ACKTR, IMPALA (practical TensorFlow implementations)."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf

from agent.buffers import ReplayBuffer
from agent.training.advanced.common import (
    cfg_get,
    checkpoint,
    compute_gae,
    finish,
    maybe_resume,
    sample_policy_action,
)
from agent.training.advanced.networks import build_policy_network, build_value_network


def train_trpo(env, **kwargs) -> dict[str, Any]:
    """TRPO-lite: natural PG via KL-constrained linear search (discrete)."""
    cfg = kwargs.get("cfg")
    episodes = int(kwargs.get("episodes", 31))
    episode_length = int(kwargs.get("episode_length", 256))
    gamma = float(kwargs.get("gamma", 0.97))
    learning_rate = float(kwargs.get("learning_rate", 3e-4))
    seed = int(kwargs.get("seed", 42))
    mode = str(kwargs.get("mode", "train"))
    checkpoint_dir = Path(kwargs["checkpoint_dir"]) if kwargs.get("checkpoint_dir") else None
    save_every_episodes = int(kwargs.get("save_every_episodes", 5))
    verbose = bool(kwargs.get("verbose", True))
    max_kl = float(cfg_get(cfg, "max_kl", 0.01))
    gae_lambda = float(cfg_get(cfg, "gae_lambda", 0.95))

    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    start_episode, policy, value_net, history_rows, _ = maybe_resume(checkpoint_dir, mode, verbose)
    if policy is None:
        policy = build_policy_network(env.n_features, 2, learning_rate, "TRPO_Policy")
        value_net = build_value_network(env.n_features, learning_rate, "TRPO_Value")
    v_opt = tf.keras.optimizers.Adam(learning_rate)
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        states, actions, rewards, dones, values, old_logits = [], [], [], [], [], []
        total_reward = 0.0
        correct = 0
        for _ in range(episode_length):
            s = np.asarray(state, np.float32).reshape(1, -1)
            logits = policy.predict(s, verbose=0)[0]
            value = float(value_net.predict(s, verbose=0)[0, 0])
            action = sample_policy_action(rng, logits)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            states.append(s[0]); actions.append(action); rewards.append(float(reward))
            dones.append(float(done)); values.append(value); old_logits.append(logits)
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state
            if done:
                break

        states_np = np.asarray(states, np.float32)
        actions_np = np.asarray(actions, np.int32)
        values_np = np.asarray(values, np.float32)
        rewards_np = np.asarray(rewards, np.float32)
        dones_np = np.asarray(dones, np.float32)
        old_logits_np = np.asarray(old_logits, np.float32)
        next_v = 0.0
        if dones_np[-1] < 0.5:
            next_v = float(value_net.predict(np.asarray(state, np.float32).reshape(1, -1), verbose=0)[0, 0])
        advantages, returns = compute_gae(rewards_np, values_np, dones_np, next_v, gamma, gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # Policy gradient direction
        with tf.GradientTape() as tape:
            logits = policy(states_np, training=True)
            log_p = tf.nn.log_softmax(logits)
            chosen = tf.reduce_sum(log_p * tf.one_hot(actions_np, 2), axis=1)
            old_log_p = tf.nn.log_softmax(tf.convert_to_tensor(old_logits_np))
            old_chosen = tf.reduce_sum(old_log_p * tf.one_hot(actions_np, 2), axis=1)
            ratio = tf.exp(chosen - old_chosen)
            surrogate = -tf.reduce_mean(ratio * advantages)
        grads = tape.gradient(surrogate, policy.trainable_variables)

        # Line search on step size with KL constraint
        old_weights = [w.copy() for w in policy.get_weights()]
        flat_grads = [g.numpy() if g is not None else np.zeros_like(v.numpy()) for g, v in zip(grads, policy.trainable_variables)]
        step = 1.0
        accepted = False
        for _ in range(10):
            new_weights = [w - step * 0.1 * g for w, g in zip(old_weights, flat_grads)]
            policy.set_weights(new_weights)
            new_logits = policy.predict(states_np, verbose=0)
            # KL(old || new)
            p_old = _softmax(old_logits_np)
            p_new = _softmax(new_logits)
            kl = float(np.mean(np.sum(p_old * (np.log(p_old + 1e-8) - np.log(p_new + 1e-8)), axis=1)))
            if kl <= max_kl:
                accepted = True
                break
            step *= 0.5
        if not accepted:
            policy.set_weights(old_weights)

        with tf.GradientTape() as tape:
            v_pred = tf.squeeze(value_net(states_np, training=True), axis=-1)
            loss_v = tf.reduce_mean(tf.square(returns - v_pred))
        v_grads = tape.gradient(loss_v, value_net.trainable_variables)
        v_opt.apply_gradients(zip(v_grads, value_net.trainable_variables))

        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(loss_v.numpy()),
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(f"[TRPO] Episode {episode:02d}/{episodes - 1} | Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f}")
        checkpoint(checkpoint_dir, episode, policy, history_rows, save_every_episodes, episodes, target_model=value_net)

    return finish(policy, history_rows, overall_start, target_model=value_net)


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=-1, keepdims=True)
    e = np.exp(x)
    return e / (e.sum(axis=-1, keepdims=True) + 1e-8)


def train_acer(env, **kwargs) -> dict[str, Any]:
    """ACER-lite: off-policy actor-critic with truncated importance sampling."""
    cfg = kwargs.get("cfg")
    episodes = int(kwargs.get("episodes", 31))
    episode_length = int(kwargs.get("episode_length", 256))
    batch_size = int(kwargs.get("batch_size", 64))
    gamma = float(kwargs.get("gamma", 0.97))
    learning_rate = float(kwargs.get("learning_rate", 3e-4))
    seed = int(kwargs.get("seed", 42))
    mode = str(kwargs.get("mode", "train"))
    checkpoint_dir = Path(kwargs["checkpoint_dir"]) if kwargs.get("checkpoint_dir") else None
    save_every_episodes = int(kwargs.get("save_every_episodes", 5))
    verbose = bool(kwargs.get("verbose", True))
    c_trunc = float(cfg_get(cfg, "truncation_c", 10.0))
    entropy_coef = float(cfg_get(cfg, "entropy_coef", 0.01))

    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    start_episode, policy, value_net, history_rows, _ = maybe_resume(checkpoint_dir, mode, verbose)
    if policy is None:
        policy = build_policy_network(env.n_features, 2, learning_rate, "ACER_Policy", softmax=True)
        value_net = build_value_network(env.n_features, learning_rate, "ACER_Value")
    pi_opt = tf.keras.optimizers.Adam(learning_rate)
    v_opt = tf.keras.optimizers.Adam(learning_rate)
    buffer = ReplayBuffer(capacity=int(cfg_get(cfg, "replay_capacity", 5000)), random_state=seed)
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        total_reward = 0.0
        correct = 0
        losses = []
        for _ in range(episode_length):
            s = np.asarray(state, np.float32).reshape(1, -1)
            probs = policy.predict(s, verbose=0)[0]
            action = sample_policy_action(rng, probs)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            buffer.add(state, action, reward, next_state, done)
            # store behavior prob in reward unused — approximate with current policy later
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state

            if len(buffer) >= batch_size:
                states, actions, rewards, next_states, dones = buffer.sample(batch_size)
                with tf.GradientTape(persistent=True) as tape:
                    pi = policy(states, training=True)
                    v = tf.squeeze(value_net(states, training=True), axis=-1)
                    v_next = tf.squeeze(value_net(next_states, training=True), axis=-1)
                    q_soft = v  # scalar baseline
                    # Importance weight truncated
                    pi_a = tf.reduce_sum(pi * tf.one_hot(actions, 2), axis=1)
                    # behavior ~ uniform-ish prior; use stop_gradient old pi as behavior approx
                    rho = tf.clip_by_value(pi_a / (tf.stop_gradient(pi_a) + 1e-6), 0.0, c_trunc)
                    targets = rewards + (1.0 - dones) * gamma * v_next
                    adv = targets - v
                    log_pi = tf.math.log(pi_a + 1e-8)
                    loss_pi = -tf.reduce_mean(rho * log_pi * tf.stop_gradient(adv))
                    entropy = -tf.reduce_mean(tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=1))
                    loss_pi = loss_pi - entropy_coef * entropy
                    loss_v = tf.reduce_mean(tf.square(targets - v))
                g_pi = tape.gradient(loss_pi, policy.trainable_variables)
                g_v = tape.gradient(loss_v, value_net.trainable_variables)
                pi_opt.apply_gradients(zip(g_pi, policy.trainable_variables))
                v_opt.apply_gradients(zip(g_v, value_net.trainable_variables))
                del tape
                losses.append(float(loss_pi.numpy() + loss_v.numpy()))

            if done:
                break

        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(np.mean(losses)) if losses else np.nan,
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(f"[ACER] Episode {episode:02d}/{episodes - 1} | Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f}")
        checkpoint(checkpoint_dir, episode, policy, history_rows, save_every_episodes, episodes, target_model=value_net)

    return finish(policy, history_rows, overall_start, target_model=value_net)


def train_acktr(env, **kwargs) -> dict[str, Any]:
    """ACKTR-lite: actor-critic with Fisher diagonal preconditioning (K-FAC approx)."""
    cfg = kwargs.get("cfg")
    episodes = int(kwargs.get("episodes", 31))
    episode_length = int(kwargs.get("episode_length", 256))
    gamma = float(kwargs.get("gamma", 0.97))
    learning_rate = float(kwargs.get("learning_rate", 3e-4))
    seed = int(kwargs.get("seed", 42))
    mode = str(kwargs.get("mode", "train"))
    checkpoint_dir = Path(kwargs["checkpoint_dir"]) if kwargs.get("checkpoint_dir") else None
    save_every_episodes = int(kwargs.get("save_every_episodes", 5))
    verbose = bool(kwargs.get("verbose", True))
    gae_lambda = float(cfg_get(cfg, "gae_lambda", 0.95))
    entropy_coef = float(cfg_get(cfg, "entropy_coef", 0.01))
    fisher_eps = float(cfg_get(cfg, "fisher_eps", 1e-3))

    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    start_episode, policy, value_net, history_rows, _ = maybe_resume(checkpoint_dir, mode, verbose)
    if policy is None:
        policy = build_policy_network(env.n_features, 2, learning_rate, "ACKTR_Policy")
        value_net = build_value_network(env.n_features, learning_rate, "ACKTR_Value")
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        states, actions, rewards, dones, values = [], [], [], [], []
        total_reward = 0.0
        correct = 0
        for _ in range(episode_length):
            s = np.asarray(state, np.float32).reshape(1, -1)
            logits = policy.predict(s, verbose=0)[0]
            value = float(value_net.predict(s, verbose=0)[0, 0])
            action = sample_policy_action(rng, logits)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            states.append(s[0]); actions.append(action); rewards.append(float(reward))
            dones.append(float(done)); values.append(value)
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state
            if done:
                break

        states_np = np.asarray(states, np.float32)
        actions_np = np.asarray(actions, np.int32)
        values_np = np.asarray(values, np.float32)
        rewards_np = np.asarray(rewards, np.float32)
        dones_np = np.asarray(dones, np.float32)
        next_v = 0.0
        if dones_np[-1] < 0.5:
            next_v = float(value_net.predict(np.asarray(state, np.float32).reshape(1, -1), verbose=0)[0, 0])
        advantages, returns = compute_gae(rewards_np, values_np, dones_np, next_v, gamma, gae_lambda)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        with tf.GradientTape() as tape:
            logits = policy(states_np, training=True)
            log_p = tf.nn.log_softmax(logits)
            chosen = tf.reduce_sum(log_p * tf.one_hot(actions_np, 2), axis=1)
            entropy = -tf.reduce_mean(tf.reduce_sum(tf.nn.softmax(logits) * log_p, axis=1))
            loss_pi = -tf.reduce_mean(chosen * advantages) - entropy_coef * entropy
        grads = tape.gradient(loss_pi, policy.trainable_variables)
        # Diagonal Fisher preconditioning: g / (g^2 + eps)
        new_grads = []
        for g in grads:
            if g is None:
                new_grads.append(None)
                continue
            fisher = tf.square(g) + fisher_eps
            new_grads.append(learning_rate * g / fisher)
        for var, g in zip(policy.trainable_variables, new_grads):
            if g is not None:
                var.assign_sub(g)

        with tf.GradientTape() as tape:
            v_pred = tf.squeeze(value_net(states_np, training=True), axis=-1)
            loss_v = tf.reduce_mean(tf.square(returns - v_pred))
        v_grads = tape.gradient(loss_v, value_net.trainable_variables)
        for var, g in zip(value_net.trainable_variables, v_grads):
            if g is not None:
                fisher = tf.square(g) + fisher_eps
                var.assign_sub(learning_rate * g / fisher)

        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(loss_pi.numpy() + loss_v.numpy()),
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(f"[ACKTR] Episode {episode:02d}/{episodes - 1} | Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f}")
        checkpoint(checkpoint_dir, episode, policy, history_rows, save_every_episodes, episodes, target_model=value_net)

    return finish(policy, history_rows, overall_start, target_model=value_net)


def train_impala(env, **kwargs) -> dict[str, Any]:
    """IMPALA-lite: V-trace actor-learner on a single process."""
    cfg = kwargs.get("cfg")
    episodes = int(kwargs.get("episodes", 31))
    episode_length = int(kwargs.get("episode_length", 256))
    gamma = float(kwargs.get("gamma", 0.97))
    learning_rate = float(kwargs.get("learning_rate", 3e-4))
    seed = int(kwargs.get("seed", 42))
    mode = str(kwargs.get("mode", "train"))
    checkpoint_dir = Path(kwargs["checkpoint_dir"]) if kwargs.get("checkpoint_dir") else None
    save_every_episodes = int(kwargs.get("save_every_episodes", 5))
    verbose = bool(kwargs.get("verbose", True))
    rho_bar = float(cfg_get(cfg, "rho_bar", 1.0))
    c_bar = float(cfg_get(cfg, "c_bar", 1.0))
    entropy_coef = float(cfg_get(cfg, "entropy_coef", 0.01))

    rng = np.random.default_rng(seed)
    tf.random.set_seed(seed)
    start_episode, policy, value_net, history_rows, _ = maybe_resume(checkpoint_dir, mode, verbose)
    if policy is None:
        policy = build_policy_network(env.n_features, 2, learning_rate, "IMPALA_Policy", softmax=True)
        value_net = build_value_network(env.n_features, learning_rate, "IMPALA_Value")
    pi_opt = tf.keras.optimizers.Adam(learning_rate)
    v_opt = tf.keras.optimizers.Adam(learning_rate)
    overall_start = time.time()

    for episode in range(start_episode, episodes):
        ep_start = time.time()
        state, _ = env.reset(seed=seed + episode)
        states, actions, rewards, dones, behavior_probs = [], [], [], [], []
        total_reward = 0.0
        correct = 0
        for _ in range(episode_length):
            s = np.asarray(state, np.float32).reshape(1, -1)
            probs = policy.predict(s, verbose=0)[0]
            action = sample_policy_action(rng, probs)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            states.append(s[0]); actions.append(action); rewards.append(float(reward))
            dones.append(float(done)); behavior_probs.append(float(probs[action]))
            total_reward += reward
            correct += int(info.get("correct", 0))
            state = next_state
            if done:
                break

        states_np = np.asarray(states, np.float32)
        actions_np = np.asarray(actions, np.int32)
        rewards_np = np.asarray(rewards, np.float32)
        dones_np = np.asarray(dones, np.float32)
        mu_a = np.asarray(behavior_probs, np.float32)

        values = value_net.predict(states_np, verbose=0).reshape(-1)
        bootstrap = 0.0
        if dones_np[-1] < 0.5:
            bootstrap = float(value_net.predict(np.asarray(state, np.float32).reshape(1, -1), verbose=0)[0, 0])
        values_ext = np.concatenate([values, [bootstrap]])

        with tf.GradientTape(persistent=True) as tape:
            pi = policy(states_np, training=True)
            pi_a = tf.reduce_sum(pi * tf.one_hot(actions_np, 2), axis=1)
            v = tf.squeeze(value_net(states_np, training=True), axis=-1)
            rho = tf.clip_by_value(pi_a / (tf.convert_to_tensor(mu_a) + 1e-8), 0.0, rho_bar)
            c = tf.clip_by_value(pi_a / (tf.convert_to_tensor(mu_a) + 1e-8), 0.0, c_bar)

            # V-trace targets (numpy-friendly loop then tensor)
            v_trace = np.zeros_like(values)
            last = bootstrap
            for t in reversed(range(len(values))):
                delta = float(rho[t]) * (rewards_np[t] + gamma * (1 - dones_np[t]) * values_ext[t + 1] - values[t])
                last = values[t] + delta + gamma * (1 - dones_np[t]) * float(c[t]) * (last - values_ext[t + 1])
                v_trace[t] = last
            v_targ = tf.convert_to_tensor(v_trace, tf.float32)
            pg_adv = rho * (rewards_np + gamma * (1 - dones_np) * np.concatenate([v_trace[1:], [bootstrap]]) - values)

            log_pi = tf.math.log(pi_a + 1e-8)
            loss_pi = -tf.reduce_mean(log_pi * tf.stop_gradient(pg_adv))
            entropy = -tf.reduce_mean(tf.reduce_sum(pi * tf.math.log(pi + 1e-8), axis=1))
            loss_pi = loss_pi - entropy_coef * entropy
            loss_v = tf.reduce_mean(tf.square(v_targ - v))

        g_pi = tape.gradient(loss_pi, policy.trainable_variables)
        g_v = tape.gradient(loss_v, value_net.trainable_variables)
        pi_opt.apply_gradients(zip(g_pi, policy.trainable_variables))
        v_opt.apply_gradients(zip(g_v, value_net.trainable_variables))
        del tape

        row = {
            "Episode": episode,
            "Total_Reward": total_reward,
            "Average_Reward": total_reward / max(episode_length, 1),
            "Episode_Accuracy": correct / max(episode_length, 1),
            "Mean_Loss": float(loss_pi.numpy() + loss_v.numpy()),
            "Episode_Time_Seconds": time.time() - ep_start,
        }
        history_rows.append(row)
        if verbose:
            print(f"[IMPALA] Episode {episode:02d}/{episodes - 1} | Reward: {total_reward:7.1f} | Acc: {row['Episode_Accuracy']:.3f}")
        checkpoint(checkpoint_dir, episode, policy, history_rows, save_every_episodes, episodes, target_model=value_net)

    return finish(policy, history_rows, overall_start, target_model=value_net)
