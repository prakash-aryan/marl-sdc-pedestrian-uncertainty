"""MAPPO training for high-level pedestrian crossing decisions + SDC driving.

Pedestrian action: Discrete(2), go or wait.
SDC action: Continuous(2), acceleration and steering.
Shared centralized critic.
"""

import os, sys, time, pickle
import jax
import jax.numpy as jnp
import optax
import flax.linen as nn
import distrax

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from jax_impl.decision_env import (
    reset, step, get_all_ped_obs, get_sdc_obs, NUM_PEDS,
    PED_OBS_SIZE, SDC_OBS_SIZE,
)

# Config
NUM_ENVS = 512
NUM_STEPS = 256
NUM_EPOCHS = 4
NUM_MINIBATCHES = 8
TOTAL_UPDATES = 5000    # longer training for better convergence
LR = 3e-4
GAMMA = 0.995
GAE_LAMBDA = 0.95
CLIP_EPS = 0.2
ENT_COEF_PED = 0.03
ENT_COEF_SDC = 0.01
VF_COEF = 0.5
MAX_GRAD_NORM = 0.5
CHECKPOINT_DIR = "checkpoints_jax"

# Networks
class PedDecisionActor(nn.Module):
    """Pedestrian high-level policy: Discrete(2), go (0) or wait (1). Jaywalking is personality-driven."""
    @nn.compact
    def __call__(self, obs):
        x = nn.relu(nn.Dense(128)(obs))
        x = nn.relu(nn.Dense(128)(x))
        return nn.Dense(2)(x)  # logits for go vs wait

class SDCActor(nn.Module):
    """SDC policy: Continuous(2), acceleration and steering."""
    @nn.compact
    def __call__(self, obs):
        x = nn.relu(nn.Dense(256)(obs))
        x = nn.relu(nn.Dense(256)(x))
        mean = nn.Dense(2)(x)
        log_std = self.param("log_std", nn.initializers.zeros, (2,))
        return mean, jnp.clip(log_std, -2.0, 0.5)

class Critic(nn.Module):
    """Shared value function."""
    @nn.compact
    def __call__(self, obs):
        x = nn.relu(nn.Dense(256)(obs))
        x = nn.relu(nn.Dense(256)(x))
        return nn.Dense(1)(x).squeeze(-1)

# Global state for critic: concat of ped and sdc obs summaries
from jax_impl.decision_env import NUM_PEDS as _NP
CRITIC_OBS_SIZE = _NP * 4 + 10  # ped positions+speeds + sdc state

def get_critic_obs(state):
    obs = jnp.zeros(CRITIC_OBS_SIZE)
    N = _NP
    obs = obs.at[0:N].set(state["ped_x"] / 120.0)
    obs = obs.at[N:2*N].set(state["ped_y"] / 120.0)
    obs = obs.at[2*N:3*N].set(state["ped_speed"] / 2.5)
    obs = obs.at[3*N:4*N].set(state["ped_traits"][:, 0])  # jaywalking tendency
    obs = obs.at[4*N].set(state["sdc_x"] / 120.0)
    obs = obs.at[4*N+1].set(state["sdc_y"] / 120.0)
    obs = obs.at[4*N+2].set(state["sdc_speed"] / 8.33)
    obs = obs.at[4*N+3].set(state["sdc_heading"] / jnp.pi)
    obs = obs.at[4*N+4].set(state["sdc_goal_x"] / 120.0)
    obs = obs.at[4*N+5].set(state["sdc_goal_y"] / 120.0)
    obs = obs.at[4*N+6].set(state["step_count"].astype(jnp.float32) / 500.0)
    return obs

# GAE
def compute_gae(rewards, values, dones, last_value):
    def _step(carry, t):
        gae, nv = carry
        r, v, d = t
        delta = r + GAMMA * nv * (1-d) - v
        gae = delta + GAMMA * GAE_LAMBDA * (1-d) * gae
        return (gae, v), gae
    _, advs = jax.lax.scan(_step, (jnp.zeros_like(last_value), last_value),
                            (rewards[::-1], values[::-1], dones[::-1]))
    advs = advs[::-1]
    return advs, advs + values

# Training
def train():
    print("=" * 60)
    print("MAPPO High-Level Decision Training: JAX GPU")
    print("=" * 60)
    print(f"Device: {jax.devices()[0]}")
    print(f"Envs: {NUM_ENVS}, Steps: {NUM_STEPS}, Updates: {TOTAL_UPDATES}")
    print(f"Ped action: Discrete(2) [go/wait]")
    print(f"SDC action: Continuous(2) [accel/steer]")
    print()

    key = jax.random.PRNGKey(0)
    k1, k2, k3, k4 = jax.random.split(key, 4)

    ped_actor = PedDecisionActor()
    sdc_actor = SDCActor()
    critic = Critic()

    ped_params = ped_actor.init(k1, jnp.zeros(PED_OBS_SIZE))
    sdc_params = sdc_actor.init(k2, jnp.zeros(SDC_OBS_SIZE))
    critic_params = critic.init(k3, jnp.zeros(CRITIC_OBS_SIZE))

    ped_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    sdc_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    crt_opt = optax.chain(optax.clip_by_global_norm(MAX_GRAD_NORM), optax.adam(LR))
    ped_os = ped_opt.init(ped_params)
    sdc_os = sdc_opt.init(sdc_params)
    crt_os = crt_opt.init(critic_params)

    env_keys = jax.random.split(k4, NUM_ENVS)
    states = jax.vmap(reset)(env_keys)

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    @jax.jit
    def collect_and_update(states, pp, sp, cp, po, so, co, key):
        # Collect rollout
        def env_step(carry, _):
            sts, key = carry
            key, k1, k2, k3 = jax.random.split(key, 4)

            ped_obs = jax.vmap(get_all_ped_obs)(sts)     # (E, P, 20)
            sdc_obs = jax.vmap(get_sdc_obs)(sts)           # (E, 30)
            crt_obs = jax.vmap(get_critic_obs)(sts)        # (E, 55)
            values = jax.vmap(critic.apply, in_axes=(None,0))(cp, crt_obs)

            # Ped actions: sample from categorical
            ped_logits = jax.vmap(jax.vmap(ped_actor.apply, in_axes=(None,0)), in_axes=(None,0))(pp, ped_obs)  # (E,P,3)
            # Sample ped actions with vmap (distrax needs single keys)
            ped_keys = jax.random.split(k1, NUM_ENVS * NUM_PEDS).reshape(NUM_ENVS, NUM_PEDS, 2)
            def sample_ped(logits, key):
                dist = distrax.Categorical(logits=logits)
                action = dist.sample(seed=key)
                lp = dist.log_prob(action)
                return action, lp
            ped_actions = jnp.zeros((NUM_ENVS, NUM_PEDS), dtype=jnp.int32)
            ped_lp = jnp.zeros((NUM_ENVS, NUM_PEDS))  # (E,P), (E,P)

            # SDC actions with vmap
            sdc_out = jax.vmap(sdc_actor.apply, in_axes=(None,0))(sp, sdc_obs)
            sdc_mean, sdc_logstd = sdc_out[0], sdc_out[1]
            sdc_keys = jax.random.split(k2, NUM_ENVS)
            def sample_sdc(mean, logstd, key):
                std = jnp.exp(logstd)
                dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
                raw = dist.sample(seed=key)
                action = jnp.tanh(raw)
                lp = dist.log_prob(raw) - jnp.sum(jnp.log(1 - action**2 + 1e-6), axis=-1)
                return action, lp
            sdc_actions, sdc_lp = jax.vmap(sample_sdc)(sdc_mean, sdc_logstd, sdc_keys)  # (E,2), (E,)

            # Step
            step_keys = jax.random.split(k3, NUM_ENVS)
            next_sts, ped_r, sdc_r, dones, infos = jax.vmap(step)(sts, ped_actions, sdc_actions, step_keys)

            # Auto-reset
            reset_keys = jax.random.split(key, NUM_ENVS)
            fresh = jax.vmap(reset)(reset_keys)
            def sel(f, c):
                if c.ndim == 0: return c
                return jnp.where(dones.reshape([NUM_ENVS]+[1]*(c.ndim-1)), f, c)
            next_sts = jax.tree.map(sel, fresh, next_sts)

            return (next_sts, key), {
                "ped_obs": ped_obs, "sdc_obs": sdc_obs, "crt_obs": crt_obs,
                "ped_actions": ped_actions, "sdc_actions": sdc_actions,
                "ped_lp": ped_lp, "sdc_lp": sdc_lp,
                "ped_rewards": ped_r, "sdc_rewards": sdc_r,
                "values": values, "dones": dones,
            }

        (final, _), rollout = jax.lax.scan(env_step, (states, key), None, length=NUM_STEPS)
        last_crt = jax.vmap(get_critic_obs)(final)
        last_v = jax.vmap(critic.apply, in_axes=(None,0))(cp, last_crt)

        # GAE
        mean_ped_r = rollout["ped_rewards"].mean(axis=-1)
        combined_r = mean_ped_r * 0.5 + rollout["sdc_rewards"] * 0.5
        advs, rets = compute_gae(combined_r, rollout["values"], rollout["dones"], last_v)

        T, E, P = NUM_STEPS, NUM_ENVS, NUM_PEDS
        # Flatten
        p_obs = rollout["ped_obs"].reshape(T*E*P, PED_OBS_SIZE)
        p_act = rollout["ped_actions"].reshape(T*E*P)
        p_lp = rollout["ped_lp"].reshape(T*E*P)
        p_adv = jnp.broadcast_to(advs[...,None], (T,E,P)).reshape(T*E*P)
        p_adv = (p_adv - p_adv.mean()) / (p_adv.std() + 1e-8)

        s_obs = rollout["sdc_obs"].reshape(T*E, SDC_OBS_SIZE)
        s_act = rollout["sdc_actions"].reshape(T*E, 2)
        s_lp = rollout["sdc_lp"].reshape(T*E)
        s_adv = advs.reshape(T*E)
        s_adv = (s_adv - s_adv.mean()) / (s_adv.std() + 1e-8)

        c_obs = rollout["crt_obs"].reshape(T*E, CRITIC_OBS_SIZE)
        c_ret = rets.reshape(T*E)

        def epoch(carry, _):
            pp, sp, cp, po, so, co, key = carry
            key, k1, k2, k3 = jax.random.split(key, 4)

            # Ped update
            pn = T*E*P; pmb = pn // NUM_MINIBATCHES
            perm = jax.random.permutation(k1, pn)
            def p_step(carry, j):
                pp, po = carry
                idx = jax.lax.dynamic_slice(perm, (j*pmb,), (pmb,))
                def loss_fn(pp):
                    logits = jax.vmap(ped_actor.apply, in_axes=(None,0))(pp, p_obs[idx])
                    dist = distrax.Categorical(logits=logits)
                    nlp = dist.log_prob(p_act[idx])
                    ent = dist.entropy()
                    ratio = jnp.exp(nlp - p_lp[idx])
                    clipped = jnp.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
                    return -jnp.minimum(ratio*p_adv[idx], clipped*p_adv[idx]).mean() - ENT_COEF_PED*ent.mean()
                loss, grads = jax.value_and_grad(loss_fn)(pp)
                updates, po = ped_opt.update(grads, po, pp)
                pp = optax.apply_updates(pp, updates)
                return (pp, po), loss
            (pp, po), _ = jax.lax.scan(p_step, (pp, po), jnp.arange(NUM_MINIBATCHES))

            # SDC update
            sn = T*E; smb = sn // NUM_MINIBATCHES
            sperm = jax.random.permutation(k2, sn)
            def s_step(carry, j):
                sp, so = carry
                idx = jax.lax.dynamic_slice(sperm, (j*smb,), (smb,))
                def loss_fn(sp):
                    out = jax.vmap(sdc_actor.apply, in_axes=(None,0))(sp, s_obs[idx])
                    mean, logstd = out[0], out[1]
                    std = jnp.exp(logstd)
                    dist = distrax.MultivariateNormalDiag(loc=mean, scale_diag=std)
                    raw = jnp.arctanh(jnp.clip(s_act[idx], -0.999, 0.999))
                    nlp = dist.log_prob(raw) - jnp.sum(jnp.log(1-s_act[idx]**2+1e-6), axis=-1)
                    ent = dist.entropy()
                    ratio = jnp.exp(nlp - s_lp[idx])
                    clipped = jnp.clip(ratio, 1-CLIP_EPS, 1+CLIP_EPS)
                    return -jnp.minimum(ratio*s_adv[idx], clipped*s_adv[idx]).mean() - ENT_COEF_SDC*ent.mean()
                loss, grads = jax.value_and_grad(loss_fn)(sp)
                updates, so = sdc_opt.update(grads, so, sp)
                sp = optax.apply_updates(sp, updates)
                return (sp, so), loss
            (sp, so), _ = jax.lax.scan(s_step, (sp, so), jnp.arange(NUM_MINIBATCHES))

            # Critic update
            cperm = jax.random.permutation(k3, sn)
            def c_step(carry, j):
                cp, co = carry
                idx = jax.lax.dynamic_slice(cperm, (j*smb,), (smb,))
                def loss_fn(cp):
                    v = jax.vmap(critic.apply, in_axes=(None,0))(cp, c_obs[idx])
                    return VF_COEF * jnp.mean((v - c_ret[idx])**2)
                loss, grads = jax.value_and_grad(loss_fn)(cp)
                updates, co = crt_opt.update(grads, co, cp)
                cp = optax.apply_updates(cp, updates)
                return (cp, co), loss
            (cp, co), _ = jax.lax.scan(c_step, (cp, co), jnp.arange(NUM_MINIBATCHES))

            return (pp, sp, cp, po, so, co, key), None

        (pp, sp, cp, po, so, co, _), _ = jax.lax.scan(
            epoch, (pp, sp, cp, po, so, co, key), None, length=NUM_EPOCHS)

        metrics = {
            "ped_reward": rollout["ped_rewards"].mean(),
            "sdc_reward": rollout["sdc_rewards"].mean(),
            "done_rate": rollout["dones"].mean(),
        }
        return final, pp, sp, cp, po, so, co, metrics

    # Training loop
    print("JIT compiling...\n")
    start = time.time()
    total_steps = 0

    for i in range(1, TOTAL_UPDATES + 1):
        key, subkey = jax.random.split(key)
        states, ped_params, sdc_params, critic_params, ped_os, sdc_os, crt_os, metrics = \
            collect_and_update(states, ped_params, sdc_params, critic_params, ped_os, sdc_os, crt_os, subkey)
        total_steps += NUM_ENVS * NUM_STEPS

        if i % 20 == 0 or i == 1:
            elapsed = time.time() - start
            m = {k: float(v) for k, v in metrics.items()}
            print(f"Update {i:>5}/{TOTAL_UPDATES} | Steps: {total_steps:>12,} | "
                  f"FPS: {total_steps/elapsed:>8,.0f} | "
                  f"Ped R: {m['ped_reward']:>7.3f} | "
                  f"SDC R: {m['sdc_reward']:>7.3f}")

        if i % 500 == 0 or i == TOTAL_UPDATES:
            ckpt = {"ped_params": ped_params, "sdc_params": sdc_params, "critic_params": critic_params}
            path = os.path.join(CHECKPOINT_DIR, f"sdc_single_{i}.pkl")
            with open(path, "wb") as f:
                pickle.dump(jax.device_get(ckpt), f)
            print(f"  Saved: {path}")

    elapsed = time.time() - start
    print(f"\nDone in {elapsed/60:.1f} min. FPS: {total_steps/elapsed:,.0f}")
    final_path = os.path.join(CHECKPOINT_DIR, "sdc_single_agent_final.pkl")
    with open(final_path, "wb") as f:
        pickle.dump(jax.device_get({"ped_params": ped_params, "sdc_params": sdc_params, "critic_params": critic_params}), f)
    print(f"Final: {final_path}")


if __name__ == "__main__":
    train()
