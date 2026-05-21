# Multi-Agent RL for Safe Autonomous Driving Under Pedestrian Behavioral Uncertainty

A JAX implementation of a multi-agent reinforcement learning environment in which a
self-driving car (SDC) and 12 pedestrians are co-trained with Multi-Agent Proximal
Policy Optimization (MAPPO). Pedestrian locomotion is scripted (shortest-path
navigation on a graph), while a reinforcement learning policy controls the high-level
go/wait crossing decision. Whether a pedestrian crosses at a designated crosswalk or
jaywalks across the road is driven by a latent personality trait that is hidden from
the SDC, which makes jaywalking a tunable source of behavioral uncertainty.

The whole environment, training loop, and inference run on the GPU through `jax.vmap`
(parallel environments) and `jax.lax.scan` (rollout collection).

This repository accompanies the paper "Multi-Agent Reinforcement Learning for Safe
Autonomous Driving Under Pedestrian Behavioral Uncertainty", accepted and presented at
the 8th Workshop on Long-term Human Motion Prediction at ICRA 2026. Preprint:
[arXiv:2605.20255](https://arxiv.org/abs/2605.20255).

## Repository layout

```
jax_impl/
  decision_env.py        Environment: map, pedestrian navigation, SDC kinematics, rewards
  train_decisions.py     MAPPO co-training (SDC + pedestrian policies + shared critic)
  train_single_agent.py  Single-agent SDC baseline (pedestrians not co-trained)
pyproject.toml
```

## Requirements

- Python 3.10 or newer
- JAX 0.6.2, Flax 0.10.7, Optax 0.2.8, Distrax, NumPy
- An NVIDIA GPU with CUDA 12 for practical training speed (CPU also works but is slow)

This project uses `uv` for environment and dependency management.

```bash
git clone https://github.com/prakash-aryan/marl-sdc-pedestrian-uncertainty.git
cd marl-sdc-pedestrian-uncertainty
uv venv
source .venv/bin/activate
uv pip install -e .
# for GPU (CUDA 12):
uv pip install -e ".[cuda]"
```

## Training

Run from the repository root.

Co-trained MAPPO (the main method):

```bash
python -m jax_impl.train_decisions
```

Single-agent SDC baseline:

```bash
python -m jax_impl.train_single_agent
```

Checkpoints are written to `checkpoints_jax/` (periodic `decisions_{update}.pkl` and a
final `decisions_final.pkl` containing the pedestrian actor, SDC actor, and critic
parameters).

### Configuration

Core hyperparameters are defined at the top of `jax_impl/train_decisions.py`
(512 parallel environments, 256-step rollouts, 5000 updates, learning rate 3e-4,
discount 0.995, GAE lambda 0.95, PPO clip 0.2, 4 epochs, 8 minibatches).

The environment exposes two knobs through environment variables:

- `MARL_NUM_PEDS` (default 12): number of pedestrian agents.
- `MARL_JW_MULT` (default 0.25): multiplier on the jaywalking probability,
  `P(jaywalk | go) = jaywalking_tendency * MARL_JW_MULT`.

For example, the jaywalking-rate ablation sweeps this multiplier:

```bash
MARL_JW_MULT=0.0  python -m jax_impl.train_decisions
MARL_JW_MULT=0.5  python -m jax_impl.train_decisions
```

## System specifications used in our runs

| Component | Specification |
|---|---|
| GPU | NVIDIA RTX 5070 Ti, 16 GB |
| CPU | AMD Ryzen 7 9700X |
| Memory | 32 GB DDR5 |
| Framework | JAX 0.6.2 with CUDA 12.6, Flax 0.10.7, Optax 0.2.8, Distrax |

With these settings the training loop runs at roughly 558,000 environment steps per
second, completing 5000 updates (about 6.55 x 10^8 environment steps in total) in
approximately 20 minutes.

## Citation

If you use this code, please cite:

```bibtex
@misc{aryan2026multiagentreinforcementlearningsafe,
      title={Multi-Agent Reinforcement Learning for Safe Autonomous Driving Under Pedestrian Behavioral Uncertainty}, 
      author={Prakash Aryan and Kaushik Raghupathruni and Timo Kehrer and Sebastiano Panichella},
      year={2026},
      eprint={2605.20255},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2605.20255}, 
}
```
