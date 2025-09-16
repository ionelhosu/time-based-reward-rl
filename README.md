# TIRE: Time-based Intrinsic Reward for Exploration

This repository contains a minimal, clean implementation of **TIRE** for Atari games with both **PPO** and **DQN** backbones.

> Paper: *Time-Based Intrinsic Reward for Enhanced Exploration in Atari Games*  
> Code reference (paper link): https://github.com/ionelhosu/time-based-reward-rl

## Method (TL;DR)

TIRE provides an intrinsic reward to states that are first-visited **earlier** within the episode:
\[
r^{\mathrm{int}}_t = \beta \exp\!\left(-\tfrac{\tau}{\kappa}\right) \cdot \mathbb{1}[\text{novel}],
\]
where \(\tau\) is the elapsed episode steps, \(\beta\) scales the signal, and \(\kappa\) controls temporal decay.
Novelty is gated via per-episode embeddings from a frozen random CNN with an \(\ell_2\) threshold.

## Installation

```bash
git clone <this-repo-or-your-fork>
cd tire_rl
python -m venv .venv && source .venv/bin/activate  # or use conda
pip install -r requirements.txt
```

> Gymnasium’s Atari package requires accepting the ROM license: `pip install "gymnasium[atari,accept-rom-license]"`

## Quick Start

### PPO (vectorized)

```bash
python -m tire.scripts.train_ppo
```

### DQN (single-env example)

```bash
python -m tire.scripts.train_dqn
```

Both scripts default to **sticky actions**, **frame-stack=4**, and **84×84 grayscale** preprocessing.

## Project Structure

```
tire_rl/
├── LICENSE
├── requirements.txt
├── configs/
│   ├── atari_dqn.yaml
│   └── atari_ppo.yaml
├── tire/
│   ├── __init__.py
│   ├── algo/
│   │   ├── dqn.py
│   │   └── ppo.py
│   ├── envs/
│   │   └── wrappers.py
│   ├── models/
│   │   ├── dqn.py
│   │   ├── encoder.py
│   │   └── policy_value.py
│   ├── utils/
│   │   └── replay.py
│   └── tire_bonus.py
└── scripts/
    ├── train_dqn.py
    └── train_ppo.py
```

## Configuration

Edit files in `configs/` or the parameters in the training scripts. Key TIRE hyperparameters:
- `beta`: intrinsic scale (default 0.1)
- `kappa`: temporal decay (default 500)
- `lam`: reward mixing weight (default 0.5)
- `delta`: novelty threshold in embedding space (default 0.5)
- `max_ep_memory`: per-episode novelty buffer cap (default 4096)

## Notes

- The implementation is intentionally compact for readability. For 2B-frame runs, plug the TIRE bonus into your distributed runner.
- You can switch the decay from exponential to hyperbolic or sqrt by changing `decay` in `TIREConfig`.
- Evaluation follows the Atari protocol with sticky actions; rewards logged in the examples are **extrinsic** for clarity.

## Citation

If you find this repository useful, please cite:

```bibtex
@inproceedings{hosu2025tire,
  title     = {Time-based Intrinsic Reward for Exploration},
  author    = {Hosu, Ionel-Alexandru and Rebedea, Traian and Trăușan-Matu, Ștefan},
  booktitle = {U.P.B. Scientific Bulletin},
  year      = {2025}
}
```

## License

MIT
