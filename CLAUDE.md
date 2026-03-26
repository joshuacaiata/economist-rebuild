# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Multi-agent reinforcement learning simulation of an economy. Mobile agents gather resources (wood, stone), trade via an order book, and build houses on a grid map. A planner agent sets tax policy across brackets, and a bank agent controls monetary policy (interest rates, money supply). All agents are trained with PPO using LSTM-based recurrent policies.

## Commands

```bash
# Install dependencies
pip install -r requirements.txt

# Run full three-phase training (default)
python main.py --config config.yaml

# Run a specific phase
python main.py --config config.yaml --phase 1        # Phase 1: mobile agents only
python main.py --config config.yaml --phase 3        # Phase 3: joint training with bank

# Two-phase training (no bank agent)
python main.py --config config.yaml --training-type two_phase --phase 1
python main.py --config config.yaml --training-type two_phase --phase 2

# Enable GPU
python main.py --config config.yaml --use-gpu

# Evaluate trained models
python eval_models.py
python run_random.py
```

## Architecture

### Training Phases (Curriculum Learning)

Training is sequential — each phase builds on the previous:

- **Phase 1**: Multiple mobile agents (default 4) train in a tax-free, no-monetary-policy environment. Planner and bank agents exist but are wrapped as inert (`ZeroTaxPlannerWrapper`, `ZeroBankPolicyWrapper` in `vectorized_env.py`) — all tax rates are zero and bank actions are no-ops. Only mobile agent policies are updated.
- **Phase 2**: Planner agent trains to set tax rates while mobile agents continue adapting. Mobile agent weights from Phase 1 are loaded.
- **Phase 3**: Bank agent is introduced. All three agent types train jointly toward equilibrium.

`TwoPhaseTrainer` runs Phases 1-2. `ThreePhaseTrainer` runs all three (Phases 1&2 combined, then Phase 3).

### Agent Policies (in `training/`)

- **MobileAgentPolicy**: CNN (processes local map view with 5 channels) → FC → LSTM → action head + value head. Action space: 5 movement/gather/build actions + trading actions (4 × max_order_price+1).
- **PlannerPolicy**: FC → LSTM → action head + value head. Observes market data, agent inventories, tax info. Outputs tax rates per bracket.
- **BankPolicy**: FC → LSTM → action head + value head. Observes inflation, interest rate, money supply. Outputs interest rate and monetary injection.

### Environment (`environments/`)

- **EconomyEnv** (`env.py`): Grid-based environment. Manages agent positions, resource tiles, building, tax collection (every `tax_period_length` steps), and bank policy.
- **TradingSystem** (`trading_system.py`): Order book matching buy/sell orders for wood and stone between agents.
- **VectorizedEnv** (`training/vectorized_env.py`): Runs `n_envs` (default 8) parallel `EconomyEnv` instances via multiprocessing with pipe-based communication.

### Configuration

All hyperparameters, map settings, and network architectures are in YAML config files. `config.yaml` is the primary config. The config dict is passed through to trainers, environments, and policies.

### Key Training Details

- Bank action bounds are **annealed** — interest rate and monetary injection limits gradually expand over `annealing_duration` updates during Phase 3.
- Entropy weight decays during training to shift from exploration to exploitation.
- Early stopping uses utility-based patience: training stops when mean utility hasn't improved for `utility_patience` updates (within `utility_tolerance`).
- Trained model weights are saved to `networks/` as `.pth` files, named by agent type and phase.
