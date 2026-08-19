# Optimal Execution Engine: Phase 3 Documentation

This document explains the technical logic, mathematics, and implementation strategy for **Phase 3: The RL Agents Layer**.

## Overview of Phase 3

Having established our Baseline Models (TWAP, VWAP, and the analytical Almgren-Chriss solver) in Phase 2, we now transition to the core intelligent component of our system: the Reinforcement Learning (RL) agents.

The objective of Phase 3 is to train Deep RL agents that can intelligently balance Market Risk and Market Impact in the simulated environment from Phase 1, outperforming the static baselines from Phase 2.

### Why RL?
While the Almgren-Chriss (AC) model provides an optimal closed-form solution, it makes several simplistic assumptions:
1. It assumes liquidity is constant over time.
2. It completely ignores microstructure signals, specifically the Order Book Imbalance (OBI).

In the real world, the order book state is dynamic. By leveraging RL, the agent can learn to dynamically adjust its trading schedule based on real-time OBI, trading more aggressively when liquidity is favorable and pausing when it is unfavorable, thereby achieving a better overall execution price than the AC solver.

---

## 1. Environment & State Representation

The RL agent interacts with the `ExecutionEnv` via a Markov Decision Process (MDP):

### Observation Space
At each timestep $t$, the agent receives a state vector $S_t$:
- **Time Remaining Ratio**: $t / T$
- **Inventory Remaining Ratio**: $x_t / X$
- **Adjusted Mid Price**: Scaled to normalize inputs.
- **Spread**: The current bid-ask spread.
- **Order Book Imbalance (OBI)**: A crucial microstructure signal in $[-1, 1]$.

### Action Space
The agent outputs an action $A_t \in [0, 1]$, representing the *fraction* of the remaining inventory it chooses to liquidate at the current timestep.

### Reward Function
The reward $R_t$ is the normalized cash flow generated from the trade at timestep $t$. A massive penalty is applied if the agent fails to liquidate the entire inventory by the final timestep, ensuring it respects the hard time horizon constraint.

---

## 2. RL Algorithms: PPO

We utilize **Proximal Policy Optimization (PPO)**, a state-of-the-art policy gradient method that strikes an excellent balance between sample efficiency and ease of tuning.

We implement two variants of the agent:
1. **MLP Agent (`MlpPolicy`)**: A standard Multi-Layer Perceptron network that relies purely on the current state observation. Since the environment state (`[Time Remaining, Inventory Remaining, Mid-Price, Spread, OBI]`) is fully observable and contains all immediate microstructural pressure info, this simple feed-forward network is extremely fast to train and perfectly suited for high-frequency execution tasks.
---

## 3. Training and Evaluation

### Vectorized Training
To significantly accelerate training, we utilize Stable Baselines 3's `make_vec_env` to run multiple environment instances in parallel. This allows the PPO agent to collect diverse experiences more rapidly, stabilizing the gradient updates and reducing wall-clock training time.

### Evaluation Metrics
We evaluate the RL agents against our Phase 2 baselines:
- **Total Cash Flow**: Did the RL agent generate more cash than TWAP or AC?
- **Execution Trajectory**: How does the agent's inventory depletion curve compare to the AC trajectory? We expect the RL agent to "trade around" the AC curve, deviating based on OBI signals.

By the end of Phase 3, we expect our trained agents (especially the LSTM variant) to demonstrate a tangible edge over traditional algorithmic execution methods, validating the efficacy of Deep RL in High-Frequency Trading execution.
