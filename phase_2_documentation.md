# Optimal Execution Engine: Phase 2 Documentation

This document explains the technical logic, mathematics, and implementation strategy for **Phase 2: The Baseline Models Layer**.

## Overview of Phase 2

Before we train a Deep Reinforcement Learning agent, we must establish **baselines**. Baselines serve two critical purposes:
1. **Benchmarking**: How do we know if our AI is actually good? We need to compare its execution costs against industry-standard, deterministic algorithms. If the RL agent loses to a simple math formula, it's not ready for production.
2. **Reward Shaping / Reference Schedules**: Sometimes, instead of letting the RL agent figure everything out from scratch, we give it a mathematical baseline (like TWAP) and ask it to "trade around" the baseline using its microstructural signals (like OBI).

In Phase 2, we will build a `baselines.py` module containing three core algorithms.

---

## 1. The Naive Baselines: TWAP & VWAP

These are the most common algorithmic execution strategies used in traditional finance. They do not use Deep Learning; they just follow a strict schedule.

### A. TWAP (Time-Weighted Average Price)
The TWAP strategy is the simplest possible approach to liquidation. It ignores price, volume, and order book dynamics entirely. It simply slices the total inventory into equal chunks and executes them at regular time intervals.

- **Formula**: $v_t = \frac{X}{T}$
  - $v_t$ = Volume to trade at step $t$
  - $X$ = Total Initial Inventory (e.g., 10,000 shares)
  - $T$ = Total Time Horizon (e.g., 100 steps)
- **Why we need it**: It provides a bare-minimum benchmark. It minimizes Temporary Market Impact (because orders are small and spread out) but takes on massive Market Risk (because the price might crash while you are waiting).

### B. VWAP (Volume-Weighted Average Price)
The VWAP strategy is smarter than TWAP. Instead of dividing the order evenly across *time*, it divides the order across historical *volume profiles*.

- **Concept**: If historical data shows that 30% of daily volume happens in the first 10 minutes of the day, the VWAP algorithm will execute 30% of our order in the first 10 minutes.
- **Why we need it**: It minimizes the footprint of our trades relative to the rest of the market. It is the gold standard benchmark for institutional execution.

---

## 2. The Analytical Baseline: Almgren-Chriss Closed-Form Solver

This is the crown jewel of the Baseline Layer. The **Almgren-Chriss (2000)** model provides an exact mathematical formula for the *optimal* trading trajectory, assuming a specific trade-off between risk and cost.

### The Core Dilemma (The Efficient Frontier)
When liquidating a massive position, you face two conflicting forces:
1. **Market Risk (Variance)**: If you trade slowly over a long time (like TWAP), the fundamental price of the asset might crash before you finish selling. You want to sell *fast* to avoid this risk.
2. **Market Impact (Cost)**: As modeled in Phase 1, if you sell *fast*, your large orders cause temporary slippage ($\kappa$) and permanently push the price down ($\gamma$). You want to sell *slow* to avoid these costs.

### The Mathematics
The Almgren-Chriss solver balances these forces by minimizing a Utility function:
$$ \text{Utility} = \text{Expected Execution Costs} + \lambda \cdot \text{Variance of Costs} $$

Where $\lambda$ (Lambda) is the **Risk Aversion Parameter**:
- If $\lambda = 0$: The trader doesn't care about risk, only market impact. The solution becomes a straight line (TWAP).
- If $\lambda > 0$: The trader is afraid the price will crash. The solution becomes a curved trajectory where they dump a huge chunk of shares immediately, and trade slower as time goes on.

### The Output (The Trading Trajectory)
We will implement the closed-form calculus solution in Python. For a given time $t$, the optimal amount of inventory $x_t$ you should still be holding is calculated using hyperbolic functions (`cosh` and `sinh`):

$$ x_t = X \cdot \frac{\sinh(\tilde{\kappa} (T-t))}{\sinh(\tilde{\kappa} T)} $$

Where $\tilde{\kappa}$ is a variable derived from our impact parameters ($\kappa, \gamma$) and risk aversion ($\lambda$).

**How this connects to Phase 3 (RL):**
The RL agent will be trained in the environment we built in Phase 1. It will have access to the Almgren-Chriss trajectory. However, because Almgren-Chriss assumes liquidity is constant and ignores the Order Book Imbalance (OBI), **the RL agent should be able to beat it** by aggressively trading when OBI is favorable and pausing when OBI is unfavorable!
