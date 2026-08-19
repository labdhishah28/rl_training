# Optimal Execution Engine: Phase 4 Documentation

## Multi-Regime Dummy Data & Deep Pre-Training

**Phase 4** upgrades the training data pipeline from a single random walk to a
**four-regime synthetic dataset**, and significantly increases both data volume
and training timesteps before any real data is introduced.

---

## Why More Dummy Training First?

Before touching real market data, the agent needs a robust *prior* — a policy
that has already seen a diverse range of price behaviours.  Training on a
single random walk (Phase 3) can cause the agent to overfit to one style of
market (e.g. low-volatility diffusion) and perform poorly when conditions
change.

The analogy in supervised learning: training on only MNIST digits and then
expecting the model to classify street signs.

---

## The Four Market Regimes

Each regime occupies ≈ 25 % of the 50,000-step dataset:

| Regime | Price Process | Spread | What It Teaches the Agent |
|---|---|---|---|
| **Trending Up** | GBM with positive drift μ = +0.02 | Tight | Execute *slowly* — price is moving in your favour (sell later = higher price) |
| **Trending Down** | GBM with negative drift μ = −0.02 | Tight | Execute *quickly* — delay increases market risk |
| **Mean-Reverting** | Ornstein-Uhlenbeck θ = 0.10 | Tight | Be *patient* — the price will snap back to its mean |
| **High Volatility** | GBM with σ = 0.15, μ = 0 | Wide | Pace execution *carefully* — wide spreads and unpredictable moves |

### Ornstein-Uhlenbeck Process

The mean-reverting regime is modelled by:

$$
dX_t = \theta(\mu - X_t)\,dt + \sigma\,dW_t
$$

where:
- θ (theta) = 0.10 — speed of mean reversion  
- μ (mu) = price at start of the segment — long-run mean  
- σ = 0.03 — noise amplitude  

---

## Training Improvements

| Parameter | Phase 3 | Phase 4 |
|---|---|---|
| Dataset size | 5,000 steps | **50,000 steps** |
| MLP timesteps | 50,000 | **300,000** |
| LSTM training | Disabled | **Enabled — 200,000 timesteps** |
| Parallel envs (MLP) | 8 | 8 |
| Parallel envs (LSTM) | — | 4 |
| Checkpoint frequency | 10,000 | 20,000 |

The LSTM agent trains on fewer parallel environments to maintain the recurrent
hidden state integrity required by `RecurrentPPO`.

---

## Files Changed

| File | Change |
|---|---|
| `train.py` | `generate_dummy_data()` → `generate_regime_data()`, 300k MLP + 200k LSTM |
| `regime_market_data.csv` | New output — 50k row synthetic dataset (auto-generated) |
| `ppo_execution_model_mlp.zip` | Re-trained MLP weights |
| `ppo_execution_model_lstm.zip` | Re-trained LSTM weights |
| `trajectories.png` | Updated — shows MLP vs LSTM vs TWAP vs AC |

---

## Expected Outcomes

After Phase 4 training, both agents should:

1. **Beat TWAP** in total cash flow across all four regimes
2. **Track the Almgren-Chriss curve closely** in trending and mean-reverting markets
3. **Deviate from AC in the high-vol regime** by trading more conservatively
   when spreads are wide — this is the RL agent's adaptive advantage

---

> [!NOTE]
> The `generate_dummy_data()` function still exists as a backward-compatible
> alias that calls `generate_regime_data()`.  Existing scripts using the old
> function name will continue to work.
