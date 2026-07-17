import numpy as np
import json

# ---------------- Parameters ----------------
TICKER = "NIFTY 50"
S0 = 24343.0          # current price
sigma_annual = 0.09    # annual volatility
mu_annual = 0.09       # annual drift
n_sims = 10000
n_days = 30            # trading days horizon

np.random.seed(42)

# ---------------- Convert annual -> daily ----------------
dt = 1.0
mu_daily = mu_annual / 252
sigma_daily = sigma_annual / np.sqrt(252)

# ---------------- Simulate GBM paths ----------------
# Z shape: (n_sims, n_days)
Z = np.random.standard_normal((n_sims, n_days))

daily_returns = (mu_daily - 0.5 * sigma_daily**2) * dt + sigma_daily * np.sqrt(dt) * Z

log_paths = np.cumsum(daily_returns, axis=1)
paths = S0 * np.exp(log_paths)

# prepend the starting price as day 0
paths = np.hstack([np.full((n_sims, 1), S0), paths])

final_prices = paths[:, -1]

# ---------------- Statistics ----------------
expected_price = final_prices.mean()
median_price = np.median(final_prices)
p5 = np.percentile(final_prices, 5)
p95 = np.percentile(final_prices, 95)
p_gain = (final_prices > S0).mean() * 100
p_loss = (final_prices < S0).mean() * 100
std_final = final_prices.std()

print(f"===== Monte Carlo GBM Simulation: {TICKER} =====")
print(f"Current Price       : {S0:,.2f}")
print(f"Annual Drift (mu)   : {mu_annual*100:.2f}%")
print(f"Annual Volatility   : {sigma_annual*100:.2f}%")
print(f"Simulations         : {n_sims:,}")
print(f"Horizon (days)      : {n_days}")
print("-" * 50)
print(f"Expected Price      : {expected_price:,.2f}")
print(f"Median Price        : {median_price:,.2f}")
print(f"Std Dev (final)     : {std_final:,.2f}")
print(f"P5  (5th pct)       : {p5:,.2f}")
print(f"P95 (95th pct)      : {p95:,.2f}")
print(f"P(gain)             : {p_gain:.2f}%")
print(f"P(loss)             : {p_loss:.2f}%")

# ---------------- Export data for HTML chart ----------------
# Sample a subset of paths for plotting (too many paths -> heavy HTML)
n_plot_paths = 150
sample_idx = np.random.choice(n_sims, n_plot_paths, replace=False)
sample_paths = paths[sample_idx].round(2).tolist()

# Percentile band across time (for fan chart)
pct_bands = {
    "p5": np.percentile(paths, 5, axis=0).round(2).tolist(),
    "p25": np.percentile(paths, 25, axis=0).round(2).tolist(),
    "p50": np.percentile(paths, 50, axis=0).round(2).tolist(),
    "p75": np.percentile(paths, 75, axis=0).round(2).tolist(),
    "p95": np.percentile(paths, 95, axis=0).round(2).tolist(),
}

# Histogram of final prices
hist_counts, hist_edges = np.histogram(final_prices, bins=40)

output = {
    "ticker": TICKER,
    "params": {
        "S0": S0,
        "mu_annual": mu_annual,
        "sigma_annual": sigma_annual,
        "n_sims": n_sims,
        "n_days": n_days,
        "mu_daily": mu_daily,
        "sigma_daily": sigma_daily,
    },
    "stats": {
        "expected_price": round(expected_price, 2),
        "median_price": round(median_price, 2),
        "std_final": round(std_final, 2),
        "p5": round(p5, 2),
        "p95": round(p95, 2),
        "p_gain": round(p_gain, 2),
        "p_loss": round(p_loss, 2),
    },
    "sample_paths": sample_paths,
    "pct_bands": pct_bands,
    "histogram": {
        "counts": hist_counts.tolist(),
        "edges": hist_edges.round(2).tolist(),
    },
}

with open("mc_output.json", "w") as f:
    json.dump(output, f)

print("\nData exported to mc_output.json")


import matplotlib.pyplot as plt

# paths = tumhari simulation matrix hai
# shape: (days+1, simulations)

plt.figure(figsize=(10,6))

# Sirf 100 random simulation paths dikhana
for i in range(100):
    plt.plot(paths[i], alpha=0.3)
    print(paths.shape)
   

plt.title("Monte Carlo Simulation - NIFTY 50")
plt.xlabel("Days")
plt.ylabel("Price")
plt.grid(True)

plt.show()

