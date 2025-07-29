import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import beta

# ---------- parameters ----------
HS_GRID = np.arange(0, 1.05, 0.05)          # bin edges
alphas   = [0.1,0.5,1.0,2.0,10]                   # the four x values → Beta(x, x)
n_samples = 20_000                          # draw the same number of samples for each curve
rng = np.random.default_rng(42)             # deterministic for easy replication
colors = plt.colormaps['tab10'].colors      # 4 distinct default colors

# ---------- sampling ----------
samples = {
    f"Beta({a},{a})": rng.beta(a, a, size=n_samples)
    for a in alphas
}

# ---------- plotting ----------
fig, ax = plt.subplots(figsize=(8, 5))

for idx, (label, data) in enumerate(samples.items()):
    ax.hist(
        data,
        bins=HS_GRID,
        density=True,              # normalize so all histograms integrate to 1
        histtype='stepfilled',
        alpha=0.35,
        edgecolor=colors[idx],
        facecolor=colors[idx],
        label=label,
    )

# optional: overlay the exact Beta PDF on top of each histogram
x_plot = np.linspace(0, 1, 400)
for idx, a in enumerate(alphas):
    ax.plot(x_plot, beta.pdf(x_plot, a, a),
            color=colors[idx], linewidth=1.5, linestyle='--')

# ---------- cosmetics ----------
ax.set_xlabel(r"$p$")
ax.set_ylabel("Probability density")
ax.set_title("Histograms (and PDFs) of symmetric Beta(x, x) distributions")
ax.legend(frameon=False, fontsize=9)
ax.set_xlim(0, 1)
ax.grid(alpha=0.2)

plt.tight_layout()
plt.show()
