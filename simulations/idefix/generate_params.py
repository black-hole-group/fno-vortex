"""
Generate the (nu, mu) parameter table for the Orszag-Tang vortex simulations.

25 simulations total:
  - 2 hardcoded test cases: nu=mu=5e-5 and nu=mu=3e-4
  - 23 randomly sampled, log-uniformly in [1e-5, 5e-2] for both nu and mu independently

Output: params.csv with columns: sim_id, nu, mu, split

Usage:
  python generate_params.py [--seed SEED]
"""

import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

N_RANDOM = 23
LOG_LOW = np.log10(1e-5)
LOG_HIGH = np.log10(5e-2)

# Two held-out test cases (matching original paper)
TEST_CASES = [
    (5e-5, 5e-5),
    (3e-4, 3e-4),
]

parser = argparse.ArgumentParser(description="Generate simulation parameter table")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed for reproducibility (default: 42)")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)
nu_rand = 10 ** rng.uniform(LOG_LOW, LOG_HIGH, N_RANDOM)
mu_rand = 10 ** rng.uniform(LOG_LOW, LOG_HIGH, N_RANDOM)

rows = []
sim_id = 0

for nu, mu in TEST_CASES:
    rows.append({"sim_id": sim_id, "nu": nu, "mu": mu, "split": "test"})
    sim_id += 1

for nu, mu in zip(nu_rand, mu_rand):
    rows.append({"sim_id": sim_id, "nu": nu, "mu": mu, "split": "train"})
    sim_id += 1

output_path = "params.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sim_id", "nu", "mu", "split"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} parameter sets to {output_path}  (seed={args.seed})")
print(f"  Test  : {sum(1 for r in rows if r['split'] == 'test')}")
print(f"  Train : {sum(1 for r in rows if r['split'] == 'train')}")

# Print summary table
print("\nsim_id  nu            mu            split")
print("-" * 48)
for r in rows:
    print(f"  {r['sim_id']:02d}    {r['nu']:.3e}    {r['mu']:.3e}    {r['split']}")

# Plot mu vs nu
train_rows = [r for r in rows if r["split"] == "train"]
test_rows  = [r for r in rows if r["split"] == "test"]

fig, ax = plt.subplots()

ax.scatter([r["nu"] for r in train_rows], [r["mu"] for r in train_rows],
           color="steelblue", label="train", zorder=3)
ax.scatter([r["nu"] for r in test_rows], [r["mu"] for r in test_rows],
           marker="*", s=150, color="black", label="test", zorder=4)

lim_lo, lim_hi = 10**LOG_LOW, 10**LOG_HIGH
ax.plot([lim_lo, lim_hi], [lim_lo, lim_hi], "k--", lw=1, label=r"$\mu = \nu$")

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlabel(r"$\nu$")
ax.set_ylabel(r"$\mu$")
ax.set_title(f"Parameter distribution (seed={args.seed})")
ax.legend(fontsize=8)
ax.set_aspect("equal")

plot_path = "params_distribution.png"
fig.savefig(plot_path, dpi=150, bbox_inches="tight")
print(f"\nSaved parameter plot to {plot_path}")
