"""
Generate the (nu, mu) parameter table for the Orszag-Tang vortex simulations.

Default (50 simulations, --nval 6):
  - 2 hardcoded test cases: nu=mu=5e-5 and nu=mu=3e-4
  - 6 randomly chosen validation cases (log-uniform in [1e-5, 5e-2])
  - 42 training cases (remainder)

Output: params.csv with columns: sim_id, nu, mu, split

Usage:
  python generate_params.py [--seed SEED] [--nsims N] [--nval K]
"""

import argparse
import numpy as np
import csv
import matplotlib.pyplot as plt

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
parser.add_argument("--nsims", type=int, default=50,
                    help="Total number of simulations (default: 50)")
parser.add_argument("--nval", type=int, default=6,
                    help="Number of validation simulations (default: 6)")
args = parser.parse_args()

n_random = args.nsims - len(TEST_CASES)
if n_random < 0:
    parser.error(
        f"--nsims must be >= {len(TEST_CASES)} (number of hardcoded test cases)"
    )
if not 0 <= args.nval <= n_random:
    parser.error(
        f"--nval must be between 0 and {n_random} "
        f"(number of non-test simulations)"
    )

rng = np.random.default_rng(args.seed)
nu_rand = 10 ** rng.uniform(LOG_LOW, LOG_HIGH, n_random)
mu_rand = 10 ** rng.uniform(LOG_LOW, LOG_HIGH, n_random)

val_indices = set(rng.choice(n_random, size=args.nval, replace=False))

rows = []
sim_id = 0

for nu, mu in TEST_CASES:
    rows.append({"sim_id": sim_id, "nu": nu, "mu": mu, "split": "test"})
    sim_id += 1

for idx, (nu, mu) in enumerate(zip(nu_rand, mu_rand)):
    split = "val" if idx in val_indices else "train"
    rows.append({"sim_id": sim_id, "nu": nu, "mu": mu, "split": split})
    sim_id += 1

output_path = "params.csv"
with open(output_path, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["sim_id", "nu", "mu", "split"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote {len(rows)} parameter sets to {output_path}  (seed={args.seed})")
print(f"  Test  : {sum(1 for r in rows if r['split'] == 'test')}")
print(f"  Val   : {sum(1 for r in rows if r['split'] == 'val')}")
print(f"  Train : {sum(1 for r in rows if r['split'] == 'train')}")

# Print summary table
print("\nsim_id  nu            mu            split")
print("-" * 48)
for r in rows:
    print(f"  {r['sim_id']:02d}    {r['nu']:.3e}    {r['mu']:.3e}    {r['split']}")

# Plot mu vs nu
train_rows = [r for r in rows if r["split"] == "train"]
val_rows   = [r for r in rows if r["split"] == "val"]
test_rows  = [r for r in rows if r["split"] == "test"]

fig, ax = plt.subplots()

ax.scatter([r["nu"] for r in train_rows], [r["mu"] for r in train_rows],
           color="steelblue", label="train", zorder=3)
ax.scatter([r["nu"] for r in val_rows], [r["mu"] for r in val_rows],
           color="darkorange", label="val", zorder=4)
ax.scatter([r["nu"] for r in test_rows], [r["mu"] for r in test_rows],
           marker="*", s=150, color="black", label="test", zorder=5)

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
