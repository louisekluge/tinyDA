"""
Aggregate diag_run*.npz produced by mlda_diagnostics.py and draw the
diagnostic plots. Safe to run on the cluster (Agg backend, no display).

Usage:
    python plot_diagnostics.py --indir . --out mlda_diagnostics.png
"""

import argparse
import glob
import json
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--indir", default=".")
p.add_argument("--out", default="mlda_diagnostics.png")
p.add_argument("--csv", default="mlda_diagnostics.csv")
args = p.parse_args()

files = sorted(glob.glob(os.path.join(os.path.expanduser(args.indir), "diag_run*.npz")))
if not files:
    raise FileNotFoundError(f"no diag_run*.npz in {args.indir}")
runs = [np.load(f, allow_pickle=True) for f in files]
print(f"{len(runs)} repetitions: {[os.path.basename(f) for f in files]}")

# sanity: every repetition must target the same posterior
maps = [r["MAP"] for r in runs]
if not all(np.array_equal(maps[0], m) for m in maps):
    print("WARNING: MAP differs across runs -- the repetitions are NOT the same problem.")

meta = json.loads(str(runs[0]["meta"]))
print(f"proposal={meta['proposal']}  levels={meta['n_levels']}  "
      f"iterations={int(runs[0]['iterations'])}  nsub={int(runs[0]['subchain_length'])}")

# ---- collect ------------------------------------------------------------

acc_keys = sorted({k for r in runs for k in r.files if k.startswith("acc__")})
ess_keys = sorted({k for r in runs for k in r.files if k.startswith("ess__")})

acc = {k: np.array([float(r[k]) for r in runs if k in r.files]) for k in acc_keys}
runtimes = np.array([float(r["runtime_seconds"]) for r in runs])

# ESS: one row per repetition, one column per parameter
ess = {}
for k in ess_keys:
    rows = [np.atleast_1d(r[k]) for r in runs if k in r.files]
    if rows and all(len(x) == len(rows[0]) for x in rows):
        ess[k] = np.vstack(rows)

# ---- report -------------------------------------------------------------

lines = ["quantity,mean,std,min,max,n"]
print("\n--- acceptance rates across repetitions ---")
for k, v in acc.items():
    name = k[len("acc__"):]
    print(f"{name:50s} {v.mean():.3f} +/- {v.std():.3f}   [{v.min():.3f}, {v.max():.3f}]")
    lines.append(f"acc:{name},{v.mean()},{v.std()},{v.min()},{v.max()},{len(v)}")

print("\n--- ESS across repetitions (pooled over parameters) ---")
for k, m in ess.items():
    name = k[len("ess__"):]
    flat = m.ravel()
    print(f"{name:50s} {flat.mean():8.0f} +/- {flat.std():7.0f}  "
          f"[{flat.min():7.0f}, {flat.max():7.0f}]")
    lines.append(f"ess:{name},{flat.mean()},{flat.std()},{flat.min()},{flat.max()},{flat.size}")

print(f"\nruntime: {runtimes.mean()/60:.1f} +/- {runtimes.std()/60:.1f} min")
lines.append(f"runtime_min,{runtimes.mean()/60},{runtimes.std()/60},"
             f"{runtimes.min()/60},{runtimes.max()/60},{len(runtimes)}")

with open(args.csv, "w") as fh:
    fh.write("\n".join(lines) + "\n")

# ---- plots --------------------------------------------------------------

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# (1) acceptance rate per level
ax = axes[0, 0]
if acc:
    names = [k[len("acc__"):].split(".")[-1] for k in acc]
    ax.boxplot(list(acc.values()), tick_labels=names, showmeans=True)
    for i, v in enumerate(acc.values(), start=1):
        ax.plot(np.full_like(v, i) + np.random.uniform(-0.06, 0.06, len(v)),
                v, ".", color="tab:blue", alpha=0.6)
ax.set_ylabel("acceptance rate")
ax.set_title(f"Acceptance rate per level ({len(runs)} repetitions)")
ax.tick_params(axis="x", rotation=20)
ax.grid(alpha=0.3)

# (2) ESS per parameter, finest level
ax = axes[0, 1]
lvl = meta["n_levels"] - 1
fine_keys = sorted(k for k in ess if f"level{lvl}_" in k)
if fine_keys:
    m = np.hstack([ess[k] for k in fine_keys])          # (n_runs, n_params)
    ax.boxplot([m[:, j] for j in range(m.shape[1])],
           tick_labels=[k.split("_")[-1] for k in fine_keys], showmeans=True)
    ax.set_title(f"ESS per parameter -- level {lvl}")
else:
    m = None
    ax.set_title("ESS per parameter (no fine-level data)")
ax.set_ylabel("ESS")
ax.grid(alpha=0.3)

# (3) ESS vs acceptance rate
ax = axes[1, 0]
if acc and m is not None:
    acc_fine = [v for k, v in acc.items() if "fine" in k.lower()]
    a = acc_fine[0] if acc_fine else list(acc.values())[-1]
    e = m.min(axis=1)
    n = min(len(a), len(e))
    ax.scatter(a[:n], e[:n], c=runtimes[:n] / 60, cmap="viridis")
    cb = plt.colorbar(ax.collections[0], ax=ax)
    cb.set_label("runtime (min)")
    ax.set_xlabel("fine-level acceptance rate")
    ax.set_ylabel("min ESS over parameters")
ax.set_title("Mixing vs acceptance")
ax.grid(alpha=0.3)

# (4) ESS per minute
ax = axes[1, 1]
if m is not None:
    eff = m.min(axis=1) / (runtimes / 60)
    ax.hist(eff, bins=max(5, len(runs) // 2), color="tab:green", alpha=0.8)
    ax.set_xlabel("min ESS per minute")
    ax.set_ylabel("repetitions")
ax.set_title("Sampling efficiency")
ax.grid(alpha=0.3)

fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"\nwrote {args.out} and {args.csv}")