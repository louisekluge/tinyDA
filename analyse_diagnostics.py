"""Sampler efficiency across the mlda_diagnostics.py config sweep.

These runs carry no QoI chains, so there is no variance-reduction
decomposition to do here -- the question they answer is narrower and more
practical: for a fixed posterior, which sampler configuration buys the most
independent samples per second, and does the multilevel machinery beat plain
Metropolis-Hastings at all?

Iteration counts differ between configs (nsub20 runs 8000 where the others
run 20000), so raw ESS is not comparable and everything is normalised by
wall-clock runtime. The `single` config is the baseline every speedup is
measured against.

ESS is reported as the minimum over parameters, not the mean. A chain is only
as converged as its worst-mixing direction, and in this posterior the tightly
constrained parameters mix far more slowly than the rest.

    python scripts/analyse_diagnostics.py --indir results
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--indir", default=os.path.join(_HERE, "results"))
p.add_argument("--out", default=os.path.join(_HERE, "results", "diagnostics.png"))
p.add_argument("--csv", default=os.path.join(_HERE, "results", "diagnostics.csv"))
p.add_argument("--expect-reps", type=int, default=6)
p.add_argument("--baseline", default="single",
               help="config that speedups are measured against")
args = p.parse_args()

indir = os.path.expanduser(args.indir)
files = sorted(glob.glob(os.path.join(indir, "diag_*.npz")))
if not files:
    raise SystemExit(f"no diag_*.npz in {indir}")

runs = defaultdict(dict)
unreadable = []
for path in files:
    m = re.match(r"^diag_(.+)_rep(\d+)\.npz$", os.path.basename(path))
    if not m:
        unreadable.append((os.path.basename(path), "unexpected filename"))
        continue
    try:
        d = np.load(path, allow_pickle=True)
        meta = json.loads(str(d["meta"]))
    except Exception as exc:
        unreadable.append((os.path.basename(path), f"{type(exc).__name__}: {exc}"))
        continue
    runs[meta["config"]][int(m.group(2))] = (d, meta)

print(f"{len(files)} files in {indir}, {len(runs)} configs")
if unreadable:
    print("\nUNREADABLE")
    for name, why in unreadable:
        print(f"  {name}: {why}")

missing = []
for name in sorted(runs):
    gaps = sorted(set(range(args.expect_reps)) - set(runs[name]))
    if gaps:
        missing.append(f"{name}: {len(runs[name])}/{args.expect_reps} reps, "
                       f"missing {', '.join(str(g) for g in gaps)}")
if missing:
    print("\nINCOMPLETE CONFIGS")
    for line in missing:
        print(f"  - {line}")


# ---------------------------------------------------------------------------

def finest_ess(d, meta):
    """ESS per parameter on the finest level of this chain.

    Levels are labelled level0..level{n-1} in chain order regardless of which
    model levels the config actually used, so the finest is always the last.
    """
    label = f"level{meta['n_levels'] - 1}"
    values = []
    for key in d.files:
        if key.startswith(f"ess__{label}_"):
            values.append(float(np.ravel(d[key])[0]))
    return np.array(values)


summary = {}
for name in sorted(runs):
    reps = runs[name]
    meta0 = next(iter(reps.values()))[1]
    per_rep = defaultdict(list)
    for d, meta in reps.values():
        ess = finest_ess(d, meta)
        rt = float(d["runtime_seconds"])
        if ess.size == 0 or rt <= 0:
            continue
        per_rep["ess_min"].append(ess.min())
        per_rep["ess_median"].append(float(np.median(ess)))
        per_rep["runtime"].append(rt)
        per_rep["rate"].append(ess.min() / rt)
    if not per_rep["rate"]:
        print(f"  {name}: no usable reps (no ESS or no runtime)")
        continue
    arr = {k: np.array(v) for k, v in per_rep.items()}
    summary[name] = dict(
        n=arr["rate"].size,
        levels=meta0["levels"],
        nsub=meta0["nsub"],
        randomize=meta0["randomize"],
        iterations=meta0["iterations"],
        runtime=float(np.median(arr["runtime"])),
        ess_min=float(np.mean(arr["ess_min"])),
        ess_median=float(np.mean(arr["ess_median"])),
        rate=float(np.mean(arr["rate"])),
        # spread of the mean over reps, which is what a difference between
        # configs has to clear to mean anything
        rate_sem=float(arr["rate"].std(ddof=1) / np.sqrt(arr["rate"].size))
        if arr["rate"].size > 1 else float("nan"),
    )

if not summary:
    raise SystemExit("no config produced usable ESS and runtime")

base = summary.get(args.baseline)
if base is None:
    print(f"\nbaseline {args.baseline!r} not present; speedups omitted")

print("\n=== efficiency on the finest level " + "=" * 52)
print(f"{'config':15s}{'levels':>10s}{'nsub':>6s}{'rand':>6s}{'iters':>8s}"
      f"{'n':>4s}{'min (s)':>10s}{'ESS min':>10s}{'ESS med':>10s}"
      f"{'ESS/s':>10s}{'speedup':>10s}")
for name in sorted(summary, key=lambda k: -summary[k]["rate"]):
    s = summary[name]
    speed = (f"{s['rate'] / base['rate']:10.2f}" if base and base["rate"] > 0
             else f"{'n/a':>10s}")
    print(f"{name:15s}"
          f"{'|'.join(str(x) for x in s['levels']):>10s}"
          f"{s['nsub']:6d}{str(s['randomize'])[0]:>6s}{s['iterations']:8d}"
          f"{s['n']:4d}{s['runtime']/60:10.1f}{s['ess_min']:10.0f}"
          f"{s['ess_median']:10.0f}{s['rate']:10.4f}{speed}")

worst = min(summary.values(), key=lambda s: s["ess_min"] / max(s["ess_median"], 1e-12))
if worst["ess_min"] / worst["ess_median"] < 0.5:
    print(f"\n  note: the worst-mixing parameter carries as little as "
          f"{100 * worst['ess_min'] / worst['ess_median']:.0f}% of the median "
          "ESS. Quoting the mean over parameters would overstate convergence.")

# --- acceptance ------------------------------------------------------------

print("\n=== acceptance by level " + "=" * 40)
for name in sorted(summary):
    acc = defaultdict(list)
    for d, meta in runs[name].values():
        for key, value in meta["acceptance"].items():
            acc[key].append(value)
    cells = "  ".join(f"{k}={np.mean(v):.3f}+/-{np.std(v):.3f}"
                      for k, v in sorted(acc.items()))
    print(f"  {name:15s}{cells}")

# --- csv -------------------------------------------------------------------

cols = ["config", "levels", "nsub", "randomize", "iterations", "n_reps",
        "runtime_seconds", "ess_min", "ess_median", "ess_min_per_second",
        "ess_rate_sem", "speedup_vs_baseline"]
lines = [",".join(cols)]
for name in sorted(summary):
    s = summary[name]
    lines.append(",".join(str(v) for v in [
        name, "|".join(str(x) for x in s["levels"]), s["nsub"], s["randomize"],
        s["iterations"], s["n"], s["runtime"], s["ess_min"], s["ess_median"],
        s["rate"], s["rate_sem"],
        s["rate"] / base["rate"] if base and base["rate"] > 0 else "",
    ]))
with open(args.csv, "w") as fh:
    fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# figures
# ---------------------------------------------------------------------------

# Okabe-Ito: legible in the common forms of colour blindness and in greyscale.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7",
           "#E69F00", "#56B4E9", "#555555", "#000000"]

fig, (ax_sweep, ax_all) = plt.subplots(1, 2, figsize=(12.5, 4.8))

# left: subchain-length sweep, holding the level set and everything else fixed
sweep = {n: s for n, s in summary.items()
         if s["levels"] == [0, 1, 2] and s["randomize"]}
if sweep:
    order = sorted(sweep, key=lambda n: sweep[n]["nsub"])
    x = [sweep[n]["nsub"] for n in order]
    y = [sweep[n]["rate"] for n in order]
    e = [sweep[n]["rate_sem"] for n in order]
    ax_sweep.errorbar(x, y, yerr=e, marker="o", color=PALETTE[0], capsize=3,
                      lw=1.5, label="3-level, randomised")
    for n, xi, yi in zip(order, x, y):
        ax_sweep.annotate(n, (xi, yi), textcoords="offset points",
                          xytext=(0, 8), fontsize=7, ha="center")
if base:
    ax_sweep.axhline(base["rate"], ls="--", color=PALETTE[6], lw=1.2,
                     label=f"{args.baseline} (single level)")
ax_sweep.set_xscale("log")
ax_sweep.set_xlabel("subchain length")
ax_sweep.set_ylabel("ESS (min over parameters) per second")
ax_sweep.set_title("subchain-length sweep")
ax_sweep.grid(alpha=0.3, which="both")
ax_sweep.legend(fontsize=8)

# right: every config, so the level-set and randomisation comparisons are visible
order = sorted(summary, key=lambda n: summary[n]["rate"])
ypos = np.arange(len(order))
colours = [PALETTE[6] if n == args.baseline else PALETTE[0] for n in order]
ax_all.barh(ypos, [summary[n]["rate"] for n in order],
            xerr=[summary[n]["rate_sem"] for n in order],
            color=colours, height=0.65, capsize=3)
ax_all.set_yticks(ypos)
ax_all.set_yticklabels([f"{n}  (n={summary[n]['n']})" for n in order], fontsize=8)
if base:
    ax_all.axvline(base["rate"], ls="--", color=PALETTE[6], lw=1.2)
ax_all.set_xlabel("ESS (min over parameters) per second")
ax_all.set_title("all configurations")
ax_all.grid(alpha=0.3, axis="x")

fig.tight_layout()
fig.savefig(args.out, dpi=150)
print(f"\nwrote {args.out} and {args.csv}")