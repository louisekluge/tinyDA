"""Compare sampler configurations across the diagnostics sweep.

Reads the small diag_*.npz summaries, not the chain files. Three questions:

1. Which configuration buys the most independent samples per second?
   Iteration counts differ between configs, so everything is normalised by
   runtime, and ESS is the minimum over parameters -- a chain is only as
   converged as its worst-mixing direction, and this prior constrains log_a
   and log_d a hundred times more tightly than the rest.

2. Does variance reduction work, and where does it fail? The telescoping
   estimator only beats the plain one if the corrections are small, and a
   correction is only small if both its terms sit at the same parameter
   value. The pairing column measures that directly.

3. Do the configurations agree on the answer? They all target the same
   posterior, so a configuration whose estimator mean sits away from the
   others is biased.

The variance-reduction ratio is reported two ways. "predicted" is the
within-run estimate, sum_l var(term_l)/ESS(term_l), which assumes the
telescoping terms are independent -- they are not, since one coupled chain
produced them all. "empirical" is the spread of the per-run estimates across
repetitions, which assumes nothing. Where the two disagree, the empirical one
is right and the gap measures how wrong the independence assumption is.

    python analyse_diagnostics.py --indir results
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
p.add_argument("--out", default=None, help="default: <indir>/diagnostics.png")
p.add_argument("--csv", default=None, help="default: <indir>/diagnostics.csv")
p.add_argument("--expect-reps", type=int, default=20)
p.add_argument("--min-reps", type=int, default=5,
               help="below this an empirical SE is too noisy to quote")
p.add_argument("--baseline", default="single")
p.add_argument("--block", default="theta", choices=["theta", "qoi"],
               help="which quantity the estimator columns describe")
p.add_argument("--bootstrap", type=int, default=4000)
args = p.parse_args()

indir = os.path.expanduser(args.indir)
out_png = args.out or os.path.join(indir, "diagnostics.png")
out_csv = args.csv or os.path.join(indir, "diagnostics.csv")

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

print(f"{len(files)} files in {indir}, {len(runs)} configurations")
if unreadable:
    print("\nUNREADABLE")
    for name, why in unreadable:
        print(f"  {name}: {why}")

gaps = []
for name in sorted(runs):
    missing = sorted(set(range(args.expect_reps)) - set(runs[name]))
    if missing:
        shown = ", ".join(str(i) for i in missing[:12])
        more = f" (+{len(missing) - 12} more)" if len(missing) > 12 else ""
        gaps.append(f"{name}: {len(runs[name])}/{args.expect_reps} reps, "
                    f"missing {shown}{more}")
if gaps:
    print("\nINCOMPLETE CONFIGURATIONS")
    for line in gaps:
        print(f"  - {line}")

PARAMS = next(iter(runs.values()))[next(iter(next(iter(runs.values()))))][1][
    "parameter_names"]


# ---------------------------------------------------------------------------

def bootstrap_ratio(a, b, n_boot, rng):
    """Percentile interval for sd(a)/sd(b), resampling repetitions.

    The two are paired -- one chain produced both -- so repetitions are
    resampled jointly; treating them as independent would overstate the width.
    """
    n = a.size
    if n < 4:
        return float("nan"), float("nan")
    idx = rng.integers(0, n, size=(n_boot, n))
    sd_a, sd_b = a[idx].std(axis=1, ddof=1), b[idx].std(axis=1, ddof=1)
    ok = sd_b > 0
    if not ok.any():
        return float("nan"), float("nan")
    return tuple(float(v) for v in np.percentile(sd_a[ok] / sd_b[ok], [2.5, 97.5]))


rng = np.random.default_rng(0)
summary = {}
for name in sorted(runs):
    reps = runs[name]
    meta0 = next(iter(reps.values()))[1]
    n_levels = meta0["n_levels"]
    block = args.block
    if block == "qoi" and not meta0.get("has_qoi"):
        continue

    per = defaultdict(list)
    for d, meta in reps.values():
        ess = np.array([float(d[f"ess__level{n_levels - 1}_{p_}"]) for p_ in PARAMS])
        runtime = float(d["runtime_seconds"])
        per["ess_min"].append(ess.min())
        per["ess_median"].append(float(np.median(ess)))
        per["runtime"].append(runtime)
        per["rate"].append(ess.min() / runtime)
        per["standard"].append(np.atleast_1d(d[f"{block}__standard"]))
        per["reduced"].append(np.atleast_1d(d[f"{block}__reduced"]))
        if f"{block}__term_var" in d.files:
            pred_vr = np.sqrt((d[f"{block}__term_var"]
                               / d[f"{block}__term_ess"]).sum(axis=0))
            pred_std = np.sqrt(d[f"{block}__top_var"] / d[f"{block}__top_ess"])
            per["pred_ratio"].append(pred_std / pred_vr)
        if "pair__match" in d.files:
            per["pairing"].append(np.atleast_1d(d["pair__match"]))
        for level in range(n_levels):
            key = f"acc__level{level}"
            if key in d.files:
                per[f"acc{level}"].append(float(d[key]))

    n = len(per["rate"])
    standard = np.vstack(per["standard"])          # (n_reps, n_components)
    reduced = np.vstack(per["reduced"])
    se_std = standard.std(axis=0, ddof=1) if n > 1 else np.full(standard.shape[1], np.nan)
    se_vr = reduced.std(axis=0, ddof=1) if n > 1 else np.full(reduced.shape[1], np.nan)
    with np.errstate(divide="ignore", invalid="ignore"):
        emp_ratio = se_std / se_vr

    # Headline on the worst-mixing parameter, which is what limits the run.
    worst = int(np.argmin(np.mean([np.array([float(d[f"ess__level{n_levels - 1}_{p_}"])
                                             for p_ in PARAMS])
                                   for d, _ in reps.values()], axis=0))) \
        if block == "theta" else 0
    lo, hi = bootstrap_ratio(standard[:, worst], reduced[:, worst],
                             args.bootstrap, rng)

    summary[name] = dict(
        n=n, n_levels=n_levels, levels=meta0["levels"], nsub=meta0["nsub"],
        randomize=meta0["randomize"], iterations=meta0["iterations"],
        runtime=float(np.median(per["runtime"])),
        ess_min=float(np.mean(per["ess_min"])),
        ess_median=float(np.mean(per["ess_median"])),
        rate=float(np.mean(per["rate"])),
        rate_sem=float(np.std(per["rate"], ddof=1) / np.sqrt(n)) if n > 1 else np.nan,
        acceptance={f"level{l}": float(np.mean(per[f"acc{l}"]))
                    for l in range(n_levels) if per[f"acc{l}"]},
        pairing=(np.vstack(per["pairing"]).mean(axis=0)
                 if per["pairing"] else np.array([])),
        emp_ratio=emp_ratio,
        pred_ratio=(np.vstack(per["pred_ratio"]).mean(axis=0)
                    if per["pred_ratio"] else np.full_like(emp_ratio, np.nan)),
        worst=worst,
        worst_name=(PARAMS[worst] if block == "theta" else "qoi"),
        ratio_lo=lo, ratio_hi=hi,
        mean_std=standard.mean(axis=0), mean_vr=reduced.mean(axis=0),
        se_std=se_std, se_vr=se_vr,
    )

if not summary:
    raise SystemExit(f"no configuration has a '{args.block}' block")

base = summary.get(args.baseline)


def mark(name):
    return name + ("*" if summary[name]["n"] < args.min_reps else "")


# --- efficiency ------------------------------------------------------------

print(f"\n=== efficiency on the finest level " + "=" * 56)
print(f"{'config':15s}{'levels':>9s}{'J':>4s}{'rand':>5s}{'iters':>7s}{'n':>4s}"
      f"{'min':>8s}{'ESSmin':>8s}{'ESSmed':>8s}{'ESS/s':>9s}{'speedup':>9s}")
for name in sorted(summary, key=lambda k: -summary[k]["rate"]):
    s = summary[name]
    speed = (f"{s['rate'] / base['rate']:9.2f}" if base and base["rate"] > 0
             else f"{'n/a':>9s}")
    print(f"{mark(name):15s}{'|'.join(map(str, s['levels'])):>9s}{s['nsub']:4d}"
          f"{str(s['randomize'])[0]:>5s}{s['iterations']:7d}{s['n']:4d}"
          f"{s['runtime'] / 60:8.1f}{s['ess_min']:8.0f}{s['ess_median']:8.0f}"
          f"{s['rate']:9.4f}{speed}")

# --- pairing and acceptance ------------------------------------------------

print(f"\n=== pairing and acceptance " + "=" * 50)
print("  pairing is the fraction of exported correction pairs sitting at a")
print("  common parameter value; a correction that is not paired contributes")
print("  the full spread of Q rather than the spread of a difference.")
for name in sorted(summary):
    s = summary[name]
    pair = ("  ".join(f"Y_{l+1}{l}={v:.1%}" for l, v in enumerate(s["pairing"]))
            if s["pairing"].size else "n/a (single level)")
    acc = "  ".join(f"{k}={v:.3f}" for k, v in sorted(s["acceptance"].items()))
    print(f"  {mark(name):15s} {pair:34s} {acc}")

# --- variance reduction ----------------------------------------------------

print(f"\n=== variance reduction, block '{args.block}' " + "=" * 40)
print(f"{'config':15s}{'on':>10s}{'SE std':>11s}{'SE vr':>11s}"
      f"{'empirical':>11s}{'95% CI':>16s}{'predicted':>11s}")
for name in sorted(summary):
    s = summary[name]
    if s["n_levels"] == 1:
        continue
    w = s["worst"]
    ci = (f"[{s['ratio_lo']:.2f}, {s['ratio_hi']:.2f}]"
          if np.isfinite(s["ratio_lo"]) else "n/a")
    print(f"{mark(name):15s}{s['worst_name']:>10s}{s['se_std'][w]:11.3e}"
          f"{s['se_vr'][w]:11.3e}{s['emp_ratio'][w]:11.2f}{ci:>16s}"
          f"{s['pred_ratio'][w]:11.2f}")
print("  'on' is the worst-mixing parameter, which is what limits the run.")
print("  predicted assumes the telescoping terms are independent; empirical")
print("  assumes nothing. A gap between them is that assumption failing.")

if any(summary[n]["n"] < args.min_reps for n in summary):
    print(f"\n* fewer than {args.min_reps} reps: SE and ratios are provisional")

# --- bias ------------------------------------------------------------------

usable = [n for n in sorted(summary) if summary[n]["n"] >= args.min_reps]
if len(usable) >= 2:
    print(f"\n=== bias: estimator means against the weighted consensus " + "=" * 18)
    comps = range(len(summary[usable[0]]["mean_vr"]))
    for c in comps:
        name_c = PARAMS[c] if args.block == "theta" else "qoi"
        sem = {n: summary[n]["se_vr"][c] / np.sqrt(summary[n]["n"]) for n in usable}
        mean = {n: summary[n]["mean_vr"][c] for n in usable}
        w = {n: 1.0 / sem[n] ** 2 for n in usable if sem[n] > 0}
        if not w:
            continue
        consensus = sum(w[n] * mean[n] for n in w) / sum(w.values())
        flagged = [(n, (mean[n] - consensus) / sem[n]) for n in usable
                   if sem[n] > 0 and abs((mean[n] - consensus) / sem[n]) > 3]
        status = ("  ".join(f"{n} {z:+.1f}sigma" for n, z in flagged)
                  if flagged else "all configurations consistent")
        print(f"  {name_c:10s} consensus={consensus:+.6g}   {status}")

# --- csv -------------------------------------------------------------------

cols = ["config", "levels", "nsub", "randomize", "iterations", "n_reps",
        "runtime_seconds", "ess_min", "ess_median", "ess_min_per_second",
        "component", "mean_standard", "mean_vr", "se_standard", "se_vr",
        "ratio_empirical", "ratio_predicted"]
lines = [",".join(cols)]
for name in sorted(summary):
    s = summary[name]
    names = PARAMS if args.block == "theta" else ["qoi"]
    for c, comp in enumerate(names[:len(s["mean_vr"])]):
        lines.append(",".join(str(v) for v in [
            name, "|".join(map(str, s["levels"])), s["nsub"], s["randomize"],
            s["iterations"], s["n"], s["runtime"], s["ess_min"], s["ess_median"],
            s["rate"], comp, s["mean_std"][c], s["mean_vr"][c],
            s["se_std"][c], s["se_vr"][c], s["emp_ratio"][c], s["pred_ratio"][c],
        ]))
with open(out_csv, "w") as fh:
    fh.write("\n".join(lines) + "\n")


# ---------------------------------------------------------------------------
# figure
# ---------------------------------------------------------------------------

# Okabe-Ito: legible in the common forms of colour blindness and in greyscale.
PALETTE = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]

# The subchain sweep holds the level set and the promotion rule fixed. J=1 is
# included although it is recorded as not randomised: promotion draws from the
# last J states, so at J=1 there is nothing to randomise and the config is not
# making a different choice. A fixed-promotion run at J>1 is a different
# choice, and stays out.
sweep = sorted((n for n, s in summary.items()
                if s["levels"] == [0, 1, 2] and s["n_levels"] > 1
                and (s["randomize"] or s["nsub"] == 1)),
               key=lambda n: summary[n]["nsub"])

fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.6))

ax = axes[0]
if sweep:
    x = [summary[n]["nsub"] for n in sweep]
    ax.errorbar(x, [summary[n]["rate"] for n in sweep],
                yerr=[summary[n]["rate_sem"] for n in sweep],
                marker="o", color=PALETTE[0], capsize=3, lw=1.5,
                label="3-level, randomised")
if base:
    ax.axhline(base["rate"], ls="--", color="grey", lw=1.2,
               label=f"{args.baseline} (single level)")
ax.set_xscale("log")
ax.set_xlabel("subchain length J")
ax.set_ylabel("ESS (min over parameters) per second")
ax.set_title("efficiency")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = axes[1]
if sweep:
    for lvl in range(max(summary[n]["n_levels"] for n in sweep) - 1):
        xs = [summary[n]["nsub"] for n in sweep if len(summary[n]["pairing"]) > lvl]
        ys = [summary[n]["pairing"][lvl] for n in sweep
              if len(summary[n]["pairing"]) > lvl]
        if xs:
            ax.plot(xs, ys, marker="o", color=PALETTE[lvl % len(PALETTE)],
                    lw=1.5, label=f"Y_{lvl+1}{lvl}")
ax.set_xscale("log")
ax.set_ylim(0, 1.02)
ax.set_xlabel("subchain length J")
ax.set_ylabel("fraction paired at a common theta")
ax.set_title("correction pairing")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

ax = axes[2]
if sweep:
    w = [summary[n]["worst"] for n in sweep]
    ax.plot([summary[n]["nsub"] for n in sweep],
            [summary[n]["emp_ratio"][wi] for n, wi in zip(sweep, w)],
            marker="o", color=PALETTE[0], lw=1.5, label="empirical")
    ax.plot([summary[n]["nsub"] for n in sweep],
            [summary[n]["pred_ratio"][wi] for n, wi in zip(sweep, w)],
            marker="s", ls="--", color=PALETTE[1], lw=1.2, label="predicted")
ax.axhline(1.0, color="grey", lw=1.0)
ax.set_xscale("log")
ax.set_xlabel("subchain length J")
ax.set_ylabel("SE(standard) / SE(variance reduced)")
ax.set_title(f"variance reduction, {args.block}")
ax.grid(alpha=0.3, which="both")
ax.legend(fontsize=8)

fig.suptitle("below 1 on the right panel means the telescoping estimator is "
             "worse than the plain one", fontsize=9, y=0.995)
fig.tight_layout()
fig.savefig(out_png, dpi=150)
print(f"\nwrote {out_png} and {out_csv}")