"""One configuration, opened up: the spread that a mean hides.

analyse_diagnostics.py compares configurations and reduces each to a mean.
This takes one of them and shows how much its repetitions differ, which
parameter limits convergence, and whether a slow repetition was slow because
it mixed badly or because of the node it landed on.

One configuration at a time. Repetitions of *different* configurations are not
repetitions of anything, and pooling them produces a boxplot of a quantity
that does not exist.

    python plot_diagnostics.py --indir results                 # list them
    python plot_diagnostics.py --indir results --config base
"""

from __future__ import annotations

import argparse
import glob
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))

p = argparse.ArgumentParser()
p.add_argument("--indir", default=os.path.join(_HERE, "results"))
p.add_argument("--config", default=None,
               help="which configuration to open up; omit to list what is there")
p.add_argument("--block", default="theta", choices=["theta", "qoi"])
p.add_argument("--out", default=None)
p.add_argument("--csv", default=None)
args = p.parse_args()

indir = os.path.expanduser(args.indir)
files = sorted(glob.glob(os.path.join(indir, "diag_*.npz")))
if not files:
    raise SystemExit(f"no diag_*.npz in {indir}")

by_config = {}
for path in files:
    d = np.load(path, allow_pickle=True)
    meta = json.loads(str(d["meta"]))
    by_config.setdefault(meta["config"], []).append((path, d, meta))

if args.config is None:
    print(f"{len(files)} files in {indir}, {len(by_config)} configurations:\n")
    for name in sorted(by_config):
        _, _, meta = by_config[name][0]
        print(f"  {name:15s} {len(by_config[name]):3d} reps   "
              f"levels={meta['levels']} nsub={meta['nsub']} "
              f"rand={meta['randomize']} aem={meta['aem']} "
              f"iters={meta['iterations']} sampler={meta['sampler']}")
    raise SystemExit("\nre-run with --config <name> to plot one of them")

if args.config not in by_config:
    raise SystemExit(f"no config {args.config!r}; have: "
                     f"{', '.join(sorted(by_config))}")

selected = sorted(by_config[args.config], key=lambda t: t[2]["rep"])
meta = selected[0][2]
n_levels = meta["n_levels"]
PARAMS = meta["parameter_names"]
out_png = args.out or os.path.join(indir, f"diag_{args.config}.png")
out_csv = args.csv or os.path.join(indir, f"diag_{args.config}.csv")

print(f"{args.config}: {len(selected)} repetitions "
      f"(reps {', '.join(str(m['rep']) for _, _, m in selected)})")
print(f"  levels={meta['levels']}  n_levels={n_levels}  nsub={meta['nsub']}  "
      f"rand={meta['randomize']}  aem={meta['aem']}  "
      f"iterations={meta['iterations']}  burnin={meta['burnin']}  "
      f"sampler={meta['sampler']}")

# Every repetition must target the same posterior, or they are not
# repetitions. The MAP is the cheapest fingerprint of that.
maps = [d["MAP"] for _, d, _ in selected]
if not all(np.allclose(maps[0], m) for m in maps):
    print("\nWARNING: MAP differs across repetitions -- these are NOT the same "
          "problem, and nothing below should be pooled.")
for _, _, m in selected:
    if m["n_levels"] != n_levels or m["nsub"] != meta["nsub"]:
        print("\nWARNING: metadata differs between repetitions of this config; "
              "the files were written by different versions of the script.")
        break


# ---------------------------------------------------------------------------

acc = {f"level{l}": np.array([float(d[f"acc__level{l}"]) for _, d, _ in selected
                              if f"acc__level{l}" in d.files])
       for l in range(n_levels)}
acc = {k: v for k, v in acc.items() if v.size}

ess = np.array([[float(d[f"ess__level{n_levels - 1}_{p_}"]) for p_ in PARAMS]
                for _, d, _ in selected])                       # (n_reps, n_par)
runtimes = np.array([float(d["runtime_seconds"]) for _, d, _ in selected])
reps = [m["rep"] for _, _, m in selected]

pairing = (np.vstack([d["pair__match"] for _, d, _ in selected])
           if "pair__match" in selected[0][1].files else np.empty((len(selected), 0)))

block = args.block
has_block = f"{block}__standard" in selected[0][1].files
if has_block:
    standard = np.vstack([np.atleast_1d(d[f"{block}__standard"]) for _, d, _ in selected])
    reduced = np.vstack([np.atleast_1d(d[f"{block}__reduced"]) for _, d, _ in selected])
    comp_names = PARAMS if block == "theta" else ["qoi"]
    comp_names = comp_names[:standard.shape[1]]

# --- report ---------------------------------------------------------------

lines = ["quantity,mean,std,min,max,n"]
print("\n--- acceptance across repetitions (exact, from recorded flags) ---")
for k, v in acc.items():
    print(f"  {k:10s} {v.mean():.3f} +/- {v.std():.3f}   [{v.min():.3f}, {v.max():.3f}]")
    lines.append(f"acc:{k},{v.mean()},{v.std()},{v.min()},{v.max()},{v.size}")

if pairing.size:
    print("\n--- correction pairing ---")
    for l in range(pairing.shape[1]):
        v = pairing[:, l]
        print(f"  Y_{l+1}{l}      {v.mean():.3f} +/- {v.std():.3f}   "
              f"[{v.min():.3f}, {v.max():.3f}]")
        lines.append(f"pairing:Y_{l+1}{l},{v.mean()},{v.std()},{v.min()},{v.max()},{v.size}")

print("\n--- ESS per parameter, finest level ---")
for j, name in enumerate(PARAMS):
    v = ess[:, j]
    print(f"  {name:10s} {v.mean():8.0f} +/- {v.std():7.0f}   "
          f"[{v.min():7.0f}, {v.max():7.0f}]")
    lines.append(f"ess:{name},{v.mean()},{v.std()},{v.min()},{v.max()},{v.size}")

if has_block and len(selected) > 1:
    print(f"\n--- estimators across repetitions, block '{block}' ---")
    print(f"  {'component':12s}{'mean std':>14s}{'mean vr':>14s}"
          f"{'SE std':>12s}{'SE vr':>12s}{'ratio':>8s}")
    for c, name in enumerate(comp_names):
        se_s, se_v = standard[:, c].std(ddof=1), reduced[:, c].std(ddof=1)
        ratio = se_s / se_v if se_v > 0 else np.nan
        print(f"  {name:12s}{standard[:, c].mean():14.6g}"
              f"{reduced[:, c].mean():14.6g}{se_s:12.3e}{se_v:12.3e}{ratio:8.2f}")
        lines.append(f"estimator:{name},{standard[:, c].mean()},{se_s},"
                     f"{reduced[:, c].mean()},{se_v},{len(selected)}")

print(f"\nruntime: {runtimes.mean()/60:.1f} +/- {runtimes.std()/60:.1f} min "
      f"[{runtimes.min()/60:.1f}, {runtimes.max()/60:.1f}]")
lines.append(f"runtime_min,{runtimes.mean()/60},{runtimes.std()/60},"
             f"{runtimes.min()/60},{runtimes.max()/60},{runtimes.size}")

with open(out_csv, "w") as fh:
    fh.write("\n".join(lines) + "\n")

# --- figure ---------------------------------------------------------------

rng = np.random.default_rng(0)        # jitter must not change between runs
jitter = lambda n, i: np.full(n, i) + rng.uniform(-0.06, 0.06, n)

fig, axes = plt.subplots(2, 2, figsize=(13, 9))

ax = axes[0, 0]
if acc:
    ax.boxplot(list(acc.values()), tick_labels=list(acc), showmeans=True)
    for i, v in enumerate(acc.values(), start=1):
        ax.plot(jitter(v.size, i), v, ".", color="tab:blue", alpha=0.6)
if pairing.size:
    twin = ax.twinx()
    for l in range(pairing.shape[1]):
        twin.plot(jitter(pairing.shape[0], n_levels + 0.5),
                  pairing[:, l], "x", color="tab:red", alpha=0.7)
    twin.set_ylabel("pairing fraction", color="tab:red")
    twin.set_ylim(0, 1.02)
    twin.tick_params(axis="y", colors="tab:red")
ax.set_ylabel("acceptance rate")
ax.set_title(f"Acceptance per level -- {args.config} ({len(selected)} reps)")
ax.grid(alpha=0.3)

ax = axes[0, 1]
ax.boxplot([ess[:, j] for j in range(ess.shape[1])],
           tick_labels=PARAMS, showmeans=True)
for j in range(ess.shape[1]):
    ax.plot(jitter(ess.shape[0], j + 1), ess[:, j], ".", color="tab:blue", alpha=0.6)
ax.set_ylabel("ESS")
ax.set_title(f"ESS per parameter -- level {n_levels - 1}")
ax.tick_params(axis="x", rotation=30)
ax.grid(alpha=0.3)

# The estimator panel is the variance-reduction story made visible: two clouds
# of per-repetition estimates, and the narrower cloud is the better estimator.
ax = axes[1, 0]
if has_block and len(selected) > 1:
    worst = int(np.argmin(ess.mean(axis=0))) if block == "theta" else 0
    worst = min(worst, standard.shape[1] - 1)
    centre = reduced[:, worst].mean()
    ax.plot(jitter(len(selected), 1), standard[:, worst] - centre, "o",
            mfc="none", mec="tab:blue", label="standard")
    ax.plot(jitter(len(selected), 2), reduced[:, worst] - centre, "o",
            color="tab:blue", label="variance reduced")
    ax.axhline(0, color="grey", lw=1.0)
    ax.set_xticks([1, 2])
    ax.set_xticklabels(["standard", "variance reduced"])
    ax.set_ylabel(f"estimate of E[{comp_names[worst]}], centred")
    ax.set_title(f"Estimator spread -- {comp_names[worst]} "
                 f"(worst-mixing of {block})")
else:
    ax.set_title("no estimator block in these files")
ax.grid(alpha=0.3, axis="y")

ax = axes[1, 1]
eff = ess.min(axis=1) / (runtimes / 60)
order = np.argsort(reps)
ax.bar(np.arange(len(eff)), eff[order], color="tab:green", alpha=0.85)
ax.set_xticks(np.arange(len(eff)))
ax.set_xticklabels([str(reps[i]) for i in order])
ax.axhline(eff.mean(), ls="--", color="grey", lw=1.2,
           label=f"mean {eff.mean():.1f}")
ax.set_xlabel("repetition")
ax.set_ylabel("min ESS per minute")
ax.set_title("Sampling efficiency per repetition")
ax.legend(fontsize=8)
ax.grid(alpha=0.3, axis="y")

fig.tight_layout()
fig.savefig(out_png, dpi=150)
print(f"\nwrote {out_png} and {out_csv}")