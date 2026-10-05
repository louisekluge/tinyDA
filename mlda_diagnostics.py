"""One repetition of one sampler configuration on the predator-prey problem.

Every run goes through MLDAChain, including the two-level ones, so a sweep
varies only its configuration and not which implementation produced it.
tinyDA.sample() would send two posteriors to DAChain instead, which made
two-level and three-level results incomparable.

Writes two files per run:

    diag_<config>_rep<NN>.npz    summary: exact acceptance, ESS, estimator
                                 values, per-term variance and ESS, pairing
                                 fractions. A few kB; aggregate these.
    chains_<config>_rep<NN>.npz  every link of every level: parameters, qoi
                                 and log-densities, no burn-in applied.
                                 Needed for anything that compares estimator
                                 *formulations*, because a state-paired
                                 correction cannot be rebuilt from a
                                 proposal-paired summary.

The problem is built under a fixed seed so every repetition targets the same
posterior; only the sampler's randomness varies with the repetition index.

    python mlda_diagnostics.py --task-id 0
    python mlda_diagnostics.py --task-id 0 --iterations 200 --no-chains
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import arviz as az
import numpy as np
import scipy.stats as stats
import tinyDA as tda
from scipy.integrate import solve_ivp

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mlda_sampling import (  # noqa: E402
    acceptance_rates,
    make_checkpointer,
    pairing_fractions,
    sample_multilevel,
)

warnings.filterwarnings("ignore", message=".*qoi group is not defined.*")


# ---------------------------------------------------------------------------
# Sweep
#
# Each config isolates one choice against `base`:
#   nsub1/2/5/20   subchain length, level set fixed
#   fixed_nsub     promotion of the subchain endpoint rather than a random
#                  state within it (the flag is named randomize_subchain_length
#                  but the length never varies)
#   two_level_*    whether the middle level earns its cost, and whether a
#                  cheap-but-coarse surrogate beats an expensive-but-close one
#   single         the baseline everything is measured against
#
# nsub20 runs fewer iterations because its cost per top-level sample grows
# like J^2; see the runtime note in the sbatch.
# ---------------------------------------------------------------------------

CONFIGS = [
    dict(name="base",         levels=[0, 1, 2], nsub=10, rand=True,  iters=20000, aem=None),
    dict(name="nsub1",        levels=[0, 1, 2], nsub=1,  rand=False, iters=20000, aem=None),
    dict(name="nsub2",        levels=[0, 1, 2], nsub=2,  rand=True,  iters=20000, aem=None),
    dict(name="nsub5",        levels=[0, 1, 2], nsub=5,  rand=True,  iters=20000, aem=None),
    dict(name="nsub20",       levels=[0, 1, 2], nsub=20, rand=True,  iters=8000,  aem=None),
    dict(name="fixed_nsub",   levels=[0, 1, 2], nsub=10, rand=False, iters=20000, aem=None),
    dict(name="two_level_hi", levels=[1, 2],    nsub=10, rand=True,  iters=20000, aem=None),
    dict(name="two_level_lo", levels=[0, 2],    nsub=10, rand=True,  iters=20000, aem=None),
    dict(name="single",       levels=[2],       nsub=0,  rand=False, iters=20000, aem=None),
]
N_REPS = 20

# nsub1 is recorded as rand=False deliberately: promotion draws uniformly from
# the last J states, so at J=1 there is one outcome and randomisation is a
# no-op. Labelling it True would record a distinction the run does not have.


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

_DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

p = argparse.ArgumentParser()
p.add_argument("--task-id", type=int, required=True,
               help="Slurm array index; maps to config x repetition")
p.add_argument("--iterations", type=int, default=None,
               help="override the config's iteration count (for quick tests)")
p.add_argument("--outdir", default=_DEFAULT_OUTDIR)
p.add_argument("--no-chains", action="store_true",
               help="skip the full chain file and write only the summary")
p.add_argument("--checkpoint-every", type=int, default=None,
               help="write a partial chain file this often, in top-level "
                    "iterations; default is a tenth of the run")
args = p.parse_args()

cfg_idx, rep = divmod(args.task_id, N_REPS)
if cfg_idx >= len(CONFIGS):
    raise SystemExit(f"task-id {args.task_id} exceeds {len(CONFIGS) * N_REPS - 1}")
cfg = CONFIGS[cfg_idx]

iterations = args.iterations if args.iterations is not None else cfg["iters"]
burnin = iterations // 5
n_levels = len(cfg["levels"])

print(f"config={cfg['name']}  rep={rep}  levels={cfg['levels']}  "
      f"nsub={cfg['nsub']}  rand={cfg['rand']}  aem={cfg['aem']}  "
      f"iters={iterations}  burnin={burnin}", flush=True)

outdir = os.path.expanduser(args.outdir)
os.makedirs(outdir, exist_ok=True)
probe = os.path.join(outdir, f".probe_{args.task_id}")
with open(probe, "w") as fh:          # fail before sampling, not after
    fh.write("ok")
os.remove(probe)


# ---------------------------------------------------------------------------
# Problem -- fixed seed, identical for every repetition
# ---------------------------------------------------------------------------

np.random.seed(987)


class PredatorPreyModel:
    def __init__(self, datapoints):
        self.datapoints = datapoints
        self.t_span = [0, self.datapoints[-1]]

    def dydx(self, t, y, a, b, c, d):
        return np.array([a * y[0] - b * y[0] * y[1],
                         c * y[0] * y[1] - d * y[1]])

    def __call__(self, parameters):
        P_0, Q_0, a, b, c, d = np.exp(parameters)
        self.y = solve_ivp(
            lambda t, y: self.dydx(t, y, a, b, c, d),
            self.t_span, np.array([P_0, Q_0]), t_eval=self.datapoints)
        if self.y.success:
            return self.y.y.flatten(), self.y.y[1, :].mean()
        return np.nan, np.nan


true_parameters = np.log(np.array([10, 5, 1.0, 0.3, 0.2, 1.0]))
t_span = [0, 12]

# The coarse levels stop integrating early, so the model's own QoI -- the mean
# predator population -- is an average over a different interval on each
# level. The estimator stays unbiased, but its corrections are differences
# between unlike functionals, so no variance reduction should be expected from
# it. Parameter projections do not have this problem and are the honest probe.
n_data_l2, n_data_l1, n_data_l0 = 25, 16, 8
t_eval_l2 = np.linspace(t_span[0], t_span[1], n_data_l2)
t_eval_l1 = t_eval_l2[:n_data_l1]
t_eval_l0 = t_eval_l2[:n_data_l0]

models = [PredatorPreyModel(t) for t in (t_eval_l0, t_eval_l1, t_eval_l2)]

sigma = 1.0
noise_l2 = np.random.normal(scale=sigma, size=(t_eval_l2.size, 2))
data_l2 = models[2](true_parameters)[0] + np.hstack((noise_l2[:, 0], noise_l2[:, 1]))
data_l2[data_l2 < 0] = 0
noise_l1 = np.hstack((noise_l2[:n_data_l1, 0], noise_l2[:n_data_l1, 1]))
data_l1 = models[1](true_parameters)[0] + noise_l1
data_l1[data_l1 < 0] = 0
noise_l0 = np.hstack((noise_l2[:n_data_l0, 0], noise_l2[:n_data_l0, 1]))
data_l0 = models[0](true_parameters)[0] + noise_l0
data_l0[data_l0 < 0] = 0
data = [data_l0, data_l1, data_l2]

mean_prior = np.array([np.log(data_l2[0]), np.log(data_l2[n_data_l2]), 0, -1, -1.5, 0])
cov_prior = np.diag([0.1, 0.1, 0.001, 0.1, 0.1, 0.001])
my_prior = stats.multivariate_normal(mean_prior, cov_prior)

PARAMETER_NAMES = ["log_P0", "log_Q0", "log_a", "log_b", "log_c", "log_d"]


def make_posterior(level, adaptive):
    """A fresh posterior. The adaptive likelihood accumulates bias statistics,
    so one must never be shared between runs."""
    cov = sigma**2 * np.eye(data[level].size)
    like = (tda.AdaptiveGaussianLogLike(data[level], cov) if adaptive
            else tda.GaussianLogLike(data[level], cov))
    return tda.Posterior(my_prior, like, models[level])


# The finest level is always exact; coarse levels get an adaptive likelihood
# only when the error model is on, since that is what needs set_bias().
posteriors = [
    make_posterior(level, adaptive=(cfg["aem"] is not None and i < n_levels - 1))
    for i, level in enumerate(cfg["levels"])
]
MAP = tda.get_MAP(make_posterior(2, adaptive=False))


# ---------------------------------------------------------------------------
# Sample -- per-repetition randomness only
# ---------------------------------------------------------------------------

np.random.seed(4242 + rep)
proposal = tda.AdaptiveMetropolis(C0=0.01 * np.eye(6), t0=100, sd=None, epsilon=1e-6)

chainfile = os.path.join(outdir, f"chains_{cfg['name']}_rep{rep:02d}.npz")
checkpoint = None
if not args.no_chains:
    every = args.checkpoint_every or max(1, iterations // 10)
    checkpoint = make_checkpointer(chainfile, every=every)

t_start = time.time()
if n_levels == 1:
    # Single-level MH goes through tinyDA.sample, which has no callback hook,
    # so there is no checkpointing here. Acceptable: the baseline is the
    # cheapest config in the sweep and the least likely to hit a walltime.
    chain = tda.sample(posteriors[0], proposal, iterations=iterations,
                       n_chains=1, initial_parameters=MAP)
else:
    chain = sample_multilevel(
        posteriors, proposal, iterations,
        subchain_lengths=cfg["nsub"],
        randomize_subchain_length=cfg["rand"],
        initial_parameters=MAP,
        adaptive_error_model=cfg["aem"],
        callback=checkpoint,
    )
runtime = time.time() - t_start
print(f"\nsampled in {runtime / 60:.1f} min", flush=True)


# ---------------------------------------------------------------------------
# Summaries
# ---------------------------------------------------------------------------

def links_to_array(links, attribute):
    if attribute == "stats":
        return np.array([[l.prior, l.likelihood, l.posterior] for l in links],
                        dtype=np.float64)
    return np.array([np.atleast_1d(getattr(l, attribute)) for l in links],
                    dtype=np.float64)


def ess_1d(x):
    x = np.asarray(x, dtype=float).ravel()
    if x.size < 8 or np.allclose(x, x[0]):
        return float(x.size)
    return float(az.ess(az.convert_to_dataset(x[None, :]))["x"].item())


def level_arrays(chain, attribute):
    """{'chains': {level: array}, 'promoted': {level: array}}, no burn-in."""
    if chain["sampler"] == "MH":
        return {0: links_to_array(chain["chain_0"], attribute)}, {}
    chains, promoted = {}, {}
    for level in range(n_levels):
        key = f"chain_l{level}_0"
        if chain.get(key) is not None:
            chains[level] = links_to_array(chain[key], attribute)
        key = f"promoted_l{level}_0"
        if chain.get(key) is not None:
            promoted[level] = links_to_array(chain[key], attribute)
    return chains, promoted


def vr_terms(chains, promoted, burnin):
    """Q_top and the telescoping terms, burned in consistently.

    The top-level chain carries its initial state, which no proposal produced,
    so it is one longer than its promoted partner and is offset by one. Lower
    levels pair directly. Burn-in is scaled to each level by chain length so
    every term is cut at the same point in the run.
    """
    top = chains[n_levels - 1]
    n_top = len(top)
    cut_top = min(burnin, n_top - 1)
    out = {"Q_0": chains[0][int(round(burnin * len(chains[0]) / n_top)):]}
    for level in range(1, n_levels):
        fine, prom = chains[level], promoted[level - 1]
        if len(fine) == len(prom) + 1:
            fine = fine[1:]
        n = min(len(fine), len(prom))
        cut = int(round(burnin * n / n_top))
        out[f"Y_{level}{level - 1}"] = fine[cut:n] - prom[cut:n]
    return top[cut_top:], out


summary = {
    "MAP": MAP,
    "runtime_seconds": np.array(runtime),
    "true_parameters": true_parameters,
}

if n_levels > 1:
    rates = acceptance_rates(chain, burnin)
    pairing = pairing_fractions(chain, burnin)
    summary["pair__match"] = np.array(
        [pairing[f"Y_{l + 1}{l}"] for l in range(n_levels - 1)])
else:
    # Single-level MH: tinyDA records acceptance on the chain object, but
    # sample() does not export it, so fall back to state changes. This is the
    # baseline only, and the bias is small at one level because there are no
    # alignment copies and no subchain that can return to its start.
    theta = links_to_array(chain["chain_0"], "parameters")
    moved = np.any(np.diff(theta[burnin:], axis=0) != 0, axis=1)
    rates = {"level0": float(moved.mean())}
    pairing = {}

for key, value in rates.items():
    summary[f"acc__{key}"] = np.array(value)

# ESS per level per parameter, and the estimator decomposition.
theta_chains, theta_promoted = level_arrays(chain, "parameters")
has_qoi = chain.get("chain_l0_0" if n_levels > 1 else "chain_0") is not None and \
    np.all(np.isfinite(links_to_array(
        chain["chain_l0_0" if n_levels > 1 else "chain_0"][:1], "qoi")))

for level, arr in theta_chains.items():
    cut = int(round(burnin * len(arr) / len(theta_chains[n_levels - 1])))
    for j, name in enumerate(PARAMETER_NAMES):
        summary[f"ess__level{level}_{name}"] = np.array(ess_1d(arr[cut:, j]))

blocks = {"theta": (theta_chains, theta_promoted)}
if has_qoi:
    blocks["qoi"] = level_arrays(chain, "qoi")

for prefix, (chains_, promoted_) in blocks.items():
    if n_levels == 1:
        top = chains_[0][burnin:]
        summary[f"{prefix}__standard"] = top.mean(axis=0)
        summary[f"{prefix}__reduced"] = top.mean(axis=0)
        summary[f"{prefix}__top_var"] = top.var(axis=0)
        summary[f"{prefix}__top_ess"] = np.array(
            [ess_1d(top[:, j]) for j in range(top.shape[1])])
        continue
    top, terms = vr_terms(chains_, promoted_, burnin)
    names = list(terms)
    summary[f"{prefix}__term_names"] = np.array(json.dumps(names))
    summary[f"{prefix}__standard"] = top.mean(axis=0)
    summary[f"{prefix}__reduced"] = sum(t.mean(axis=0) for t in terms.values())
    summary[f"{prefix}__top_var"] = top.var(axis=0)
    summary[f"{prefix}__top_ess"] = np.array(
        [ess_1d(top[:, j]) for j in range(top.shape[1])])
    summary[f"{prefix}__term_var"] = np.array(
        [t.var(axis=0) for t in terms.values()])
    summary[f"{prefix}__term_ess"] = np.array(
        [[ess_1d(t[:, j]) for j in range(t.shape[1])] for t in terms.values()])

summary["meta"] = np.array(json.dumps({
    "config": cfg["name"], "rep": rep, "levels": cfg["levels"],
    "nsub": cfg["nsub"], "randomize": cfg["rand"], "aem": cfg["aem"],
    "iterations": iterations, "burnin": burnin, "n_levels": n_levels,
    "sampler": chain["sampler"], "runtime_seconds": runtime,
    "parameter_names": PARAMETER_NAMES, "has_qoi": bool(has_qoi),
    "acceptance": rates, "pairing": pairing,
    "chain_file": os.path.basename(chainfile) if not args.no_chains else None,
}))

outfile = os.path.join(outdir, f"diag_{cfg['name']}_rep{rep:02d}.npz")
np.savez_compressed(outfile, **summary)

# Multilevel runs already wrote their chain file through the checkpointer; the
# single-level baseline has no callback hook, so write it here in the same
# layout, so analysis sees one format.
if n_levels == 1 and not args.no_chains:
    theta = links_to_array(chain["chain_0"], "parameters")
    moved = np.concatenate(
        ([True], np.any(np.diff(theta, axis=0) != 0, axis=1)))
    make_checkpointer(chainfile)(
        {"levels": 1, "subchain_lengths": [],
         "chain_l0_0": chain["chain_0"],
         "accepted_l0_0": moved},
        iterations, iterations)


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

print(f"{cfg['name']} rep {rep}: {runtime / 60:.1f} min, "
      f"sampler={chain['sampler']}")
for key, value in sorted(rates.items()):
    print(f"  acceptance   {key:10s} {value:.3f}")
for key, value in sorted(pairing.items()):
    print(f"  pairing      {key:10s} {value:.3f}")
ess_top = [float(summary[f"ess__level{n_levels - 1}_{n}"]) for n in PARAMETER_NAMES]
print(f"  ESS (finest) min={min(ess_top):.0f}  median={np.median(ess_top):.0f}")
if n_levels > 1:
    for prefix in blocks:
        std = np.sqrt(summary[f"{prefix}__top_var"] / summary[f"{prefix}__top_ess"])
        vr = np.sqrt((summary[f"{prefix}__term_var"]
                      / summary[f"{prefix}__term_ess"]).sum(axis=0))
        ratio = std / vr
        print(f"  VR ratio ({prefix}): "
              + "  ".join(f"{r:.2f}" for r in ratio))
print(f"saved -> {outfile}")
if not args.no_chains:
    print(f"        {chainfile}  ({os.path.getsize(chainfile) / 1024**2:.1f} MB)")