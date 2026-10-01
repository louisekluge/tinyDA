"""
Repeated-run diagnostics for the tinyDA MLDA predator-prey example.

One invocation = one independent repetition (one seed). Saves a small .npz
with per-level acceptance rates, per-parameter ESS, and timing.

The model/prior/data are constructed with a FIXED seed so every repetition
targets the same posterior; only the sampler randomness varies with --run-id.

Usage:
    python mlda_diagnostics.py --run-id 0 --iterations 20000
"""

import argparse
import json
import time
import os

import numpy as np
import scipy.stats as stats
from scipy.integrate import solve_ivp

import arviz as az
import tinyDA as tda

import warnings
warnings.filterwarnings("ignore", message=".*qoi group is not defined.*")


# --------------------------------------------------------------------------
# Sweep configurations
# --------------------------------------------------------------------------

CONFIGS = [
    dict(name="base",         levels=[0,1,2], nsub=10, rand=True,  prop="am", iters=20000, aem=None),
    dict(name="nsub1",        levels=[0,1,2], nsub=1,  rand=True,  prop="am", iters=20000, aem=None),
    dict(name="nsub2",        levels=[0,1,2], nsub=2,  rand=True,  prop="am", iters=20000, aem=None),
    dict(name="nsub5",        levels=[0,1,2], nsub=5,  rand=True,  prop="am", iters=20000, aem=None),
    dict(name="nsub20",       levels=[0,1,2], nsub=20, rand=True,  prop="am", iters=8000,  aem=None),
    dict(name="fixed_nsub",   levels=[0,1,2], nsub=10, rand=False, prop="am", iters=20000, aem=None),
    dict(name="two_level_hi", levels=[1,2],   nsub=10, rand=True,  prop="am", iters=20000, aem=None),
    dict(name="two_level_lo", levels=[0,2],   nsub=10, rand=True,  prop="am", iters=20000, aem=None),
    dict(name="single",       levels=[2],     nsub=0,  rand=False, prop="am", iters=20000, aem=None),
    dict(name="aem_si",       levels=[0,1,2], nsub=10, rand=True,  prop="am", iters=20000, aem="state-independent"),
]
N_REPS = 6

# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

_DEFAULT_OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")

p = argparse.ArgumentParser()
p.add_argument("--task-id", type=int, required=True,
               help="Slurm array index; maps to config x repetition")
p.add_argument("--iterations", type=int, default=None,
               help="override the config's iteration count (for quick tests)")
p.add_argument("--outdir", default=_DEFAULT_OUTDIR)
args = p.parse_args()

cfg_idx, rep = divmod(args.task_id, N_REPS)
if cfg_idx >= len(CONFIGS):
    raise SystemExit(f"task-id {args.task_id} exceeds {len(CONFIGS)*N_REPS-1}")
cfg = CONFIGS[cfg_idx]

iterations = args.iterations if args.iterations is not None else cfg["iters"]
burnin = iterations // 5
n_levels = len(cfg["levels"])

print(f"config={cfg['name']}  rep={rep}  levels={cfg['levels']}  "
      f"nsub={cfg['nsub']}  iters={iterations}")

_outdir = os.path.expanduser(args.outdir)
os.makedirs(_outdir, exist_ok=True)
_probe = os.path.join(_outdir, f".probe_{args.task_id}")
with open(_probe, "w") as fh:
    fh.write("ok")
os.remove(_probe)

# --------------------------------------------------------------------------
# Problem setup -- FIXED seed, identical for every repetition
# --------------------------------------------------------------------------

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
            self.t_span,
            np.array([P_0, Q_0]),
            t_eval=self.datapoints,
        )
        if self.y.success:
            return self.y.y.flatten(), self.y.y[1, :].mean()
        else:
            return np.nan, np.nan


true_parameters = np.log(np.array([10, 5, 1.0, 0.3, 0.2, 1.0]))
t_span = [0, 12]

# model hierarchy: coarse levels stop integrating early
n_data_l2, n_data_l1, n_data_l0 = 25, 16, 8
t_eval_l2 = np.linspace(t_span[0], t_span[1], n_data_l2)
t_eval_l1 = t_eval_l2[:n_data_l1]
t_eval_l0 = t_eval_l2[:n_data_l0]

my_model_l2 = PredatorPreyModel(t_eval_l2)
my_model_l1 = PredatorPreyModel(t_eval_l1)
my_model_l0 = PredatorPreyModel(t_eval_l0)

# data
sigma = 1.0
noise_l2 = np.random.normal(scale=sigma, size=(t_eval_l2.size, 2))
data_l2 = my_model_l2(true_parameters)[0] + np.hstack((noise_l2[:, 0], noise_l2[:, 1]))
data_l2[data_l2 < 0] = 0

noise_l1 = np.hstack((noise_l2[:n_data_l1, 0], noise_l2[:n_data_l1, 1]))
data_l1 = my_model_l1(true_parameters)[0] + noise_l1
data_l1[data_l1 < 0] = 0

noise_l0 = np.hstack((noise_l2[:n_data_l0, 0], noise_l2[:n_data_l0, 1]))
data_l0 = my_model_l0(true_parameters)[0] + noise_l0
data_l0[data_l0 < 0] = 0

# prior
mean_prior = np.array([np.log(data_l2[0]), np.log(data_l2[n_data_l2]), 0, -1, -1.5, 0])
cov_prior = np.diag([0.1, 0.1, 0.001, 0.1, 0.1, 0.001])
my_prior = stats.multivariate_normal(mean_prior, cov_prior)

# likelihoods and posteriors
my_loglike_l2 = tda.GaussianLogLike(data_l2, sigma**2 * np.eye(data_l2.size))
my_loglike_l1 = tda.AdaptiveGaussianLogLike(data_l1, sigma**2 * np.eye(data_l1.size))
my_loglike_l0 = tda.AdaptiveGaussianLogLike(data_l0, sigma**2 * np.eye(data_l0.size))

all_posteriors = [
    tda.Posterior(my_prior, my_loglike_l0, my_model_l0),
    tda.Posterior(my_prior, my_loglike_l1, my_model_l1),
    tda.Posterior(my_prior, my_loglike_l2, my_model_l2),
]
my_posteriors = [all_posteriors[i] for i in cfg["levels"]]

MAP = tda.get_MAP(all_posteriors[-1])


# --------------------------------------------------------------------------
# Per-repetition randomness
# --------------------------------------------------------------------------
np.random.seed(4242 + rep)

if cfg["prop"] == "am":
    my_proposal = tda.AdaptiveMetropolis(C0=0.01*np.eye(6), t0=100, sd=None, epsilon=1e-6)
else:
    my_proposal = tda.DREAMZ(M0=1000, delta=1, Z_method="lhs", adaptive=True)

# --------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------

t_start = time.time()
if n_levels == 1:
    chain = tda.sample(my_posteriors[0], my_proposal,
                       iterations=iterations, n_chains=1,
                       initial_parameters=MAP)
else:
    kwargs = dict(iterations=iterations, n_chains=1, initial_parameters=MAP,
                  subchain_length=cfg["nsub"],
                  randomize_subchain_length=cfg["rand"])
    if cfg["aem"] is not None:
        kwargs["adaptive_error_model"] = cfg["aem"]
    chain = tda.sample(my_posteriors, my_proposal, **kwargs)
runtime = time.time() - t_start

# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def _as_array(links):
    """Stack the parameter vectors out of a list of tinyDA Link objects."""
    return np.array([l.parameters for l in links], dtype=float)

def acceptance_rates(chain, n_levels, burnin=0):
    """Acceptance rate per level, post burn-in.

    sample() returns arrays, not chain objects, so acceptance is recovered
    from state changes. burnin is given in fine-level iterations and scaled
    to each level by the ratio of chain lengths.
    """
    rates = {}
    finest = _as_array(chain[f"chain_l{n_levels-1}_0"])
    n_fine = finest.shape[0]

    for level in range(n_levels):
        key = f"chain_l{level}_0"
        if key not in chain:
            continue
        samples = _as_array(chain[key])
        cut = int(round(burnin * samples.shape[0] / n_fine))
        post = samples[cut:]
        if post.shape[0] < 2:
            continue
        moved = np.any(np.diff(post, axis=0) != 0, axis=1)
        rates[f"level{level}"] = float(moved.mean())
    return rates


def ess_per_level(chain, burnin):
    """ESS per parameter, per level, via tinyDA's arviz bridge."""
    out = {}
    for level in list(range(n_levels)) + ["fine", "coarse"]:
        try:
            idata = tda.to_inference_data(chain, level=level, burnin=burnin)
        except Exception:
            continue
        try:
            ess = az.ess(idata)
        except Exception:
            continue
        for var in ess.data_vars:
            vals = np.atleast_1d(np.asarray(ess[var].values, dtype=float)).ravel()
            out[f"level{level}_{var}"] = vals
    return out


rates = acceptance_rates(chain, n_levels, burnin=burnin)
ess = ess_per_level(chain, burnin)

if not rates:
    print("WARNING: no acceptance-rate attributes found; inspect the chain object.")
if not ess:
    print("WARNING: to_inference_data produced nothing; check the level argument.")

payload = {
    "runtime_seconds": np.array(runtime),
    "MAP": MAP,
    "meta": np.array(json.dumps({
        "config": cfg["name"], "rep": rep, "levels": cfg["levels"],
        "nsub": cfg["nsub"], "randomize": cfg["rand"], "proposal": cfg["prop"],
        "aem": cfg["aem"], "iterations": iterations, "burnin": burnin,
        "n_levels": n_levels, "acceptance": rates,
    })),
}
for k, v in rates.items():
    payload[f"acc__{k}"] = np.array(v)
for k, v in ess.items():
    payload[f"ess__{k}"] = np.asarray(v)

outfile = os.path.join(_outdir, f"diag_{cfg['name']}_rep{rep:02d}.npz")
np.savez(outfile, **payload)

print(f"\nrun {args.run_id}: {runtime/60:.1f} min for {args.iterations} iterations")
for k, v in sorted(rates.items()):
    print(f"  acceptance  {k:45s} {v:.3f}")
for k, v in sorted(ess.items()):
    print(f"  ESS         {k:45s} min={v.min():8.0f}  median={np.median(v):8.0f}")
print(f"saved -> {outfile}")