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

import numpy as np
import scipy.stats as stats
from scipy.integrate import solve_ivp

import arviz as az
import tinyDA as tda


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

p = argparse.ArgumentParser()
p.add_argument("--run-id", type=int, required=True,
               help="repetition index; seeds the sampler only")
p.add_argument("--iterations", type=int, default=20000)
p.add_argument("--burnin", type=int, default=None,
               help="default: iterations // 5")
p.add_argument("--subchain-length", type=int, default=10)
p.add_argument("--randomize-subchain-length", action="store_true", default=True)
p.add_argument("--proposal", choices=["am", "dreamz"], default="am",
               help="coarsest-level proposal; am avoids the DREAMZ archive slowdown")
p.add_argument("--outdir", default=".")
args = p.parse_args()

burnin = args.burnin if args.burnin is not None else args.iterations // 5


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
my_loglike_l1 = tda.GaussianLogLike(data_l1, sigma**2 * np.eye(data_l1.size))
my_loglike_l0 = tda.GaussianLogLike(data_l0, sigma**2 * np.eye(data_l0.size))

my_posteriors = [
    tda.Posterior(my_prior, my_loglike_l0, my_model_l0),
    tda.Posterior(my_prior, my_loglike_l1, my_model_l1),
    tda.Posterior(my_prior, my_loglike_l2, my_model_l2),
]
n_levels = len(my_posteriors)

MAP = tda.get_MAP(my_posteriors[-1])


# --------------------------------------------------------------------------
# Per-repetition randomness
# --------------------------------------------------------------------------

np.random.seed(4242 + args.run_id)

if args.proposal == "am":
    my_proposal = tda.AdaptiveMetropolis(
        C0=0.01 * np.eye(6), t0=100, sd=None, epsilon=1e-6
    )
else:
    my_proposal = tda.DREAMZ(M0=1000, delta=1, Z_method="lhs", adaptive=True)


# --------------------------------------------------------------------------
# Sample
# --------------------------------------------------------------------------

t_start = time.time()
chain = tda.sample(
    my_posteriors,
    my_proposal,
    iterations=args.iterations,
    n_chains=1,
    initial_parameters=MAP,
    subchain_length=args.subchain_length,
    randomize_subchain_length=args.randomize_subchain_length,
)
runtime = time.time() - t_start


# --------------------------------------------------------------------------
# Diagnostics
# --------------------------------------------------------------------------

def _walk(obj):
    """Yield every chain-like object inside whatever sample() returned."""
    if isinstance(obj, dict):
        for v in obj.values():
            yield from _walk(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _walk(v)
    else:
        yield obj


def acceptance_rates(chain):
    """Mean acceptance rate per level, found by introspection.

    tinyDA stores acceptance flags as lists of bools on the chain objects
    (e.g. accepted_fine / accepted_coarse). Attribute names differ between
    chain types, so collect anything that looks like one.
    """
    rates = {}
    for obj in _walk(chain):
        for name in dir(obj):
            if not name.startswith("accepted"):
                continue
            try:
                flags = getattr(obj, name)
            except Exception:
                continue
            if isinstance(flags, (list, np.ndarray)) and len(flags) > 0:
                arr = np.asarray(flags, dtype=float)
                if arr.ndim == 1:
                    rates[f"{type(obj).__name__}.{name}"] = float(arr.mean())
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


rates = acceptance_rates(chain)
ess = ess_per_level(chain, burnin)

if not rates:
    print("WARNING: no acceptance-rate attributes found; inspect the chain object.")
if not ess:
    print("WARNING: to_inference_data produced nothing; check the level argument.")

payload = {
    "run_id": np.array(args.run_id),
    "iterations": np.array(args.iterations),
    "burnin": np.array(burnin),
    "subchain_length": np.array(args.subchain_length),
    "runtime_seconds": np.array(runtime),
    "MAP": MAP,
    "meta": np.array(json.dumps({
        "proposal": args.proposal,
        "randomize_subchain_length": bool(args.randomize_subchain_length),
        "n_levels": n_levels,
        "acceptance": rates,
    })),
}
for k, v in rates.items():
    payload[f"acc__{k}"] = np.array(v)
for k, v in ess.items():
    payload[f"ess__{k}"] = np.asarray(v)

outfile = f"{args.outdir.rstrip('/')}/diag_run{args.run_id:03d}.npz"
np.savez(outfile, **payload)

print(f"\nrun {args.run_id}: {runtime/60:.1f} min for {args.iterations} iterations")
for k, v in sorted(rates.items()):
    print(f"  acceptance  {k:45s} {v:.3f}")
for k, v in sorted(ess.items()):
    print(f"  ESS         {k:45s} min={v.min():8.0f}  median={np.median(v):8.0f}")
print(f"saved -> {outfile}")