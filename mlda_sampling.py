"""A sampling wrapper for variance-reduction experiments.

alternative to tinyDA.sample() that forces every run through
MLDAChain, including two-level ones, so the only thing that varies across a
sweep is the configuration.

It also keeps exact account of acceptances. 

Returns the same dict shape tinyDA.sample() returns for MLDA, so the existing
extractors work on it unchanged, plus:

    accepted_l{k}_{j}   bool array, aligned with chain_l{k}_{j}
    n_local_l{k}_{j}    how many of that level's links were generated locally
                        rather than copied down from a finer level

Single-chain sequential sampling only; Ray parallelism is not reproduced here.
"""

from __future__ import annotations

import copy
import os
import time
from itertools import compress

import numpy as np
from tinyDA.chain import MLDAChain


def sample_multilevel(
    posteriors,
    proposal,
    iterations,
    subchain_lengths=1,
    randomize_subchain_length=False,
    initial_parameters=None,
    adaptive_error_model=None,
    store_coarse_chain=True,
    n_chains=1,
    progressbar=True,
    block=None,
    report_every=None,
    callback=None,
):
    """Sample a hierarchy of two or more posteriors through MLDAChain.

    Parameters mirror tinyDA.sample() where they overlap. `subchain_lengths`
    may be an int (used for every level) or a list of length
    len(posteriors) - 1, coarse to fine.

    Periodic output
    ---------------
    MLDAChain.sample() is append-only and holds no per-call state, so calling
    it repeatedly continues the chain rather than restarting it. Sampling in
    blocks is therefore exactly equivalent to one long call -- verified
    bit-for-bit on the chains, the acceptance flags and the adaptive error
    model -- which makes it safe to stop between blocks and report or save.

    block : int or None
        Top-level iterations per block. Defaults to whatever `report_every`
        and the callback need, or the whole run if neither is set.
    report_every : int or None
        Print a progress line every this many top-level iterations: elapsed,
        estimated remaining, and exact acceptance per level.
    callback : callable or None
        Called as callback(result, done, total) at each block boundary, with
        `result` the same dict this function returns. Use make_checkpointer()
        for the common case of writing partial output, so a job killed at the
        walltime leaves something behind instead of nothing.
    """
    if not isinstance(posteriors, list) or len(posteriors) < 2:
        raise ValueError(
            "sample_multilevel needs a list of at least two posteriors; "
            "use tinyDA.sample for single-level MCMC"
        )

    n_levels = len(posteriors)
    if isinstance(subchain_lengths, int):
        subchain_lengths = [subchain_lengths] * (n_levels - 1)
    if len(subchain_lengths) != n_levels - 1:
        raise ValueError(
            f"subchain_lengths must have length {n_levels - 1}, "
            f"got {len(subchain_lengths)}"
        )

    # randomize_subchain_length does not randomise the subchain *length*: the
    # subchain always runs its full length and a uniformly random one of the
    # last J states is promoted instead of the last. At J == 1 the draw has a
    # single outcome, so it is a silent no-op -- worth refusing rather than
    # recording a config as randomised when it cannot be.
    if randomize_subchain_length and min(subchain_lengths) < 2:
        raise ValueError(
            "randomize_subchain_length has no effect with a subchain length "
            "of 1; it promotes a uniform draw from the last J states"
        )

    if initial_parameters is None:
        initial_parameters = [None] * n_chains
    elif isinstance(initial_parameters, np.ndarray):
        initial_parameters = [initial_parameters] * n_chains

    # Reporting defaults to one line per percent of the run. progressbar=False
    # means "be quiet", so it silences this too; report_every=0 silences it
    # while leaving the chain-start lines in place.
    if not progressbar:
        report_every = 0
    elif report_every is None:
        report_every = max(1, iterations // 50)

    # Pick a block size that satisfies whichever of the two hooks are in use.
    wanted = [b for b in (block, report_every,
                          getattr(callback, "every", None)) if b]
    step = min(wanted) if wanted else iterations
    step = max(1, min(step, iterations))

    chains = [
        MLDAChain(
            posteriors,
            copy.deepcopy(proposal),
            subchain_lengths,
            randomize_subchain_length,
            initial_parameters[i],
            adaptive_error_model,
            store_coarse_chain,
        )
        for i in range(n_chains)
    ]

    collect = lambda: _collect(chains, n_levels, iterations, subchain_lengths,
                               randomize_subchain_length, store_coarse_chain)

    if progressbar:
        for i in range(n_chains):
            print(f"Sampling chain {i + 1}/{n_chains}")

    tracker = _AcceptanceTracker(chains, n_levels)
    started = time.time()
    done = 0
    last_report, last_report_time = 0, 0.0
    while done < iterations:
        n = min(step, iterations - done)
        for chain in chains:
            chain.sample(n, progressbar=progressbar and step >= iterations)
        done += n
        tracker.update()

        if report_every and (done % report_every == 0 or done == iterations):
            elapsed = time.time() - started
            remaining = elapsed * (iterations - done) / done
            # Rate over this reporting interval, not the whole run: a run that
            # is slowing down shows it here, where a cumulative average hides
            # it behind the fast early iterations.
            recent = (done - last_report) / max(elapsed - last_report_time, 1e-9)
            print(f"  {100 * done // iterations:3d}%  {done}/{iterations}  "
                  f"{elapsed/60:6.1f} min elapsed, ~{remaining/60:6.1f} left  "
                  f"{recent:7.1f} it/s  "
                  + "  ".join(f"{k}={v:.3f}" for k, v in sorted(tracker.rates().items())),
                  flush=True)
            last_report, last_report_time = done, elapsed

        if callback is not None:
            every = getattr(callback, "every", None)
            if every is None or done % every == 0 or done == iterations:
                callback(collect(), done, iterations)

    return collect()


class _AcceptanceTracker:
    """Running acceptance per level, updated incrementally.

    Reporting a hundred times over a long run must not cost O(chain length)
    each time, or progress reporting is itself quadratic. Each level keeps a
    cursor into its acceptance list and only the entries added since the last
    update are counted.

    The first entry of every level's list is the seeded initial state rather
    than a proposal, so cursors start past it.
    """

    def __init__(self, chains, n_levels):
        self._nodes = []           # (level, chain index, node, has_is_local)
        for j, chain in enumerate(chains):
            self._nodes.append((n_levels - 1, j, chain, False))
            node = chain.proposal
            for level in reversed(range(n_levels - 1)):
                self._nodes.append((level, j, node, True))
                node = node.proposal
        self._cursor = {(lv, j): 1 for lv, j, _, _ in self._nodes}
        self._accepted = {(lv, j): 0 for lv, j, _, _ in self._nodes}
        self._total = {(lv, j): 0 for lv, j, _, _ in self._nodes}

    def update(self):
        for level, j, node, has_is_local in self._nodes:
            key = (level, j)
            flags = node.accepted
            start = self._cursor[key]
            if start >= len(flags):
                continue
            new = flags[start:len(flags)]
            if has_is_local:
                # Only locally generated links count; the rest were copied
                # down from a finer level by align_chain and carry that
                # level's acceptance, not this one's.
                mask = list(node.is_local)[start:len(flags)]
                new = [a for a, local in zip(new, mask) if local]
            self._accepted[key] += int(sum(bool(a) for a in new))
            self._total[key] += len(new)
            self._cursor[key] = len(flags)

    def rates(self):
        out = {}
        for level in sorted({lv for lv, _, _, _ in self._nodes}):
            accepted = sum(v for (lv, _), v in self._accepted.items() if lv == level)
            total = sum(v for (lv, _), v in self._total.items() if lv == level)
            if total:
                out[f"level{level}"] = accepted / total
        return out


def make_checkpointer(path, every=None, store_model_output=False):
    """A callback that writes the chain so far, atomically.

    Partial output beats none when a job hits the walltime, but a process
    killed midway through a write leaves a truncated file that reads as
    corrupt. Writing to a temporary name and renaming makes the replacement
    atomic, so the checkpoint on disk is always a complete earlier one.

    Set `every` to checkpoint less often than the sampling block; writing is
    O(chain length), so on a long run it is worth doing sparingly.
    """

    def checkpoint(result, done, total):
        store = {
            "done": np.array(done),
            "total": np.array(total),
            "levels": np.array(result["levels"]),
            "subchain_lengths": np.array(result["subchain_lengths"]),
            "complete": np.array(done >= total),
        }
        for key, value in result.items():
            if key.startswith("accepted_"):
                store[key] = value
            elif key.startswith(("chain_l", "promoted_l")) and value is not None:
                for name, arr in _links_to_arrays(value, store_model_output).items():
                    store[f"{key}__{name}"] = arr
        target = path if path.endswith(".npz") else path + ".npz"
        tmp = f"{target}.{os.getpid()}.tmp"
        os.makedirs(os.path.dirname(os.path.abspath(target)), exist_ok=True)
        # Write through a file handle: np.savez_compressed silently appends
        # '.npz' to a path that lacks it, which would rename the temp file out
        # from under the os.replace below.
        with open(tmp, "wb") as fh:
            np.savez_compressed(fh, **store)
        os.replace(tmp, target)

    checkpoint.every = every
    return checkpoint


def _links_to_arrays(links, with_model_output=False):
    """Link list -> {attribute: (n, d) float64 array}.

    Parameters stay float64 because the pairing test is an exact equality on
    them. model_output is optional: it is the largest array by an order of
    magnitude and no estimator formulation needs it.
    """
    out = {
        "parameters": np.array([l.parameters for l in links], dtype=np.float64),
        "qoi": np.array([np.atleast_1d(l.qoi) for l in links], dtype=np.float64),
        "stats": np.array([[l.prior, l.likelihood, l.posterior] for l in links],
                          dtype=np.float64),
    }
    if with_model_output:
        out["model_output"] = np.array(
            [np.atleast_1d(l.model_output) for l in links], dtype=np.float64)
    return out


def _collect(chains, n_levels, iterations, subchain_lengths,
             randomize_subchain_length, store_coarse_chain):
    """Build tinyDA's MLDA result dict, plus the acceptance flags."""
    result = {
        "sampler": "MLDA",
        "n_chains": len(chains),
        "iterations": iterations + 1,
        "levels": n_levels,
        "subchain_lengths": subchain_lengths,
        "randomize_subchain_length": randomize_subchain_length,
    }

    top = n_levels - 1
    for j, chain in enumerate(chains):
        result[f"chain_l{top}_{j}"] = chain.chain
        result[f"accepted_l{top}_{j}"] = np.asarray(chain.accepted, dtype=bool)
        result[f"n_local_l{top}_{j}"] = len(chain.chain)

    # Walk down the nested MLDA proposals, one level per step.
    current = [chain.proposal for chain in chains]
    for level in reversed(range(n_levels - 1)):
        for j, node in enumerate(current):
            if not store_coarse_chain:
                result[f"chain_l{level}_{j}"] = None
                result[f"promoted_l{level}_{j}"] = None
                continue
            # A level's chain also receives links copied down from the finer
            # level by align_chain. is_local marks the ones this level
            # generated, and tinyDA exports only those -- so the acceptance
            # flags have to be filtered the same way or they will not line up.
            is_local = list(node.is_local)
            result[f"chain_l{level}_{j}"] = list(compress(node.chain, is_local))
            result[f"promoted_l{level}_{j}"] = node.promoted
            accepted = list(node.accepted)
            if len(accepted) != len(is_local):
                raise RuntimeError(
                    f"level {level}: {len(accepted)} acceptance flags against "
                    f"{len(is_local)} links -- tinyDA's bookkeeping has changed"
                )
            result[f"accepted_l{level}_{j}"] = np.asarray(
                list(compress(accepted, is_local)), dtype=bool)
            result[f"n_local_l{level}_{j}"] = int(np.sum(is_local))
        current = [node.proposal for node in current]

    return result


def acceptance_rates(chain, burnin=0):
    """Exact post-burn-in acceptance per level, from the recorded flags.

    burnin is in top-level samples and is scaled to each level by the ratio of
    chain lengths, so every level is cut at the same point in the run.
    """
    n_levels = chain["levels"]
    n_top = len(chain[f"accepted_l{n_levels - 1}_0"])
    rates = {}
    for level in range(n_levels):
        flags = chain.get(f"accepted_l{level}_0")
        if flags is None:
            continue
        # The first entry of every level's list is the seeded initial state,
        # not a proposal, and is always True. Skipping it keeps this in step
        # with the running figures printed during sampling.
        cut = max(1, int(round(burnin * len(flags) / n_top)))
        post = flags[cut:]
        if post.size:
            rates[f"level{level}"] = float(post.mean())
    return rates


def pairing_fractions(chain, burnin=0):
    """Fraction of exported pairs sitting at a common parameter value.

    The correction Y_l = Q_l - Q_{l-1} is only a difference if both terms are
    evaluated at the same theta. The top-level chain carries its initial
    state, which no proposal produced, so it is one longer than its promoted
    partner and must be offset by one; lower levels pair directly.
    """
    n_levels = chain["levels"]
    out = {}
    for level in range(n_levels - 1):
        fine = chain.get(f"chain_l{level + 1}_0")
        promoted = chain.get(f"promoted_l{level}_0")
        if fine is None or promoted is None:
            continue
        a = np.array([l.parameters for l in fine], dtype=float)
        b = np.array([l.parameters for l in promoted], dtype=float)
        if len(a) == len(b) + 1:
            a = a[1:]
        n = min(len(a), len(b))
        cut = int(round(burnin * n / len(a))) if len(a) else 0
        matched = np.all(a[cut:n] == b[cut:n], axis=1)
        out[f"Y_{level + 1}{level}"] = float(matched.mean()) if matched.size else float("nan")
    return out