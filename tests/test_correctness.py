
import numpy as np
import pytest
from scipy.stats import multivariate_normal, chisquare

from tinyDA.chain import DAChain, MLDAChain
from tinyDA.posterior import Posterior
from tinyDA.proposal import GaussianRandomWalk
from tinyDA.distributions import DefaultGaussianLogLike

# --------------------------------------------------------------------------------------------
# Linear-Gaussian problem with an analytic posterior.
# prior N(0, I), fine model G(theta) = theta, noise N(0, 0.1 I), data y = (1, 1)
# => posterior N(10/11, 1/11) in each dimension.
# The coarse models are deliberately biased so that a wrong acceptance ratio
# shows up as a biased fine-level posterior.

NOISE = 0.1
DATA = np.ones(2)
POST_MEAN = (DATA / NOISE) / (1 + 1 / NOISE)
POST_VAR = 1 / (1 + 1 / NOISE)


def model_fine(theta):
    return np.asarray(theta, dtype=float)


def model_medium(theta):
    return 0.75 * np.asarray(theta, dtype=float) + 0.25


def model_coarse(theta):
    return 0.5 * np.asarray(theta, dtype=float) + 0.5


def make_posterior(model):
    prior = multivariate_normal(mean=np.zeros(2), cov=np.eye(2))
    likelihood = DefaultGaussianLogLike(DATA, covariance=NOISE * np.eye(2))
    return Posterior(prior, likelihood, model=model)


def make_proposal():
    return GaussianRandomWalk(C=0.3 * np.eye(2), adaptive=False)


def make_da(L, randomize, store=True):
    return DAChain(
        make_posterior(model_coarse),
        make_posterior(model_fine),
        make_proposal(),
        subchain_length=L,
        randomize_subchain_length=randomize,
        initial_parameters=np.zeros(2),
        store_coarse_chain=store,
    )


def make_mlda(Ls, randomize, store=True):
    return MLDAChain(
        [make_posterior(model_coarse), make_posterior(model_medium), make_posterior(model_fine)],
        make_proposal(),
        subchain_lengths=Ls,
        randomize_subchain_length=randomize,
        initial_parameters=np.zeros(2),
        store_coarse_chain=store,
    )


def batch_means_se(x, n_batches=25):
    n = len(x) // n_batches * n_batches
    means = x[:n].reshape(n_batches, -1).mean(axis=1)
    return means.std(ddof=1) / np.sqrt(n_batches)


def assert_matches_posterior(samples, n_sigma=4.0):
    for d in range(samples.shape[1]):
        x = samples[:, d]
        se_mean = batch_means_se(x)
        assert abs(x.mean() - POST_MEAN[d]) < n_sigma * se_mean, (
            f"dim {d}: mean {x.mean():.4f} vs {POST_MEAN[d]:.4f} (se {se_mean:.4f})"
        )
        sq = (x - POST_MEAN[d]) ** 2
        se_var = batch_means_se(sq)
        assert abs(sq.mean() - POST_VAR) < n_sigma * se_var, (
            f"dim {d}: var {sq.mean():.4f} vs {POST_VAR:.4f} (se {se_var:.4f})"
        )

        # --------------------------------------------------------------------------------------------
# 1. Exact checks of what enters the acceptance ratio.


def spy_on_acceptance(level):
    """Wrap level.get_acceptance and record the arguments and the promoted link."""
    calls = []
    original = level.get_acceptance

    def wrapped(proposal_link, previous_link, proposal_link_below, previous_link_below):
        calls.append(
            (proposal_link, previous_link, proposal_link_below, previous_link_below,
             level.promoted[-1])
        )
        return original(proposal_link, previous_link, proposal_link_below, previous_link_below)

    level.get_acceptance = wrapped
    return calls


@pytest.mark.parametrize("randomize", [False, True])
def test_mlda_acceptance_uses_promoted_link(randomize):
    np.random.seed(1)
    chain = make_mlda([3, 3], randomize)
    calls_top = spy_on_acceptance(chain.proposal)              # used by MLDAChain.sample
    calls_mid = spy_on_acceptance(chain.proposal.proposal)     # used by MLDA.make_mlda_proposal
    chain.sample(300, progressbar=False)

    for calls in (calls_top, calls_mid):
        assert len(calls) > 0
        for proposal, previous, proposal_below, previous_below, promoted in calls:
            # the coarse proposal in the ratio is the link that was promoted ...
            assert proposal_below is promoted
            # ... and the fine proposal was created from exactly those parameters.
            assert proposal.parameters is proposal_below.parameters
            # the current fine state is the state the coarse subchain started from.
            assert previous.parameters is previous_below.parameters


def test_da_acceptance_uses_promoted_link():
    np.random.seed(2)
    L = 4
    chain = make_da(L, randomize=True)
    seen = []
    original = chain._get_state_independent_acceptance

    def wrapped(proposal_link_fine):
        assert proposal_link_fine.parameters is chain.promoted_coarse[-1].parameters
        assert chain.chain_fine[-1].parameters is chain.chain_coarse[-(L + 1)].parameters
        seen.append(True)
        return original(proposal_link_fine)

    chain._get_state_independent_acceptance = wrapped
    chain.sample(300, progressbar=False)
    assert len(seen) > 0


# --------------------------------------------------------------------------------------------
# 2. Bookkeeping invariants (exact indexing of the coarse chain).


@pytest.mark.parametrize("L, randomize", [(1, False), (3, False), (3, True), (5, True)])
def test_da_promoted_bookkeeping(L, randomize):
    np.random.seed(3)
    n = 300
    chain = make_da(L, randomize)
    chain.sample(n, progressbar=False)

    assert len(chain.chain_fine) == n + 1
    assert len(chain.promoted_coarse) == n + 1
    assert len(chain.subchain_lengths) == n
    assert len(chain.chain_coarse) == 1 + n * (L + 1)
    assert all(1 <= J <= L for J in chain.subchain_lengths)
    if not randomize:
        assert set(chain.subchain_lengths) == {L}

    for k in range(n):
        start = k * (L + 1)          # first link of subchain k
        realign = (k + 1) * (L + 1)  # link appended after the fine accept/reject
        J = chain.subchain_lengths[k]

        # the promoted link is the J-th state of the subchain.
        assert chain.promoted_coarse[k + 1] is chain.chain_coarse[start + J]
        # the coarse chain is re-aligned with the fine state.
        assert chain.chain_coarse[realign].parameters is chain.chain_fine[k + 1].parameters
        # an accepted fine move is exactly the promoted coarse state.
        if chain.accepted_fine[k + 1]:
            assert chain.chain_fine[k + 1].parameters is chain.promoted_coarse[k + 1].parameters
        # if nothing moved before step J, the fine state cannot have been "accepted".
        if chain.promoted_coarse[k + 1] is chain.chain_coarse[start]:
            assert not chain.accepted_fine[k + 1]


@pytest.mark.parametrize("randomize", [False, True])
def test_mlda_promoted_lengths(randomize):
    np.random.seed(4)
    n = 100
    chain = make_mlda([2, 3], randomize)
    chain.sample(n, progressbar=False)
    # one promoted link per call from the next-finer level.
    assert len(chain.proposal.promoted) == n * 3
    assert len(chain.proposal.proposal.promoted) == n * 3 * 2


# --------------------------------------------------------------------------------------------
# 3. Regressions and validation.


def test_mlda_without_storing_coarse_chain_runs():
    np.random.seed(5)
    chain = make_mlda([2, 2], randomize=False, store=False)
    chain.sample(20, progressbar=False)
    assert len(chain.chain) == 21


@pytest.mark.parametrize("factory", [lambda: make_da(1, True), lambda: make_mlda([1, 2], True)])
def test_randomize_requires_subchain_length_above_one(factory):
    with pytest.raises(ValueError):
        factory()


@pytest.mark.parametrize("factory", [lambda: make_da(3, True, store=False),
                                     lambda: make_mlda([2, 2], True, store=False)])
def test_randomize_requires_stored_coarse_chain(factory):
    with pytest.raises(ValueError):
        factory()


def test_random_proposal_index_is_uniform():
    np.random.seed(6)
    L = 5
    chain = make_da(L, randomize=True)
    draws = np.array([chain._get_random_proposal_index() for _ in range(20000)])
    assert draws.min() == -L and draws.max() == -1
    counts = np.bincount(draws + L, minlength=L)
    assert chisquare(counts).pvalue > 1e-3


# --------------------------------------------------------------------------------------------
# 4. End-to-end: fine-level samples match the analytic posterior.


@pytest.mark.parametrize("randomize", [False, True])
def test_da_matches_analytic_posterior(randomize):
    np.random.seed(7)
    chain = make_da(3, randomize)
    chain.sample(6000, progressbar=False)
    samples = np.array([link.parameters for link in chain.chain_fine[500:]])
    assert_matches_posterior(samples)


@pytest.mark.parametrize("randomize", [False, True])
def test_mlda_matches_analytic_posterior(randomize):
    np.random.seed(8)
    chain = make_mlda([3, 3], randomize)
    chain.sample(4000, progressbar=False)
    samples = np.array([link.parameters for link in chain.chain[500:]])
    assert_matches_posterior(samples)