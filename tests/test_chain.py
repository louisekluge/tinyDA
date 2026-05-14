import pytest

import numpy as np
from scipy.stats import multivariate_normal
import warnings

from tinyDA.chain import Chain, DAChain, MLDAChain
from tinyDA.posterior import Posterior
from tinyDA.proposal import GaussianRandomWalk
from tinyDA.proposal import MLDA
from tinyDA.link import Link
from tinyDA.distributions import DefaultGaussianLogLike
from tinyDA.distributions import AdaptiveGaussianLogLike

#--------------------------------------------------------------------------------------------
np.random.seed(42)

#--------------------------------------------------------------------------------------------
# Simple forward model: identity map
def forward_model(theta):
    return np.array(theta)

# Coarse forward model = identity but scaled (simplest possible coarse model)
def forward_model_coarse(theta):
    return 0.9 * np.array(theta)

#--------------------------------------------------------------------------------------------
# Prior: 2D Gaussian N(0, I)
prior_mu = np.zeros(2)
prior_cov = np.eye(2)
prior = multivariate_normal(mean=prior_mu, cov=prior_cov)

# Synthetic data for likelihood
true_params = np.array([0.2, -0.3])
data = forward_model(true_params) + 0.05 * np.random.randn(2)
likelihood = DefaultGaussianLogLike(data, covariance=0.05 * np.eye(2))
likelihood_coarse = DefaultGaussianLogLike(data, covariance=0.2 * np.eye(2))

likelihood_adaptive = AdaptiveGaussianLogLike(data, covariance=0.05 * np.eye(2))
likelihood_adaptive_coarse = AdaptiveGaussianLogLike(data, covariance=0.2 * np.eye(2))

# Posterior object
posterior = Posterior(prior, likelihood, model=forward_model)
posterior_coarse = Posterior(prior, likelihood_coarse, model=forward_model_coarse)
posterior_fine = posterior
posteriors = [posterior_coarse, posterior_fine]

posterior_adaptive_fine = Posterior(prior, likelihood_adaptive, model=forward_model)
posterior_adaptive_coarse = Posterior(prior, likelihood_adaptive_coarse, model=forward_model_coarse)
posteriors_adaptive = [posterior_adaptive_coarse, posterior_adaptive_fine]

proposal = GaussianRandomWalk(C=np.eye(2)*0.1, adaptive=False)
proposal_da = GaussianRandomWalk(C=np.eye(2)*0.1, adaptive=False)
proposal_mlda_base = GaussianRandomWalk(C=np.eye(2)*0.1, adaptive=False)

#--------------------------------------------------------------------------------------------
#sanity checks

# test mean is correct
def assert_mean_close(samples, target, atol=0.1):
    mean = samples.mean(axis=0)
    assert np.allclose(mean, target, atol=atol), f"Mean {mean} not close to {target}"

# test variance is not zero
def assert_variance_nonzero(samples, min_std=1e-3):
    std = samples.std(axis=0)
    assert np.all(std > min_std), f"Std too small: {std}"

# test chain moves
def assert_chain_moves(samples):
    diffs = np.diff(samples, axis=0)
    assert np.any(np.abs(diffs) > 0), "Chain did not move"

# test acceptance is reasonable
def assert_reasonable_acceptance(chain, min_rate=0.05, max_rate=0.8):
    rate = np.mean(chain.accepted)
    assert min_rate < rate < max_rate, f"Bad acceptance rate: {rate}"

# test acceptance is reasonable for coarse and fine chain
def assert_reasonable_acceptance_da(da_chain,
                                   min_coarse=0.05, max_coarse=0.9,
                                   min_fine=0.01, max_fine=0.8):
    
    coarse_rate = np.mean(da_chain.accepted_coarse)
    fine_rate = np.mean(da_chain.accepted_fine)

    assert min_coarse < coarse_rate < max_coarse, \
        f"Bad coarse acceptance: {coarse_rate}"
    
    assert min_fine < fine_rate < max_fine, \
        f"Bad fine acceptance: {fine_rate}"

# test chain is not too corralated
def assert_not_too_correlated(samples, max_lag1_corr=0.95):
    x = samples[:, 0]  # check one dimension
    corr = np.corrcoef(x[:-1], x[1:])[0, 1]
    assert abs(corr) < max_lag1_corr, f"Too correlated: {corr}"

# compare early and late samples
def assert_stationary(samples):
    n = len(samples)
    first = samples[: n // 2].mean(axis=0)
    second = samples[n // 2 :].mean(axis=0)

    assert np.allclose(first, second, atol=0.2), \
        f"Chain not stationary: {first} vs {second}"
#--------------------------------------------------------------------------------------------
@pytest.mark.parametrize(
    "posterior",
    [
        (posterior),
    ],
)
def test_posterior_create_link(posterior):

    theta0 = np.array([0.1, 0.1])
    link = posterior.create_link(theta0)
    
    assert isinstance(link.parameters, np.ndarray)
    assert isinstance(link.model_output, np.ndarray)
    assert np.isfinite(link.posterior)

#--------------------------------------------------------------------------------------------
# test simple chain (MH)

@pytest.mark.parametrize(
    "posterior, proposal, initial_parameters",
    [
        (posterior, proposal, None),
        (posterior, proposal, np.array([0.4, 1.02])),
    ],
)
def test_chain_constructor(posterior, proposal, initial_parameters):
    
    chain = Chain(posterior, proposal, initial_parameters=initial_parameters)
    
    assert isinstance(chain.chain, list)
    assert all(isinstance(x, Link) for x in chain.chain)
    assert all(isinstance(x.parameters, np.ndarray) for x in chain.chain)
    assert isinstance(chain.posterior, Posterior)
    assert isinstance(chain.proposal, GaussianRandomWalk)
    assert isinstance(chain.initial_parameters, np.ndarray)
    assert all(isinstance(y, float) for y in chain.initial_parameters)
    assert isinstance(chain.accepted, list)
    assert all(isinstance(a, bool) for a in chain.accepted)

#--------------------------------------------------------------------------------------------
# test sample for MH chain

@pytest.mark.parametrize(
    "iterations, progressbar",
    [
        (0, False),
        (100, False),
        (200, True),
    ],
)
def test_sample_for_MHchain(iterations, progressbar):

    chain = Chain(posterior, proposal, initial_parameters=None)
    chain.sample(iterations, progressbar=progressbar)
    
    assert len(chain.chain) == iterations + 1
    assert len(chain.accepted) == iterations + 1
    assert isinstance(chain.chain, list)
    assert all(isinstance(x, Link) for x in chain.chain)
    assert all(isinstance(x.parameters, np.ndarray) for x in chain.chain)
    assert isinstance(chain.accepted, list)
    assert all(isinstance(a, bool) for a in chain.accepted)
    assert isinstance(chain.proposal, GaussianRandomWalk)

#--------------------------------------------------------------------------------------------
# Class DAChain 

@pytest.mark.parametrize(
    "posterior_coarse, posterior_fine, proposal, subchain_length, randomize_subchain_length, initial_parameters, adaptive_error_model, store_coarse_chain",
    [
        (posterior_coarse, posterior_fine, proposal_da, 2, False, np.array([0.6, 0.782]), None, False),
        (posterior_coarse, posterior_fine, proposal_da, 2, True, None, None, True),
    ],
)
def test_DA_chain_constructor(posterior_coarse, posterior_fine, proposal, subchain_length, randomize_subchain_length, initial_parameters, adaptive_error_model, store_coarse_chain):

    da_chain = DAChain(
        posterior_coarse,
        posterior_fine,
        proposal_da,
        subchain_length=subchain_length,
        randomize_subchain_length=randomize_subchain_length,
        initial_parameters=initial_parameters,
        adaptive_error_model=adaptive_error_model,
        store_coarse_chain=store_coarse_chain
    )

    assert isinstance(da_chain.posterior_coarse, Posterior)
    assert isinstance(da_chain.posterior_fine, Posterior)
    assert isinstance(da_chain.proposal, GaussianRandomWalk)
    assert isinstance(da_chain.subchain_length, int)
    assert isinstance(da_chain.randomize_subchain_length, bool)
    
    assert isinstance(da_chain.initial_parameters, np.ndarray)
    assert all(isinstance(y, float) for y in da_chain.initial_parameters)
    if (initial_parameters is not None):
        assert len(da_chain.initial_parameters) == len(initial_parameters)
        np.testing.assert_array_equal(
            da_chain.initial_parameters,
            initial_parameters
        )
    
    assert isinstance(da_chain.chain_coarse, list)
    assert all(isinstance(x, Link) for x in da_chain.chain_coarse)
    assert all(isinstance(x.parameters, np.ndarray) for x in da_chain.chain_coarse)
    assert len(da_chain.chain_coarse) == 1
    
    assert isinstance(da_chain.accepted_coarse, list)
    assert all(isinstance(b, bool) for b in da_chain.accepted_coarse)
    assert len(da_chain.accepted_coarse) == 1
    
    assert isinstance(da_chain.is_coarse, list)
    assert all(isinstance(c, bool) for c in da_chain.is_coarse)
    assert len(da_chain.is_coarse) == 1
    
    assert isinstance(da_chain.promoted_coarse, list)
    assert all(isinstance(x, Link) for x in da_chain.promoted_coarse)
    assert all(isinstance(x.parameters, np.ndarray) for x in da_chain.promoted_coarse)
    assert len(da_chain.promoted_coarse) == 1
    
    assert isinstance(da_chain.subchain_lengths, list)
    assert all(isinstance(c, int) for c in da_chain.subchain_lengths)
    assert len(da_chain.subchain_lengths) == 0
    
    assert isinstance(da_chain.chain_fine, list)
    assert all(isinstance(x, Link) for x in da_chain.chain_fine)
    assert all(isinstance(x.parameters, np.ndarray) for x in da_chain.chain_fine)
    assert len(da_chain.chain_fine) == 1
    
    assert isinstance(da_chain.accepted_fine, list)
    assert all(isinstance(c, bool) for c in da_chain.accepted_fine)
    assert len(da_chain.accepted_fine) == 1
    
    assert isinstance(da_chain.adaptive_error_model, (str, type(None))) 
    #assert isinstance(da_chain.bias, (RecursiveSampleMoments,type(None)))
    assert isinstance(da_chain.store_coarse_chain, bool)

#--------------------------------------------------------------------------------------------
# test sample for DA chain

@pytest.mark.parametrize(
    "iterations, progressbar",
    [
        (0, False),
        (100, False),
        (200, True),
    ],
)
def test_sample_for_DAchain(iterations, progressbar):

    da_chain = DAChain(
        posterior_coarse,
        posterior_fine,
        proposal_da,
        subchain_length=2,
        randomize_subchain_length=True,
        initial_parameters=None,
        adaptive_error_model=None,
        store_coarse_chain=True
    )
    
    da_chain.sample(iterations, progressbar=progressbar)

    100

    assert len(da_chain.chain_coarse) == (iterations * 3) + 1
    assert len(da_chain.accepted_coarse) == (iterations * 3) + 1
    assert len(da_chain.is_coarse) == (iterations * 3) + 1
    assert len(da_chain.promoted_coarse) == iterations + 1
    assert da_chain.subchain_length == 2
    assert len(da_chain.chain_fine) == iterations + 1
    assert len(da_chain.accepted_fine) == iterations + 1
    
    assert isinstance(da_chain.chain_coarse, list)
    assert all(isinstance(x, Link) for x in da_chain.chain_coarse)
    assert all(isinstance(x.parameters, np.ndarray) for x in da_chain.chain_coarse)
    
    
    assert isinstance(da_chain.accepted_coarse, list)
    assert all(isinstance(b, bool) for b in da_chain.accepted_coarse)
    
    
    assert isinstance(da_chain.is_coarse, list)
    assert all(isinstance(c, bool) for c in da_chain.is_coarse)
    
    
    assert isinstance(da_chain.promoted_coarse, list)
    assert all(isinstance(x, Link) for x in da_chain.promoted_coarse)
    assert all(isinstance(x.parameters, np.ndarray) for x in da_chain.promoted_coarse)
    
    
    assert isinstance(da_chain.subchain_lengths, list)
    assert all(isinstance(c, int) for c in da_chain.subchain_lengths)
    
    
    assert isinstance(da_chain.chain_fine, list)
    assert all(isinstance(x, Link) for x in da_chain.chain_fine)
    assert all(isinstance(x.parameters, np.ndarray) for x in da_chain.chain_fine)
    
    
    assert isinstance(da_chain.accepted_fine, list)
    assert all(isinstance(c, bool) for c in da_chain.accepted_fine)

#--------------------------------------------------------------------------------------------
# Class MLDAChain 

@pytest.mark.parametrize(
    "posteriors, proposal, subchain_lengths, randomize_subchain_length, initial_parameters, adaptive_error_model",
    [
        ([posterior_coarse, posterior_fine], proposal_mlda_base, [2], False, None, None),
        ([posterior_coarse, posterior_fine], proposal_mlda_base, [2], False, np.array([0.6, 0.782]), None),
        #(posteriors_adaptive, proposal_mlda_base, [2], False, np.array([0.6, 0.782]), 'state-independent'), test failt wegen ''MLDA' object has no attribute 'bias''
        #([posterior_coarse, posterior_fine], proposal_mlda_base, [2], False, np.array([0.6, 0.782]), 'state-dependent'), funktioniert nicht (chain wird nicht generiert wegen ''MLDAChain' object has no attribute 'bias'')
        
    ],
)
def test_MLDA_chain_constructor(posteriors, proposal, subchain_lengths, randomize_subchain_length, initial_parameters, adaptive_error_model):

    mlda_chain = MLDAChain(
        posteriors=posteriors,
        proposal=proposal,
        subchain_lengths=subchain_lengths,
        randomize_subchain_length=randomize_subchain_length,
        initial_parameters=initial_parameters,
        adaptive_error_model=adaptive_error_model
    )

    assert isinstance(mlda_chain.posterior, Posterior)
    assert isinstance(mlda_chain.level, int)
    assert isinstance(mlda_chain.proposal, MLDA)
    
    assert isinstance(mlda_chain.subchain_length, int)
    assert isinstance(mlda_chain.randomize_subchain_length, bool)
    
    assert isinstance(mlda_chain.initial_parameters, np.ndarray)
    assert all(isinstance(y, float) for y in mlda_chain.initial_parameters)
    if (initial_parameters is not None):
        assert len(mlda_chain.initial_parameters) == len(initial_parameters)
        np.testing.assert_array_equal(
            mlda_chain.initial_parameters,
            initial_parameters
        )
    
    assert isinstance(mlda_chain.chain, list)
    assert all(isinstance(x, Link) for x in mlda_chain.chain)
    assert all(isinstance(x.parameters, np.ndarray) for x in mlda_chain.chain)
    assert len(mlda_chain.chain) == 1
    
    assert isinstance(mlda_chain.accepted, list)
    assert all(isinstance(b, bool) for b in mlda_chain.accepted)
    assert len(mlda_chain.accepted) == 1
    
    assert isinstance(mlda_chain.adaptive_error_model, (str, type(None))) 
    #assert isinstance(da_chain.bias, (RecursiveSampleMoments,type(None)))
    assert isinstance(mlda_chain.store_coarse_chain, bool)

#--------------------------------------------------------------------------------------------
# test sample for MLDA chain

@pytest.mark.parametrize(
    "iterations, progressbar",
    [
        (0, False),
        (100, False),
        (200, True),
    ],
)
def test_sample_for_MLDAchain(iterations, progressbar):

    mlda_chain = MLDAChain(
        posteriors=posteriors,
        proposal=proposal_mlda_base,
        subchain_lengths=[2],
        randomize_subchain_length=False,
        initial_parameters=np.array([0.6, 0.782]),
        adaptive_error_model=None
    )
    
    mlda_chain.sample(iterations, progressbar=progressbar)

    assert isinstance(mlda_chain.proposal, MLDA)
    
    assert isinstance(mlda_chain.chain, list)
    assert all(isinstance(x, Link) for x in mlda_chain.chain)
    assert all(isinstance(x.parameters, np.ndarray) for x in mlda_chain.chain)
    assert len(mlda_chain.chain) == iterations + 1
    
    assert isinstance(mlda_chain.accepted, list)
    assert all(isinstance(b, bool) for b in mlda_chain.accepted)
    assert len(mlda_chain.accepted) == iterations + 1


#--------------------------------------------------------------------------------------------
# sanity checks mh

def test_mh_sample_stats():
    
    chain = Chain(posterior, proposal, initial_parameters=None)
    chain.sample(iterations=1000, progressbar=True)

    samples = np.array([link.parameters for link in chain.chain])

    samples = samples[100:]  # burn-in

    assert_chain_moves(samples)
    assert_variance_nonzero(samples)
    assert_mean_close(samples, target=true_params, atol=0.1)

    assert_reasonable_acceptance(chain)
    assert_not_too_correlated(samples)
    assert_stationary(samples)

#--------------------------------------------------------------------------------------------
# sanity checks da
def test_da_sample_stats():

    da_chain = DAChain(
        posterior_coarse,
        posterior_fine,
        proposal_da,
        subchain_length=2,
        randomize_subchain_length=True,
        initial_parameters=None,
        adaptive_error_model=None,
        store_coarse_chain=True
    )
    
    da_chain.sample(iterations=1000, progressbar=False)

    samples = np.array([link.parameters for link in da_chain.chain_fine])
    samples = samples[100:]  # burn-in

    assert_chain_moves(samples)
    assert_variance_nonzero(samples)
    assert_mean_close(samples, target=true_params, atol=0.1)
    assert_not_too_correlated(samples)
    assert_stationary(samples)
    assert_reasonable_acceptance_da(da_chain)

#--------------------------------------------------------------------------------------------
# sanity checks mlda
def test_mlda_sample_stats():

    mlda_chain = MLDAChain(
        posteriors=posteriors,
        proposal=proposal_mlda_base,
        subchain_lengths=[2],
        randomize_subchain_length=False,
        initial_parameters=None,
        adaptive_error_model=None
    )
    
    mlda_chain.sample(iterations=1000, progressbar=False)

    samples = np.array([link.parameters for link in mlda_chain.chain])
    samples = samples[100:]  # burn-in

    assert_chain_moves(samples)
    assert_variance_nonzero(samples)
    assert_mean_close(samples, target=true_params, atol=0.1)
    assert_not_too_correlated(samples)
    assert_stationary(samples)
    assert_reasonable_acceptance(mlda_chain)
    
    

            
