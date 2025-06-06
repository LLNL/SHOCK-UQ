"""
Bayesian Calibration and MCMC Sampling for Linear Hugoniot Model
"""

import corner
import emcee
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import invgamma, norm, multivariate_t
from statsmodels.tsa.stattools import acf


def log_likelihood(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.float64:
    """
    Compute the log-likelihood of the linear Hugoniot model parameters.

    Parameters
    ----------
    theta : np.ndarray
        Model parameters [c0, s, sigma_sq].
    x : np.ndarray
        Input data (predictor variable).
    y : np.ndarray
        Observed data (response variable).

    Returns
    -------
    np.float64
        Log-likelihood value.
    """
    c0, s, sigma_sq = theta
    if sigma_sq <= 0:  # Only accept positive variances
        return np.float64(-np.inf)
    sigma = np.sqrt(sigma_sq)  # Standard deviation is the square root of variance
    prediction_mean = c0 * x + s
    loglik = np.sum(norm.logpdf(y, loc=prediction_mean, scale=sigma))
    return loglik


def log_prior(theta: np.ndarray) -> np.float64:
    """
    Compute the log of the prior distribution for the model parameters.

    Parameters
    ----------
    theta : np.ndarray
        Model parameters [c0, s, sigma_sq].

    Returns
    -------
    np.float64
        Log-prior value.
    """
    _, _, sigma_sq = theta
    if sigma_sq <= 0:  # Ensures variance is positive
        return np.float64(-np.inf)
    return -np.log(sigma_sq)


def log_posterior(
    theta: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.float64:
    """
    Compute the log of the posterior distribution of the model parameters.

    Parameters
    ----------
    theta : np.ndarray
        Model parameters [c0, s, sigma_sq].
    x : np.ndarray
        Input data (predictor variable).
    y : np.ndarray
        Observed data (response variable).

    Returns
    -------
    np.float64
        Log-posterior value.
    """
    return log_prior(theta) + log_likelihood(theta, x, y)


def get_samples(
    df: pd.DataFrame,
    nsteps: int,
    nwalkers: int,
    ndim: int = 3,
    burn_in: int = 10_000,
    thin: int = 1,
) -> np.ndarray:
    """
    Run MCMC sampling for the linear Hugoniot model.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the input and response data.
    nsteps : int
        Number of MCMC steps.
    nwalkers : int
        Number of MCMC walkers.
    ndim : int, optional
        Number of model parameters (default is 3).
    burn_in : int, optional
        Number of burn-in samples to discard (default is 10,000).
    thin : int, optional
        Thinning interval for the samples (default is 1).

    Returns
    -------
    samples : np.ndarray
        Array of MCMC samples after burn-in and thinning.
    acceptance_fraction : np.ndarray
        Array of shape (nwalkers,) with the acceptance fraction for each walker.
    """
    # Prepare data
    X = df["Up_km_s"].values
    y = df["Us_km_s"].values

    # Initial starting values for chains
    pos = [
        np.array([2, 1.0, 2]) + 1e-4 * np.random.randn(ndim) for _ in range(nwalkers)
    ]

    # Create the sampler
    sampler = emcee.EnsembleSampler(nwalkers, ndim, log_posterior, args=(X, y))

    # Run the MCMC
    sampler.run_mcmc(pos, nsteps)

    # Extract samples and discard burn-in
    samples = sampler.get_chain(thin=thin, discard=burn_in)

    return samples, sampler.acceptance_fraction


def trace_plot(samples: np.ndarray) -> None:
    """
    Plot trace plots for MCMC samples of each model parameter.

    Parameters
    ----------
    samples : np.ndarray
        Array of MCMC samples with shape (n_samples, n_parameters).

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """
    # Plot trace plots
    fig, axes = plt.subplots(3, figsize=(10, 7), sharex=True)
    param_labels = ["$S$", "$C_0$", r"$\sigma^2$"]
    for i in range(3):
        axes[i].plot(samples[:, i], color="b", alpha=0.3)
        axes[i].set_ylabel(param_labels[i])
    axes[-1].set_xlabel("Iteration")
    plt.suptitle("MCMC Samples of Model Parameters")
    plt.tight_layout()
    plt.show()


def plot_samples_with_marginals(
    samples: np.ndarray,
    beta_hat: np.ndarray,
    Sigma: np.ndarray,
    nu: int,
) -> None:
    """
    Plot histograms of MCMC samples and overlay marginal posterior densities.

    Parameters
    ----------
    samples : np.ndarray
        Array of MCMC samples.
    beta_hat : np.ndarray
        Posterior mean estimates for model parameters.
    Sigma : np.ndarray
        Posterior covariance matrix for model parameters.
    nu : int
        Degrees of freedom for the posterior.

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """
    # Define grids, indices, and labels for each variable
    grids = [np.linspace(1.49, 1.54, 6_000), np.linspace(3.86, 3.94, 6_000)]
    indices = [0, 1]
    labels = ["$S$", "$C_0$"]

    # Plot histograms of the samples with actual posterior density
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

    legend_handles = []
    legend_labels = []

    for ax, grid, idx, label in zip(axes, grids, indices, labels):
        # Plot histogram
        ax.hist(
            samples[:, idx],
            bins=50,
            color="k",
            alpha=0.7,
            density=True,
            histtype="step",
            linewidth=2,
            label="MCMC Sample Histogram",
        )

        # Marginal posterior density
        rv = multivariate_t(loc=beta_hat[1 - idx], shape=Sigma[1 - idx, 1 - idx], df=nu)
        ax.plot(
            grid,
            rv.pdf(grid),
            color="red",
            lw=2,
            label="Posterior Density",
        )

        ax.set_xlabel(label)
        ax.set_ylabel("Posterior Density")
        # Collect handles/labels only from the first axis (they are the same for both)
        if idx == 0:
            handles, labels_ = ax.get_legend_handles_labels()
            legend_handles.extend(handles)
            legend_labels.extend(labels_)

    fig.suptitle("Marginal Posterior Distributions of $C_0$ and $S$")

    # Add a single legend below both subplots
    fig.legend(
        legend_handles,
        legend_labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.05),
        ncol=2,
        frameon=False,
    )

    plt.tight_layout()
    plt.show()


def posterior_pairs_plot(
    samples: np.ndarray,
    beta_hat: np.ndarray,
) -> None:
    """
    Generate a corner plot of the posterior samples and overlay posterior mean
    estimates.

    Parameters
    ----------
    samples : np.ndarray
        Array of MCMC samples.
    beta_hat : np.ndarray
        Posterior mean estimates for model parameters.

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """
    labels = ["$C_0$", "$S$", r"$\sigma^2$"]
    corner.corner(
        samples,
        labels=labels,
        truths=[beta_hat[1], beta_hat[0], 0.0045],
    )


def plot_marginal_density_sigma_sq(
    df: int,
    s_sq: np.float64,
    samples: np.ndarray,
) -> None:
    """
    Plot the marginal posterior distribution of sigma squared with a histogram
    of the MCMC samples.

    Parameters
    ----------
    df : int
        Degrees of freedom for the inverse gamma distribution.
    s_sq : np.float64
        Scale parameter for the inverse gamma distribution.
    samples : np.ndarray
        Array of MCMC samples for sigma squared.

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """
    nu = df.shape[0] - 2  # Degrees of freedom
    a = nu / 2
    b = nu / 2 * s_sq

    # Create the histogram
    plt.hist(
        samples[:, 2],
        bins=40,
        density=True,
        alpha=0.6,
        color="black",
        histtype="step",
        linewidth=2,
        label="Histogram of MCMC Samples",
    )

    # Evaluate the inverse gamma distribution
    x = np.linspace(min(samples[:, 2]), max(samples[:, 2]), 1000)
    posterior_pdf = invgamma.pdf(x, a, scale=b)

    # Overlay the inverse gamma distribution
    plt.plot(x, posterior_pdf, "r-", label=f"Inverse Gamma(a={a:.2f}, b={b:.2f})")
    plt.xlabel("Value")
    plt.ylabel("Density")
    plt.title(r"Posterior Distribution of $\sigma^2$")
    plt.legend()
    plt.show()


def plot_acf_of_chains(
    samples: np.ndarray,
    nlags: int = 50,
    figsize: tuple[int, int] = (15, 4),
) -> None:
    """
    Plot autocorrelation function (ACF) for each parameter chain from MCMC samples.

    Parameters
    ----------
    samples : np.ndarray
        3D array of MCMC samples (n_samples, n_other_dim, n_chains).
    nlags : int, optional
        Number of lags to compute the autocorrelation (default is 50).
    figsize : tuple[int, int], optional
        Figure size for the plot (default is (15, 4)).

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """
    if samples.ndim != 3:
        raise ValueError(
            "samples array must be 3-dimensional (n_samples, n_other_dim, n_chains)"
        )

    param_labels = ["$S$", "$C_0$", r"$\sigma^2$"]

    fig, axes = plt.subplots(1, 3, figsize=figsize, sharey=True)

    plt.suptitle("Autocorrelation (ACF) Plots of MCMC Samples")

    for i in range(3):
        acf_vals = acf(samples[:, 0, i], nlags=nlags, fft=True)
        axes[i].stem(range(len(acf_vals)), acf_vals, basefmt=" ")
        axes[i].set_title(param_labels[i])
        axes[i].set_xlabel("Lag")
        if i == 0:
            axes[i].set_ylabel("Autocorrelation")

    plt.tight_layout()
    plt.show()
