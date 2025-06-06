"""
Bootstrap Analysis and Visualization for Linear Hugoniot Model Parameters
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from eosuq import least_squares


def draw_samples(
    df: pd.DataFrame,
    n: int = 10_000,
) -> np.ndarray:
    """
    Generate bootstrap samples of linear model parameters (C_0 and S)
    using least squares fitting on resampled data.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing the variables required for least squares fitting.
    n : int, optional
        Number of bootstrap samples to draw (default is 10,000).

    Returns
    -------
    np.ndarray
        Array of shape (n, 2) containing bootstrap estimates of [C_0, S].
    """
    # Pre-allocate space to store bootstrap estimates of C_0 and s
    samples = np.zeros((n, 2))

    for i in range(n):
        # Sample original data with replacement
        df_resample = df.sample(frac=1, replace=True)

        # Compute least squares estimate of C_0 and S
        beta_hat, _ = least_squares.compute_beta_hat(df_resample)

        # Fit the least squares model
        samples[i] = beta_hat

    return samples


def plot_marginal_bootstrap_distributions(samples: np.ndarray) -> None:
    """
    Plot marginal bootstrap distributions for linear model parameters.

    Parameters
    ----------
    samples : np.ndarray
        Array of shape (n, 2) containing bootstrap estimates of [C_0, s].

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """
    # Set-up new figure
    plt.figure(figsize=(10, 4))

    # Main title for the entire plot
    plt.suptitle("Bootstrap Distribution of Linear Model Parameters", fontsize=16)

    # Histogram for intercept
    plt.subplot(1, 2, 1)
    plt.hist(
        samples[:, 0],
        bins=50,
        color="k",
        alpha=0.7,
        density=True,
        histtype="step",
        linewidth=2,
    )
    plt.title("Intercept")
    plt.xlabel("$C_0$")
    plt.ylabel("Density")
    plt.grid(color="lightgrey")

    # Histogram for the slope
    plt.subplot(1, 2, 2)
    plt.hist(
        samples[:, 1],
        bins=50,
        color="k",
        alpha=0.7,
        density=True,
        histtype="step",
        linewidth=2,
    )
    plt.title("Slope")
    plt.xlabel("$S$")
    plt.ylabel("Density")
    plt.grid(color="lightgrey")

    plt.tight_layout()
    plt.show()
