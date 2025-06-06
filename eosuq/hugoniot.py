"""
Hugoniot Curve Sampling and Visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import UnivariateSpline
from scipy.stats import multivariate_t


def compute_Hugoniot(
    df: pd.DataFrame,
    Up_grid: np.ndarray,
    beta: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute the Hugoniot curve in the pressure-volume plane for a given set
    of parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental data with columns 'Up_km_s' and 'rho0_g_cc'.
    Up_grid : np.ndarray
        Array of particle velocities (Up) at which to evaluate the Hugoniot.
    beta : np.ndarray
        Model parameters for shock velocity (Us = beta[0] + beta[1] * Up).

    Returns
    -------
    P : np.ndarray
        Array of computed pressures (GPa) along the Hugoniot.
    V : np.ndarray
        Array of computed specific volumes (cc/g) along the Hugoniot.
    """
    # Evaluate shock wave velocity on a grid
    Us_grid = beta[0] + beta[1] * Up_grid

    # Compute initial volume for points on Up_grid
    spline = UnivariateSpline(df["Up_km_s"], df["rho0_g_cc"], s=0.05)
    rho0_grid = spline(Up_grid)
    V0_grid = 1 / rho0_grid

    # Set initial pressure
    P0 = 0.0001  # 1 bar = 1e-4 GPa

    # Compute Hugoniot for sampled beta
    V = V0_grid * (Us_grid - Up_grid) / Us_grid
    P = P0 + rho0_grid * Us_grid * Up_grid

    return P, V


def sample_Hugoniot(
    df: pd.DataFrame,
    beta_hat: np.ndarray,
    Sigma: np.ndarray,
    nu: int,
    n_sample: int,
) -> pd.DataFrame:
    """
    Sample Hugoniot curves from the posterior distribution of model parameters.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental data with columns 'Up_km_s' and 'rho0_g_cc'.
    beta_hat : np.ndarray
        Posterior mean of the model parameters.
    Sigma : np.ndarray
        Posterior covariance matrix of the model parameters.
    nu : int
        Degrees of freedom for the multivariate t-distribution.
    n_sample : int
        Number of parameter samples to draw.

    Returns
    -------
    samples_df : pd.DataFrame
        DataFrame containing sampled Hugoniot curves. Each row contains:
        - 'sample_index': Index of the sample
        - 'V': Specific volume array for the sample
        - 'P': Pressure array for the sample
    """
    # Up grid on which to evaluate Hugoniot
    Up_grid = np.linspace(
        start=0.95 * df["Up_km_s"].min(),
        stop=1.05 * df["Up_km_s"].max(),
        num=100,
    )

    # Sample betas from posterior distribution
    rv = multivariate_t(loc=beta_hat, shape=Sigma, df=nu)
    betas = rv.rvs(size=n_sample, random_state=42)

    # Create a list to store results for all samples
    results = []

    for idx, beta in enumerate(betas):
        # Evaluate Hugoniot in P-V plane for beta sample
        P, V = compute_Hugoniot(df, Up_grid, beta)

        # Store the Hugoniot in a dictionary
        results.append(
            {
                "sample_index": idx,
                "V": V,
                "P": P,
            }
        )

    # Add Hugoniot for beta_hat as the final sample
    P, V = compute_Hugoniot(df, Up_grid, beta_hat)
    results.append(
        {
            "sample_index": idx + 1,
            "V": V,
            "P": P,
        }
    )

    # Convert results to a DataFrame
    samples_df = pd.DataFrame(results)

    return samples_df


def plot_Hugoniot_samples(
    df: pd.DataFrame,
    beta_hat: np.ndarray,
    Sigma: np.ndarray,
    nu: int,
    n_sample: int,
) -> None:
    """
    Plot sampled Hugoniot curves in the pressure-volume plane with the
    experimental data.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing experimental data with columns 'V_cc_g' and 'P_GPa'.
    beta_hat : np.ndarray
        Posterior mean of the model parameters.
    Sigma : np.ndarray
        Posterior covariance matrix of the model parameters.
    nu : int
        Degrees of freedom for the multivariate t-distribution.
    n_sample : int
        Number of parameter samples to draw and plot.

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """
    # Sample Hugoniot curves in (P, V) space
    Hugoniot_samples = sample_Hugoniot(df, beta_hat, Sigma, nu, n_sample)

    # Plot the Hugoniot curves
    plt.figure(figsize=(5, 5))
    for idx, row in Hugoniot_samples.iterrows():
        color = "gray" if idx != Hugoniot_samples.shape[0] - 1 else "orange"
        plt.plot(row["V"], row["P"], c=color)
    # Overlay experimental data
    plt.scatter(
        df["V_cc_g"],
        df["P_GPa"],
        s=40,
        marker="o",
        zorder=2,
        alpha=0.8,
        facecolor="none",
        edgecolors="blue",
    )
    plt.title("Hugoniot from Posterior Distribution", fontsize=15)
    plt.xlabel("Volume [cc/g]", fontsize=14)
    plt.ylabel("Pressure [GPa]", fontsize=14)
    plt.tick_params(axis="both", labelsize=13)
    plt.grid(True)
    plt.tight_layout()
    plt.show()
