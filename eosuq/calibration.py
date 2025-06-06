"""
Posterior Inference and Visualization for Linear Hugoniot Model Parameters
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scipy.stats import multivariate_t

from eosuq import least_squares


def compute_posterior_parameters(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.float64, int]:
    """
    Compute the posterior parameters for the linear Hugoniot model.

    This function calculates the least squares estimates of the regression coefficients,
    the posterior scale matrix, the residual variance, and the degrees of freedom for
    the model, given a DataFrame of measurements.

    Parameters
    ----------
    df : pd.DataFrame
        Input data containing the independent and dependent variables.

    Returns
    -------
    beta_hat : np.ndarray
        Estimated regression coefficients (posterior mean).
    scale_matrix : np.ndarray
        Posterior scale matrix for the coefficients.
    s_sq : np.float64
        Estimated residual variance.
    nu : int
        Degrees of freedom for the posterior distribution.
    """
    # Compute the least squares estimate of regression coefficients
    beta_hat, X = least_squares.compute_beta_hat(df)

    # Degrees of freedom
    nu = df.shape[0] - 2

    # Predicted values
    y_hat = X @ beta_hat

    # Residual variance estimate
    s_sq = np.sum((df["Us_km_s"] - y_hat) ** 2) / nu

    # Scale matrix for the posterior distribution
    XTX_inv = np.linalg.inv(X.T @ X)
    scale_matrix = s_sq * XTX_inv

    return beta_hat, scale_matrix, s_sq, nu


def evaluate_joint_pdf(
    beta_hat: np.ndarray,
    Sigma: np.ndarray,
    nu: int,
    c0_range: tuple[float, float] = (3.86, 3.94),
    s_range: tuple[float, float] = (1.49, 1.54),
    ngrid: int = 3000,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate the bivariate t-distribution PDF for linear Hugoniot model parameters on a grid.

    This function creates a meshgrid over the specified parameter ranges and evaluates
    the probability density function (PDF) of a bivariate Student's t-distribution
    at each point in the grid.

    Parameters
    ----------
    beta_hat : np.ndarray
        Mean vector (posterior mean) of the regression coefficients, shape (2,).
    Sigma : np.ndarray
        Scale (covariance) matrix of the distribution, shape (2, 2).
    nu : int
        Degrees of freedom for the t-distribution.
    c0_range : tuple[float, float], optional
        Tuple specifying the (min, max) range for the first parameter ($C_0$).
        Default is (3.86, 3.94).
    s_range : tuple[float, float], optional
        Tuple specifying the (min, max) range for the second parameter ($S$).
        Default is (1.49, 1.54).
    ngrid : int, optional
        Number of points in each dimension of the grid. Default is 3000.

    Returns
    -------
    X : np.ndarray
        2D array of grid values for the first parameter ($C_0$).
    Y : np.ndarray
        2D array of grid values for the second parameter ($S$).
    Z : np.ndarray
        2D array of evaluated PDF values at each (X, Y) grid point.
    """
    c0_grid = np.linspace(*c0_range, ngrid)
    s_grid = np.linspace(*s_range, ngrid)
    X, Y = np.meshgrid(c0_grid, s_grid)
    rv = multivariate_t(loc=beta_hat, shape=Sigma, df=nu)
    pos = np.stack([X, Y], axis=-1)
    Z = rv.pdf(pos)
    return X, Y, Z


def plot_joint_posterior(
    beta_hat: np.ndarray,
    Sigma: np.ndarray,
    nu: int,
) -> None:
    """
    Create a contour plot of the joint posterior distribution of C_0 and S.

    Visualizes the joint posterior PDF of the model parameters as a filled contour plot,
    and marks the posterior mean.

    Parameters
    ----------
    beta_hat : np.ndarray
        Posterior mean vector of the regression coefficients.
    Sigma : np.ndarray
        Posterior scale (covariance) matrix.
    nu : int
        Degrees of freedom for the posterior distribution.

    Returns
    --------
    None
        This function produces a plot and does not return a value.
    """
    # Evaluate joint PDF on a grid
    X, Y, Z = evaluate_joint_pdf(beta_hat, Sigma, nu)

    # Plot the contour plot
    plt.figure(figsize=(8, 6))
    contour = plt.contourf(X, Y, Z, levels=50, cmap="viridis")
    plt.colorbar(contour, label="PDF")
    plt.scatter(
        x=beta_hat[0],
        y=beta_hat[1],
        s=15,
        c="red",
        marker="x",
        label="Posterior Mean",
    )
    plt.title("Posterior Distribution of ($C_0$, $S$)", fontsize=15)
    plt.xlabel("$C_0$", fontsize=14)
    plt.ylabel("$S$", fontsize=14)
    plt.legend()
    plt.tick_params(axis="both", labelsize=13)
    plt.show()
