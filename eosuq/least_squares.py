"""
Least Squares Estimation and Visualization for Linear Hugoniot Model
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import t


def compute_beta_hat(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """
    Computes the least squares estimate (beta_hat) for a linear regression model
    of shock wave velocity (Us_km_s) as a function of particle velocity (Up_km_s).

    The model is: Us_km_s = beta_0 + beta_1 * Up_km_s + error

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'Up_km_s' (predictor) and 'Us_km_s' (response).

    Returns
    -------
    beta_hat : np.ndarray
        Estimated regression coefficients (intercept and slope) as a 1D array of shape (2,).
    X : np.ndarray
        Design matrix used in the regression, with a column of ones for the intercept
        and a column for 'Up_km_s'.
    """
    # Extract the predictor variable (Up_km_s) and add an intercept column
    X = df["Up_km_s"].values
    X = np.column_stack((np.ones(len(X)), X))

    # Extract the response variable
    y = df["Us_km_s"].values

    # Compute posterior mean using the normal equation
    XTX = X.T @ X
    XTy = X.T @ y
    beta_hat = np.linalg.solve(XTX, XTy)

    return beta_hat, X


def plot_least_squares(df: pd.DataFrame, alpha: float = 0.05) -> None:
    """
    Fits and visualizes a linear regression of shock wave velocity (Us_km_s) on
    particle velocity (Up_km_s), including least squares fit, confidence intervals,
    and prediction intervals.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing columns 'Up_km_s' (predictor) and 'Us_km_s' (response).
    alpha : float, optional
        Significance level for the confidence and prediction intervals (default is 0.05,
        corresponding to a 95% interval).

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """

    # Get predictions
    beta_hat, X = compute_beta_hat(df)
    prediction_mean = X @ beta_hat
    y = df["Us_km_s"].values

    # Residuals and standard error
    residuals = y - prediction_mean
    n, p = X.shape  # Number of observations and predictors
    s_sq = np.sum(residuals**2) / (n - p)  # Variance of residuals
    s = np.sqrt(s_sq)  # Standard deviation of residuals

    # Compute standard errors for predictions
    XTX_inv = np.linalg.inv(X.T @ X)  # Inverse of X^T X
    standard_errors = s * np.sqrt(np.sum((X @ XTX_inv) * X, axis=1))

    # Compute confidence intervals
    t_value = t.ppf(1 - alpha / 2, df=n - p)  # Critical t-value
    mean_ci_lower = prediction_mean - t_value * standard_errors
    mean_ci_upper = prediction_mean + t_value * standard_errors

    # Compute prediction intervals
    prediction_ci_lower = prediction_mean - t_value * np.sqrt(standard_errors**2 + s_sq)
    prediction_ci_upper = prediction_mean + t_value * np.sqrt(standard_errors**2 + s_sq)

    # Plotting
    plt.figure(figsize=(4, 6))
    plt.scatter(
        df["Up_km_s"],
        df["Us_km_s"],
        color="blue",
        alpha=0.8,
        facecolor="none",
        edgecolors="blue",
    )

    # Plot prediction interval
    plt.fill_between(
        df["Up_km_s"],
        prediction_ci_lower,
        prediction_ci_upper,
        color="skyblue",
        alpha=0.3,
        label=f"{int((1 - alpha) * 100)}% Prediction Interval",
        zorder=1,
    )

    # Plot confidence interval
    plt.fill_between(
        df["Up_km_s"],
        mean_ci_lower,
        mean_ci_upper,
        color="orange",
        alpha=0.7,
        label=f"{int((1 - alpha) * 100)}% Confidence Interval",
        zorder=2,
    )

    plt.plot(
        df["Up_km_s"],
        prediction_mean,
        color="red",
        label="Least Squares Fit",
        linewidth=1,
    )

    plt.title("Linear Regression of $U_s$ on $U_p$", fontsize=15)
    plt.xlabel("Particle Velocity - $U_p$ [km/s]", fontsize=14)
    plt.ylabel("Shock Wave Velocity - $U_s$ [km/s]", fontsize=14)
    plt.tick_params(axis="both", labelsize=13)
    plt.legend()
    plt.grid()
    plt.show()
