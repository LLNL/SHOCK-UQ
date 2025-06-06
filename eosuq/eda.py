"""
Raw Data Visualization for Shock Hugoniot Analysis
"""

import matplotlib.pyplot as plt
import pandas as pd


def plot_raw_data(df: pd.DataFrame) -> None:
    """
    Plot shock wave velocity versus particle velocity from a DataFrame.

    Parameters
    ----------
    df : pd.DataFrame)
        A DataFrame containing at least the columns 'Up_km_s' (particle
        velocity in km/s) and 'Us_km_s' (shock velocity in km/s).

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """
    plt.figure(figsize=(4, 6))
    plt.scatter(
        df["Up_km_s"],
        df["Us_km_s"],
        color="blue",
        alpha=0.8,
        facecolor="none",
        edgecolors="blue",
    )
    plt.title("Shock Wave versus Particle Velocity", fontsize=15)
    plt.xlabel("Particle Velocity - Up [km/s]", fontsize=14)
    plt.ylabel("Shock Velocity - Us [km/s]", fontsize=14)
    plt.tick_params(axis="both", labelsize=13)
    plt.grid()
    plt.show()


def plot_experiment_type_distribution(df: pd.DataFrame) -> None:
    """
    Create a horizontal bar chart showing the distribution of experiment types.

    Parameters
    ----------
    df : pd.DataFrame
        A DataFrame containing an 'exp' column with experiment type codes.

    Returns
    -------
    None
        This function produces a plot and does not return a value.
    """
    label_map = {
        "im1": "Impedance\nMatching",
        "sf2": "Shock and Free\nSurface Velocity",
        "ssp": "Sound Speed",
        "sp1": "Shock and\nParticle Velocity",
    }
    df = df.copy()
    df["exp_full"] = df["exp"].map(label_map)
    exp_counts = df["exp_full"].value_counts().iloc[::-1]
    ax = exp_counts.plot(kind="barh", color="skyblue", edgecolor="black")
    plt.xlabel("Number of Experiments")
    plt.ylabel("")
    plt.title("Distribution of Experiment Type")
    plt.grid(axis="x", linestyle="--", alpha=0.7, zorder=0)
    ax.set_axisbelow(True)
    plt.tight_layout()
    plt.show()
