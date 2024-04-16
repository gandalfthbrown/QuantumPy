

import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import norm, t, laplace, cauchy
import numpy as np
from quantum_asset import QuantumFinancialAsset
from distributions import find_matching_distributions
import random



def visualize_asset_dynamics(asset_data, best_objective_function, distributions,n):
    """
    Visualize the dynamics of the financial assets using the optimized objective function
    and the matching probability distributions.

    Args:
        asset_data (list): A list of tuples containing (initial_price, volatility, time, expected_price).
        best_objective_function (callable): The optimized objective function from Monte Carlo simulations.
        distributions (list): A list of probability distributions matching the asset data.
    """
    fig, axs = plt.subplots(2, 2, figsize=(12, 8))
    axs = axs.ravel()

    # Plot the probability density functions of the matching distributions
    for i, dist in enumerate(distributions):
        x = np.linspace(dist.ppf(0.01), dist.ppf(0.99), 100)
        axs[0].plot(x, dist.pdf(x), label=dist.__name__)
    axs[0].set_title("Probability Density Functions of Matching Distributions")
    axs[0].legend()

    # Plot the objective function weights
    objective_function_params = [random.Random() for _ in range(len(distributions))]
    axs[1].bar(range(len(objective_function_params)), objective_function_params)
    axs[1].set_title("Objective Function Weights")
    axs[1].set_xticks(range(len(objective_function_params)))
    axs[1].set_xticklabels([f"Weight {i+1}" for i in range(len(objective_function_params))], rotation=45)

    # Plot the predicted vs. expected prices
    predicted_prices = []
    expected_prices = []
    for initial_price, volatility, time, expected_price in asset_data:
        asset = QuantumFinancialAsset(initial_price, volatility, n)