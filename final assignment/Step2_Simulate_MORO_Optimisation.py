import os
import random
import numpy as np
import pandas as pd
from ema_workbench import ema_logging, MultiprocessingEvaluator, ScalarOutcome
from ema_workbench.em_framework import samplers
from ema_workbench.em_framework.optimization import HyperVolume, EpsilonProgress
from problem_formulation import get_model_for_problem_formulation

"""
Overview

This Python script is designed to set up and run optimizations for a dike model using the EMA Workbench. 
The script performs the following key tasks:

1. Statistic Calculation: Computes specified statistics (90th percentile or mean) for given data.
2. Outcome Processing: Converts outcomes into a structured DataFrame format.
3. Main Optimization Routine: Runs the optimization process multiple times with different random seeds, 
   minimizing damage, investment, and deaths for either the 90th percentile or mean. 
   - Collect Convergence Metrics: Besides that the main optimization part also collects HyperVolume and Epsilon 
   progress convergence.

### Execution
The script is executed with a specified statistic_type (e.g., `'90'` or `'mean'`). This statistic type determines 
how the optimization outcomes are calculated.

## Usage
To run the script, specify the desired statistic type in the `main` function call at the end of the script, also ensure 
all necessary dependencies are installed, then click run.
For more information, see: README.md.
"""

def calculate_statistic(data, stat_type):
    """
    Calculate the specified statistic of the given data.

    Parameters:
        data (np.ndarray): The data array to compute the statistic for.
        stat_type (str): The type of statistic to compute ('90', 'mean', 'median').

    Returns:
        float: The computed statistic value.
    """
    if stat_type == '90':
        return np.percentile(data, 90)
    elif stat_type == 'mean':
        return np.mean(data)
    else:
        raise ValueError("Invalid stat_type. Expected one of: '90', 'mean'.")


def process_outcomes(outcomes):
    """
    Process the outcomes dictionary into a DataFrame.

    Parameters:
        outcomes (dict): Dictionary of outcomes.

    Returns:
        pd.DataFrame: Processed outcomes as a DataFrame.
    """
    processed = {}
    for key, value in outcomes.items():
        if isinstance(value, np.ndarray):
            processed[key] = value.sum(axis=1) if value.ndim > 1 else value
        else:
            processed[key] = value
    return pd.DataFrame(processed)


def main(stat_type):
    """
    Main function to set up the dike model, run optimizations, and save results.

    The optimization process is executed 5 times with different random seeds.
    Each run minimizes three outcome variables:
    - Damage
    - Investment
    - Deaths

    The specific statistic (90th percentile or mean) used for these variables
    is determined by the stat_type parameter.

    The optimization uses 15000 function evaluations (nfe) and 100 scenarios.

    Also this function saves HyperVolume and Epsilon progress convergence metrics, to be used later on.
    """
    problem_formulation_id = 7
    output_dir = os.path.join('data', 'output_data', 'Step2', stat_type)
    os.makedirs(output_dir, exist_ok=True)


    for i in [1362, 1363, 1364, 1365]:
        random.seed(i)
        ema_logging.log_to_stderr(ema_logging.INFO)

        # Initialize the dike model from the problem formulation
        dike_model, planning_steps = get_model_for_problem_formulation(problem_formulation_id)
        n_scenarios = 100

        # Sample scenarios
        scenarios = samplers.sample_uncertainties(dike_model, n_scenarios)

        # Define robustness functions for optimization
        robustness_functions = [
            ScalarOutcome(f'Damage {stat_type.capitalize()} Statistic', kind=ScalarOutcome.MINIMIZE,
                          variable_name='Combined Expected Annual Damage',
                          function=lambda data: calculate_statistic(data, stat_type)),
            ScalarOutcome(f'Investment {stat_type.capitalize()} Statistic', kind=ScalarOutcome.MINIMIZE,
                          variable_name='Combined Dike Investment Costs',
                          function=lambda data: calculate_statistic(data, stat_type)),
            ScalarOutcome(f'Deaths {stat_type.capitalize()} Statistic', kind=ScalarOutcome.MINIMIZE,
                          variable_name='Combined Expected Number of Deaths',
                          function=lambda data: calculate_statistic(data, stat_type))
        ]

        convergence_metrics = [HyperVolume(minimum=[0, 0, 0], maximum=[870000000, 240000000, 0.9]),
                               EpsilonProgress()]
        nfe = 15000

        # Run optimization
        with MultiprocessingEvaluator(dike_model, n_processes=-3) as evaluator:
            archive, convergence = evaluator.robust_optimize(
                robustness_functions, scenarios, nfe=nfe,
                convergence=convergence_metrics, epsilons=[0.05, 0.05, 0.05]
            )

        # Save optimization archive
        archive_path = os.path.join(output_dir,f'optimization_archive_{stat_type}_seed_{i}.csv')
        archive.to_csv(archive_path)

        # Save convergence metrics
        convergence_df = pd.DataFrame({
            'nfe': convergence.nfe,
            'epsilon_progress': convergence.epsilon_progress,
            'hypervolume': convergence.hypervolume
        })

        convergence_path = os.path.join(output_dir, f'convergence_metrics_{stat_type}_seed_{i}.csv')
        convergence_df.to_csv(convergence_path)


if __name__ == "__main__":
    # Example usage: Change '90' to 'mean' as needed
    main('90')
