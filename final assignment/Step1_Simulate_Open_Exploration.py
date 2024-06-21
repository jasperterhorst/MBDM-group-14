import os
import pandas as pd
import random
from ema_workbench import (Model, Policy, ema_logging, MultiprocessingEvaluator, Samplers, SequentialEvaluator)
from problem_formulation import get_model_for_problem_formulation
import numpy as np

"""
Overview

This Python script is designed to set up and run analyses for a dike model using the EMA Workbench. 
The script performs the following key tasks:

1. Random Policy Analysis: Runs experiments with randomly generated policies.
2. Sensitivity Analysis: Performs Sobol Sensitivity Analysis.
3. No Policy Analysis: Evaluates a 'do nothing' policy under a large number of scenarios.
4. Policy Preparation: Creates a dictionary for a 'do nothing' policy.
5. Outcome Processing: Converts outcomes into a structured DataFrame format.


### Execution
The script is executed directly. It initializes the model, sets up logging and random seeds, 
and performs the specified analyses. Some analyses are commented out and can be activated by uncommenting the relevant sections.

## Usage
To run the script, ensure all necessary dependencies are installed and simply execute the script. 
For more information, see: README.md.
"""

def get_do_nothing_dict(model):
    """
    Prepare a dictionary for a 'do nothing' policy.

    Parameters
    ----------
    model : Model
        The model for which the policy dictionary is being prepared.

    Returns
    -------
    dict
        A dictionary with all policy levers set to zero.
    """
    return {lever.name: 0 for lever in model.levers}


def process_outcomes(outcomes):
    """
    Process outcomes to handle multidimensional arrays and structure the DataFrame correctly.

    Parameters
    ----------
    outcomes : dict
        Dictionary of outcomes from the model.

    Returns
    -------
    pd.DataFrame
        A DataFrame with processed outcomes.
    """
    processed = {}
    for key, value in outcomes.items():
        if isinstance(value, np.ndarray):
            processed[key] = value.sum(axis=1) if value.ndim > 1 else value
        else:
            processed[key] = value

    return pd.DataFrame(processed)


if __name__ == "__main__":

    problem_formulation_id = 8

    # Set up logging and seed
    random.seed(1361)
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Initialize the dike model from the problem formulation
    dike_model, planning_steps = get_model_for_problem_formulation(problem_formulation_id)

    # Ensure the output directory exists
    output_dir = os.path.join('data', 'output_data')
    os.makedirs(output_dir, exist_ok=True)

    # Random Policy Analysis
    with MultiprocessingEvaluator(dike_model, n_processes=-1) as evaluator:
        random_experiments, random_outcomes = evaluator.perform_experiments(
            scenarios=2000, policies=200
        )
    random_experiments.to_csv(os.path.join(output_dir, 'random_experiments.csv'))
    random_outcomes_df = process_outcomes(random_outcomes)
    random_outcomes_df['policy'] = random_experiments['policy']
    random_outcomes_df.to_csv(os.path.join(output_dir, 'random_outcomes.csv'))

    # Sobol Sensitivity Analysis
    base_case_policy = [Policy("Do Nothing Policy", **get_do_nothing_dict(dike_model))]
    with MultiprocessingEvaluator(dike_model, n_processes=-1) as evaluator:
        sobol_experiments, sobol_outcomes = evaluator.perform_experiments(
            scenarios=16384, policies=base_case_policy, uncertainty_sampling=Samplers.SOBOL
        )
    sobol_experiments.to_csv(os.path.join(output_dir, 'sobol_experiments.csv'))
    sobol_outcomes_df = process_outcomes(sobol_outcomes)
    sobol_outcomes_df.to_csv(os.path.join(output_dir, 'sobol_outcomes.csv'))

    # No Policy Analysis
    with MultiprocessingEvaluator(dike_model, n_processes=-1) as evaluator:
        no_policy_experiments, no_policy_outcomes = evaluator.perform_experiments(
            scenarios=500000, policies=base_case_policy
        )
    no_policy_experiments.to_csv(os.path.join(output_dir, 'no_policy_experiments.csv'))
    no_policy_outcomes_df = process_outcomes(no_policy_outcomes)
    no_policy_outcomes_df['policy'] = no_policy_experiments['policy']
    no_policy_outcomes_df.to_csv(os.path.join(output_dir, 'no_policy_outcomes.csv'))

