import os
import pandas as pd
import random
from ema_workbench import (Model, Policy, ema_logging, MultiprocessingEvaluator)
from problem_formulation import get_model_for_problem_formulation
import numpy as np

"""
Overview

This Python script is designed to evaluate policies for a dike model using the EMA Workbench. 
It performs the following key tasks:

1. Policy Setup: Loads Pareto optimal policies from a CSV file and initializes them for evaluation.
2. Experimentation: Executes experiments with specified policies across scenarios.
3. Outcome Processing: Structures outcomes into a outcomes and experiments DataFrame format which is saved in
data/output_data/Step3.

### Execution
Execute the script directly to initialize the model, set up logging and random seeds, and conduct policy evaluations.

## Usage
To run the script, ensure all necessary dependencies are installed and simply execute the script. 
For more information, see: README.md.
"""

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

def main():
    """
    Main function to evaluate Pareto optimal policies for a dike model using the EMA Workbench.

    This function performs the following steps:
    1. Sets up the problem formulation ID and initializes logging and random seed.
    2. Initializes the dike model based on the specified problem formulation.
    3. Loads Pareto optimal policies from a CSV file containing cleaned solutions.
    4. Creates policy objects from the loaded data, ensuring unique naming based on policy type.
    5. Executes experiments using multiprocessing for efficient evaluation across scenarios.
    6. Processes outcomes to summarize and structure the results into a DataFrame format.
    7. Saves the combined experiment details and processed outcomes into the 'Step3' directory.
    8. Prints confirmation messages upon successful completion of data saving.

    Note:
    - The function assumes that the required modules (os, pandas, random, ema_workbench) and functions
      (get_model_for_problem_formulation, process_outcomes) are correctly imported and available.
    - Ensure that the 'combined_nondominated_solutions_cleaned.csv' file is correctly formatted and accessible
      in the 'Step2' directory under 'data/output_data'.

    Returns:
    -------
    None
    """

    problem_formulation_id = 8

    # Set up logging and seed
    random.seed(1361)
    ema_logging.log_to_stderr(ema_logging.INFO)

    # Initialize the dike model from the problem formulation
    dike_model, planning_steps = get_model_for_problem_formulation(problem_formulation_id)

    # Construct the path to the merged_archives.csv file
    current_dir = os.getcwd()
    data_dir = os.path.join(current_dir, 'data', 'output_data')
    file_path = os.path.join(data_dir, 'Step2', 'combined_nondominated_solutions_cleaned.csv')

    # Load the Pareto optimal policies
    policies_df = pd.read_csv(file_path)

    # Initialize policy counters
    policy_counts = {'90th Percentile': 1, 'Mean': 1}

    # Create the policies with type-specific counters
    policies = []
    for i, row in policies_df.iterrows():
        policy_type = row['Type']
        policy_index = policy_counts[policy_type]
        policy_name = f"{policy_type} Policy {str(policy_index).zfill(2)}"
        policies.append(Policy(policy_name, **row.drop('Type').to_dict()))
        policy_counts[policy_type] += 1  # Increment the count for this type

    # Ensure the output directory exists
    os.makedirs(data_dir, exist_ok=True)

    # Perform experiments for each policy
    experiments_list = []
    outcomes_list = []
    with MultiprocessingEvaluator(dike_model, n_processes=-2) as evaluator:
        experiments, outcomes = evaluator.perform_experiments(
            scenarios=1000, policies=policies
        )
        experiments['policy'] = experiments['policy'].astype(str)
        experiments_list.append(experiments)
        processed_outcomes = process_outcomes(outcomes)
        processed_outcomes['policy'] = experiments['policy']
        outcomes_list.append(processed_outcomes)

    # Combine results and save
    combined_experiments = pd.concat(experiments_list)
    combined_outcomes = pd.concat(outcomes_list)

    save_dir = os.path.join(data_dir, 'Step3')
    os.makedirs(save_dir, exist_ok=True)

    combined_experiments.to_csv(os.path.join(save_dir, 'policy_evaluation_experiments.csv'), index=False)
    combined_outcomes.to_csv(os.path.join(save_dir, 'policy_evaluation_outcomes.csv'), index=False)

    print(f"Policy evaluation experiments saved to {os.path.join(save_dir, 'policy_evaluation_experiments.csv')}")
    print(f"Policy evaluation outcomes saved to {os.path.join(save_dir, 'policy_evaluation_outcomes.csv')}")

if __name__ == "__main__":
    main()
