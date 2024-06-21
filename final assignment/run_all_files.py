import subprocess
import os

def run_notebook(notebook_path):
    command = f"jupyter nbconvert --to notebook --execute {notebook_path} --output {notebook_path}"
    subprocess.run(command, shell=True, check=True)

def run_python_script(script_path):
    command = f"python {script_path}"
    subprocess.run(command, shell=True, check=True)

# Define the paths to your files
files = [
    "Step1_Notebook_Open_Exploration.ipynb",
    "Step1_Simulate_Open_Exploration.py",
    "Step1b_Notebook_Policy_Impact_Exploration.ipynb",
    "Step2_Notebook_MORO_Optimisation_Outcomes.ipynb",
    "Step2_Simulate_MORO_Optimisation.py",
    "Step3_Notebook_SNR_and_Maximum_Regret.ipynb",
    "Step3_Simulate_MORO_Policies_Under_Uncertainty.py",
    "Step4_Notebook_MORO_Scenario_Discovery.ipynb"
]

# Run the files consecutively
for file in files:
    if file.endswith(".ipynb"):
        run_notebook(file)
    elif file.endswith(".py"):
        run_python_script(file)
    else:
        print(f"Unknown file type: {file}")

print("All files have been run successfully.")
