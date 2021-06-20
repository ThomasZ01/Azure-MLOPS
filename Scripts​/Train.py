from azureml.core import Experiment, ScriptRunConfig, Environment, Dataset
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Workspace
import os
import sys
from params import *

sys.path.append('/.../AZURE-MLOPS/Configs')


#SUBSCRIPTION_ID = os.getenv('SUBSCRIPTION_ID')

SUBSCRIPTION_ID = "d07d6a2c-86a6-4297-9733-aeb8be2a98bc"

ws = Workspace.get(name=workspace_name,
                   subscription_id=SUBSCRIPTION_ID,
                   resource_group=resource_group)

# Register a Dataset
#blob_store = Datastore.get(ws, datastore_name='datastore1')
#csv_paths = (blob_store, 'wine_quality.csv')
#tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
#tab_ds = tab_ds.register(workspace=ws, name='wine_quality_table')

# Get the dataset
wine_ds = Dataset.get_by_name(
    workspace=ws, name=dataset_name, version=dataset_version)

# Create or get a Python environment for the experiment
env = Environment("experiment_env")

# Ensure the required packages are installed
packages = CondaDependencies.create(conda_packages=conda_packages,
                                    pip_packages=pip_packages)
env.python.conda_dependencies = packages
env.register(workspace=ws)
env = Environment.get(workspace=ws, name=environment_name)

# create or retrieve a compute target
cluster = ws.compute_targets[compute_target_name]

# Create a script config
script_config = ScriptRunConfig(source_directory=source_directory,
                                script=script,
                                arguments=['--depth', depth,
                                           '--n_estimators', n_estimators,
                                           '--ds', wine_ds.as_named_input(named_dataset)],
                                compute_target=cluster,
                                environment=env)

# submit the experiment
experiment = Experiment(workspace=ws, name=experiment_name)
run = experiment.submit(config=script_config)

run.wait_for_completion(show_output=False)
