
from azureml.core import Experiment, ScriptRunConfig, Environment, Dataset
from azureml.core.conda_dependencies import CondaDependencies
from azureml.core import Workspace
import os

SUBSCRIPTION_ID = os.getenv('SUBSCRIPTION_ID')

ws = Workspace.get(name="Training_WS",
                   subscription_id=SUBSCRIPTION_ID,
                   resource_group='ML_OPS_RG')

# Register a Dataset
#blob_store = Datastore.get(ws, datastore_name='datastore1')
#csv_paths = (blob_store, 'wine_quality.csv')
#tab_ds = Dataset.Tabular.from_delimited_files(path=csv_paths)
#tab_ds = tab_ds.register(workspace=ws, name='wine_quality_table')

# Get the dataset
wine_ds = Dataset.get_by_name(
    workspace=ws, name='wine_quality_table', version=1)

# Create or get a Python environment for the experiment
env = Environment("experiment_env")

# Ensure the required packages are installed
packages = CondaDependencies.create(conda_packages=['pip'],
                                    pip_packages=['azureml-defaults', 'azure.storage.blob', 'joblib', 'sklearn', 'argparse', 'pandas', 'seaborn', 'ipykernel', 'matplotlib'])
env.python.conda_dependencies = packages
env.register(workspace=ws)
env = Environment.get(workspace=ws, name='experiment_env')

# create or retrieve a compute target
cluster = ws.compute_targets['comp1']

# Create a script config
script_config = ScriptRunConfig(source_directory='.',
                                script='train_script.py',
                                arguments=['--depth', 4,
                                            '--n_estimators', 100,
                                            '--ds', wine_ds.as_named_input('wine_dataset')],
                                compute_target=cluster,
                                environment=env)

# submit the experiment
experiment = Experiment(workspace=ws, name='experiment4')
run = experiment.submit(config=script_config)

run.wait_for_completion(show_output=False)
