# Parameters

workspace_name = "Training_WS"
resource_group = "ML_OPS_RG"
dataset_name = "wine_quality_table"
dataset_version = 1
environment_name = "experiment_env"
conda_packages = ['pip']
pip_packages = ['azureml-defaults', 'azure.storage.blob', 'joblib',
                'sklearn', 'argparse', 'pandas', 'seaborn', 'ipykernel', 'matplotlib']
compute_target_name = 'comp1'
source_directory = "Scripts​"
script = "Train_Script.py"
depth = 4
n_estimators = 20
named_dataset = "wine_dataset"
experiment_name = 'experiment4'
