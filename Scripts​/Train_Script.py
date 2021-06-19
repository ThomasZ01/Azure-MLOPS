# Get the experiment run context
try:
    from azureml.core import Run
    import pandas as pd
    import os
    from azure.storage.blob import BlobServiceClient, BlobClient, ContainerClient, __version__
    from io import StringIO
    import joblib
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestRegressor
    import argparse
    import seaborn as sns
    import matplotlib.pyplot as plt

    run = Run.get_context()

    # Set parameter
    parser = argparse.ArgumentParser()
    parser.add_argument('--depth', type=int, dest='depth', default=1)
    parser.add_argument('--ds', type=str, dest='ds_id')
    parser.add_argument('--n_estimators', type=int, dest='n_estimators')
    args = parser.parse_args()
    depth = args.depth
    n_estimators = args.n_estimators

    seed = 42

    dataset = run.input_datasets['wine_dataset']
    df = dataset.to_pandas_dataframe()

    # Split into train and test sections
    y = df.pop("quality")
    X_train, X_test, y_train, y_test = train_test_split(
        df, y, test_size=0.2, random_state=seed)

    ########## MODELLING ##############################################################################

    # Fit a model on the train section
    regr = RandomForestRegressor(
        max_depth=depth, random_state=seed, n_estimators=n_estimators)
    regr.fit(X_train, y_train)

    # Report training set score
    train_score = regr.score(X_train, y_train) * 100
    # Report test set score
    test_score = regr.score(X_test, y_test) * 100

    # Save the trained model
    os.makedirs('outputs', exist_ok=True)
    joblib.dump(value=regr, filename='outputs/model.pkl')

    ##########################################
    ##### PLOT FEATURE IMPORTANCE ############
    ##########################################
    # Calculate feature importance in random forest
    importances = regr.feature_importances_
    labels = df.columns
    feature_df = pd.DataFrame(list(zip(labels, importances)), columns=[
        "feature", "importance"])
    feature_df = feature_df.sort_values(by='importance', ascending=False,)

    for arg in vars(args):
        run.log(str(arg), getattr(args, arg))


except Exception as e:
    run.log('error', e.message)


row_count = (len(df))
run.log('observations', row_count)
run.log('train_score', train_score)
run.log('test_score', test_score)


run.complete()
