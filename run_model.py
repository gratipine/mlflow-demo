# %%
# Data from https://www.kaggle.com/gpreda/reddit-vaccine-myths/download
# %% Libraries
import mlflow
import pandas as pd
import src.feat_engineering as feat
import os
import shutil
from sklearn.ensemble import RandomForestRegressor

# %% Data in
data_in = pd.read_csv("data/reddit_vm.csv")

# %% Feature engineering
data_in_prep = feat.add_body_flag(data_in)
data_in_prep["timestamp"] = pd.to_datetime(data_in_prep["timestamp"])
data_in_prep = feat.add_date_difference_from_start(data_in_prep, date_column="timestamp")
data_in_prep["diff_from_start"] = data_in_prep["diff_from_start"].dt.days

# %% Create mlflow experiment
experiment_name = "Vaccination reddit - GCP demo"
try:
    experiment_number = mlflow.create_experiment(
        experiment_name)
except Exception as e:
    experiment_number = (
        mlflow.get_experiment_by_name(experiment_name)
        .experiment_id)
# %%
# We are interested in feature combinations
feature_combinations = [
    ["comms_num"], 
    ["comms_num", "body_present"],
    ["comms_num", "diff_from_start"],
    ["comms_num", "body_present", "diff_from_start"],
    ["body_present"],
    ["body_present", "diff_from_start"],
    ["diff_from_start"]] 

# %% run mlflow experiment
for feature_comb in feature_combinations:
    
    # create a temporary folder to hold artifacts
    artifact_location = ".mlflow_temp"
    if os.path.exists(artifact_location):
        shutil.rmtree(artifact_location)
    os.makedirs(artifact_location)

    with mlflow.start_run(
            run_name=f"runs",
            experiment_id=experiment_number):
        
        # log the model
        mlflow.sklearn.autolog()

        seed = 123
        
        model = RandomForestRegressor(random_state=seed)
        X = data_in_prep[feature_comb]
        Y = data_in_prep["score"]
        model.fit(X, Y)

        # log param
        mlflow.log_params({
            "seed": 123,
            "features_used": feature_comb
        })

        # log metrics
        mlflow.log_metrics({
            "score": model.score(X, Y)
        })
        


# %% 