# %%
import mlflow
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
client = mlflow.tracking.MlflowClient()
# %%
out = client.list_run_infos(
    experiment_id=experiment_number, max_results=2)
# %%
out[0]
# %%
run_id = "c0db744fe1e841b29db92ac1605ebb3e"
run_full_info = client.get_run(run_id)
# %%
run_full_info.data.params
# %%
client.get_metric_history(run_id, "score")
# %%

# %%
client.list_artifacts(run_id)
# %%
runs = client.search_runs("1", order_by=["metrics.score DESC"])
# %%
runs = client.search_runs("1")
# %%
type(run_full_info)
# %%
len(runs)
# %%
