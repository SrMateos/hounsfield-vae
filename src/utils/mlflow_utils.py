import ast
from typing import Any, Dict, Optional

import mlflow


def setup_mlflow(
    config: Dict[str, Any],
    experiment_name: str = "medical_image_translation",
    run_id: Optional[str] = None,
):
    """
    Sets up MLFlow tracking with the given configuration.

    Args:
        config: Configuration dictionary
        experiment_name: Name of the MLFlow experiment
        run_id: Existing run to continue (used when resuming from a checkpoint)

    Returns:
        active_run: MLFlow active run object
    """
    # Set tracking URI if specified in config
    if "mlflow" in config and "tracking_uri" in config["mlflow"]:
        mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    else:
        mlflow.set_tracking_uri("http://localhost:5000")

    # Set the experiment
    mlflow.set_experiment(experiment_name)

    # Start (or continue) the run
    if run_id is not None:
        active_run = mlflow.start_run(run_id=run_id)
    else:
        active_run = mlflow.start_run(run_name=config.get("run_name", None))


    # Log tags if specified
    if "mlflow" in config and "tags" in config["mlflow"]:
        mlflow.set_tags(config["mlflow"]["tags"])

    return active_run

def log_config(config: Dict[str, Any], prefix: str = ""):
    """
    Recursively logs configuration parameters to MLFlow.

    Args:
        config: Configuration dictionary
        prefix: Prefix for parameter names in nested dictionaries
    """
    for key, value in config.items():
        param_name = f"{prefix}{key}" if prefix else key

        if isinstance(value, dict):
            log_config(value, f"{param_name}.")
        elif isinstance(value, (int, float, str, bool, list)):
            mlflow.log_param(param_name, value)


def load_run_config(run_id: str) -> Dict[str, Any]:
    """
    Rebuild the data/inference configuration of a training run from its MLflow params,
    so that analysis scripts reproduce the same test split.

    Params are logged by `log_config` with dotted names (``data.train_split``). Older runs
    may lack some of them (lists were not logged), in which case the defaults of
    config/defaults.yaml are used.
    """
    params = mlflow.tracking.MlflowClient().get_run(run_id).data.params

    def get(key, default, cast):
        return cast(params[key]) if key in params else default

    return {
        "data": {
            "patch_size": get("data.patch_size", [64, 64, 64], ast.literal_eval),
            "train_split": get("data.train_split", 0.7, float),
            "val_split": get("data.val_split", 0.1, float),
            "test_split": get("data.test_split", 0.2, float),
            "tasks": get("data.tasks", [1, 2], ast.literal_eval),
            "task1": True,
        },
        "inference": {
            "sw_batch_size": get("inference.sw_batch_size", 12, int),
            "overlap": get("inference.overlap", 0.75, float),
            "mode": get("inference.mode", "gaussian", str),
        },
        "debug": get("debug", "False", str).lower() == "true",
    }
