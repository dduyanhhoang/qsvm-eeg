import argparse
import yaml
import mlflow
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tempfile
import logging
import os
from pathlib import Path
from loguru import logger
from datetime import datetime
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

from qsvm_eeg.data import make_dataset
from qsvm_eeg.models.registry import get_model

# 1. Quiet the noise
logging.getLogger("mlflow").setLevel(logging.WARNING)
logging.getLogger("alembic").setLevel(logging.WARNING)


def evaluate_and_log(model, X_test, y_test, run_id, figures_dir) -> str:
    """
    Evaluates the model, logs aligned metrics, and cleans up local artifacts.
    """
    logger.info(f"Evaluating {model.name}...")

    t0 = datetime.now()
    y_pred = model.predict(X_test)
    inference_time = (datetime.now() - t0).total_seconds()

    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    r_val, _ = pearsonr(y_test, y_pred)

    n = len(y_pred)
    ci = 1.96 * np.std(y_pred - y_test) / np.sqrt(n)

    log_msg = (
        f"{model.name:<20} | "
        f"MSE: {mse:>8.4f} | "
        f"RMSE: {rmse:>8.4f} | "
        f"R2: {r2:>8.4f} | "
        f"Pearson: {r_val:>8.4f} | "
        f"CI: {ci:>8.4f}"
    )

    logger.success(log_msg)

    mlflow.log_metrics({
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "pearson": r_val,
        "ci": ci,
        "inference_time": inference_time
    })

    figures_path = Path(figures_dir)
    figures_path.mkdir(parents=True, exist_ok=True)

    fig1 = plt.figure(figsize=(10, 5))
    plt.plot(y_test, label='Actual', alpha=0.7)
    plt.plot(y_pred, label='Prediction', linestyle='--')
    plt.title(f"{model.name}: RMSE={rmse:.2f} | R2={r2:.2f}")
    plt.legend()

    plot_path1 = figures_path / f"{run_id}_{model.name}_timeseries.png"
    fig1.savefig(plot_path1, dpi=300)
    mlflow.log_artifact(str(plot_path1), artifact_path="figures")
    plt.close(fig1)

    fig2 = plt.figure(figsize=(8, 6))
    plt.scatter(y_pred, y_test, alpha=0.5, color='purple')
    plt.plot([min(y_pred), max(y_pred)], [min(y_pred), max(y_pred)], 'k--')
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title(f"Correlation: {model.name} (R={r_val:.2f})")

    plot_path2 = figures_path / f"{run_id}_{model.name}_corr.png"
    fig2.savefig(plot_path2, dpi=300)
    mlflow.log_artifact(str(plot_path2), artifact_path="figures")
    plt.close(fig2)

    try:
        os.remove(plot_path1)
        os.remove(plot_path2)
    except Exception as e:
        logger.warning(f"Could not delete temp figures: {e}")

    return log_msg


def main():
    parser = argparse.ArgumentParser(description="QSVM-EEG Experiment Runner")
    parser.add_argument("--config", default="configs/default.yaml", help="Path to config file")
    parser.add_argument("--models", nargs="+", help="Specific models to run")
    parser.add_argument("--samples", type=int, help="Override total samples limit")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    if args.models:
        for m in config['models']:
            config['models'][m]['enabled'] = (m in args.models)

    if args.samples is not None:
        config['samples'] = args.samples
        logger.info(f"Overriding config: samples = {args.samples}")

    mlflow.set_experiment(config['experiment_name'])

    try:
        X, y = make_dataset(config['patients'], config)
    except Exception as e:
        logger.error(f"Data Pipeline Failed: {e}")
        return

    X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                        test_size=0.2,
                                                        random_state=config['random_state'],
                                                        shuffle=True)

    logger.info(f"Train: {X_train.shape} | Test: {X_test.shape}")

    final_summary = []

    for model_name, model_conf in config['models'].items():
        if not model_conf.get('enabled', False):
            continue

        logger.info(f"Starting Run: {model_name}")

        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            log_file = tmp_path / f"{model_name}_{timestamp}.log"

            log_handler_id = logger.add(log_file, level="INFO")

            with mlflow.start_run(run_name=model_name) as run:

                try:
                    model = get_model(model_name, config)
                except Exception as e:
                    logger.error(f"Failed to init {model_name}: {e}")
                    continue

                run_config = config.copy()
                run_config.pop('models', None)
                run_config.pop('dirs', None)

                for key, value in model_conf.items():
                    run_config[f"model.{key}"] = value

                if hasattr(model, "backend"):
                    run_config["backend"] = model.backend
                else:
                    run_config.pop("backend", None)

                if hasattr(model, "batch_size"):
                    run_config["batch_size"] = model.batch_size
                else:
                    run_config.pop("batch_size", None)

                mlflow.log_params(run_config)
                mlflow.log_param("model_name", model_name)

                try:
                    t0 = datetime.now()
                    train_meta = model.train(X_train, y_train)
                    train_time = (datetime.now() - t0).total_seconds()

                    mlflow.log_metric("train_time", train_time)
                    mlflow.log_params(train_meta.get("best_params", {}))

                    result_msg = evaluate_and_log(
                        model, X_test, y_test,
                        run.info.run_id,
                        config['dirs']['figures']
                    )
                    final_summary.append(result_msg)

                    model_path = Path(config['dirs']['models']) / f"{run.info.run_id}_{model_name}.pkl"
                    model.save(str(model_path))
                    mlflow.log_artifact(str(model_path))

                    mlflow.log_artifact(str(log_file), artifact_path="logs")

                except Exception as e:
                    logger.exception(f"Run failed for {model_name}: {e}")
                finally:
                    logger.remove(log_handler_id)

    if final_summary:
        logger.info("FINAL SUMMARY REPORT")
        for line in final_summary:
            logger.success(line)


if __name__ == "__main__":
    main()
