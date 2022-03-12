#!/usr/bin/env python
"""
This script to test the production model pipeline
"""
import argparse
import logging
import wandb
import mlflow
import pandas as pd
from sklearn.metrics import mean_absolute_error


logging.basicConfig(level=logging.INFO, format="%(asctime)-15s %(message)s")
logger = logging.getLogger()


def go(args):
    """
    Test the regression model with provided test dataset.
    """
    run = wandb.init(job_type="test_model")
    run.config.update(args)

    logger.info("Downloading artifacts")

    model_path = run.use_artifact(args.mlflow_model).download()

    test_dataset_path = run.use_artifact(args.test_dataset).file()

    x_test = pd.read_csv(test_dataset_path)
    y_test = x_test.pop("price")

    logger.info("Loading model and performing inference on test set")
    sk_pipe = mlflow.sklearn.load_model(model_path)
    y_pred = sk_pipe.predict(x_test)

    logger.info("Scoring")
    r2 = sk_pipe.score(x_test, y_test)

    mae = mean_absolute_error(y_test, y_pred)

    logger.info("Score: {}".format(r2))
    logger.info("MAE: {}".format(mae))

    # Log MAE and r2
    run.summary['r2'] = r2
    run.summary['mae'] = mae


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test the mlflow model with the test dataset")

    parser.add_argument(
        "--mlflow_model",
        type=str,
        help="Input MLFlow model",
        required=True
    )

    parser.add_argument(
        "--test_dataset",
        type=str,
        help="Test dataset",
        required=True
    )

    args = parser.parse_args()

    go(args)
