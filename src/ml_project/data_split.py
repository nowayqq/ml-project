from pathlib import Path
from typing import Tuple
from pandas_profiling import ProfileReport
from sklearn.model_selection import train_test_split

import click
import pandas as pd


def generate_report(csv_path: Path):
    return ProfileReport(pd.read_csv(
        csv_path)).to_file("../../EDA_report.html")


def get_data(
        csv_path: Path, random_state: int, test_split_ratio: float,
        gen_rep=False
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    data = pd.read_csv(csv_path)
    click.echo(f"Dataset shape: {data.shape}.")
    features = data.drop("Cover_Type", axis=1)
    target = data["Cover_Type"]
    features_train, features_val, target_train, target_val = train_test_split(
        features, target, test_size=test_split_ratio, random_state=random_state
    )
    if gen_rep:
        generate_report(data)
    return features_train, features_val, target_train, target_val
