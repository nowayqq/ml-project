import click

from pathlib import Path
from sklearn.metrics import accuracy_score
from joblib import dump

import data_split as ds
import pipeline as pp


@click.command()
@click.option(
    '-d',
    '--dataset-path',
    default='../../data/train.csv',
    type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.option(
    '--random-state',
    default=42,
    type=int,
    show_default=True
)
@click.option(
    '--test-split-ratio',
    default=0.2,
    type=click.FloatRange(0, 1, min_open=True, max_open=True),
    show_default=True
)
@click.option(
    '--gen_rep',
    default=False,
    type=bool,
    show_default=True
)
@click.option(
    '--max-iter',
    default=1000,
    type=int,
    show_default=True
)
@click.option(
    '--logreg-c',
    default=1.0,
    type=float,
    show_default=True
)
@click.option(
    '--save-model-path',
    default="../../data/model.joblib",
    type=click.Path(dir_okay=False, writable=True, path_type=Path),
    show_default=True
)
def train(
        dataset_path: Path,
        save_model_path: Path,
        random_state: int,
        test_split_ratio: float,
        max_iter: int,
        logreg_c: float,
        gen_rep: bool,
) -> None:
    features_train, features_val, target_train, target_val = ds.get_data(
        dataset_path,
        random_state,
        test_split_ratio,
        gen_rep
    )
    pipeline = pp.create_pipeline(max_iter, logreg_c, random_state)
    pipeline.fit(features_train, target_train)
    accuracy = accuracy_score(target_val, pipeline.predict(features_val))
    click.echo(f'Accuracy: {accuracy}.')
    dump(pipeline, save_model_path)
    click.echo(f"Model is saved to {save_model_path}.")


train()
