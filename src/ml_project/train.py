from pathlib import Path

import click

from .data import get_data


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
def train(
        dataset_path: Path,
        random_state: int,
        test_split_ratio: float,
        gen_rep: bool,
) -> None:
    features_train, features_val, target_train, target_val = get_data(
        dataset_path,
        random_state,
        test_split_ratio,
        gen_rep
    )
