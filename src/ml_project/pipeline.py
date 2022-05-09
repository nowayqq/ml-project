from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_pipeline(
    max_iter: int, logreg_C: float, random_state: int
) -> Pipeline:
    pipeline_steps = [("scaler", StandardScaler()), (
        "classifier",
        LogisticRegression(
            random_state=random_state, max_iter=max_iter, C=logreg_C
        ),
    )]
    return Pipeline(steps=pipeline_steps)
