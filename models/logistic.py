from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from fairlearn.reductions import ExponentiatedGradient, DemographicParity


def make_lr_model(random_state=42):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        )),
    ])


def make_fairlearn_dp_model(random_state=42):
    """
    ExponentiatedGradient passes sample_weight internally, and sklearn
    Pipeline requires step-specific names like model__sample_weight.
    Using plain LogisticRegression avoids that issue.
    """
    base_model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
        random_state=random_state,
    )

    return ExponentiatedGradient(
        estimator=base_model,
        constraints=DemographicParity(difference_bound=0.05),
        max_iter=20,
    )
