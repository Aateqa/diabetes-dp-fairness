from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from diffprivlib.models import LogisticRegression as DPLogisticRegression

from config import RANDOM_STATE


def make_non_private_lr(random_state=RANDOM_STATE):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=random_state,
        )),
    ])


def make_dp_lr(epsilon, random_state=RANDOM_STATE):
    """
    Differentially private Logistic Regression via diffprivlib.
    data_norm bounds the L2 norm of each training example after scaling.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", DPLogisticRegression(
            epsilon=epsilon,
            data_norm=10.0,
            max_iter=1000,
            random_state=random_state,
        )),
    ])
