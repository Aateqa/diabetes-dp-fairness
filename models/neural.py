from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def make_mlp_model(random_state=42):
    """
    MLPClassifier does not accept class_weight in the constructor.
    Class balancing is handled in cross_validation.py via sample_weight.
    """
    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            solver="adam",
            alpha=1e-4,
            max_iter=100,
            early_stopping=True,
            n_iter_no_change=10,
            random_state=random_state,
        )),
    ])
