from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

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


def make_xgb_model(random_state=42):
    return XGBClassifier(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=6,
        eval_metric="logloss",
        random_state=random_state,
        n_jobs=-1,
    )


def make_lgbm_model(random_state=42):
    return LGBMClassifier(
        n_estimators=200,
        max_depth=8,
        learning_rate=0.05,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )


def make_catboost_model(random_state=42):
    return CatBoostClassifier(
        iterations=200,
        depth=4,
        learning_rate=0.05,
        loss_function="Logloss",
        eval_metric="AUC",
        class_weights=[1, 6],
        random_seed=random_state,
        verbose=False,
    )


def make_rf_model(random_state=42):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


def make_mlp_model(random_state=42):
    """
    MLPClassifier does not take class_weight in the constructor.
    Class balancing is handled in cross_validation.py using sample_weight.
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


def make_fairlearn_dp_model(random_state=42):
    """
    Fairlearn constrained model.

    Important:
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
        constraints=DemographicParity(),
        eps=0.05,
        max_iter=20,
    )


def get_models(random_state=42):
    return {
        "Logistic Regression": make_lr_model(random_state),
        "Random Forest": make_rf_model(random_state),
        "XGBoost": make_xgb_model(random_state),
        "LightGBM": make_lgbm_model(random_state),
        "CatBoost": make_catboost_model(random_state),
        "MLP": make_mlp_model(random_state),
        "Fairlearn-DP": make_fairlearn_dp_model(random_state),
    }