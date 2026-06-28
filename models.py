from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def get_models(random_state=42):
    """
    Returns baseline models for diabetes fairness analysis.
    """

    models = {
        "Logistic Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(
                max_iter=1000,
                class_weight="balanced",
                random_state=random_state,
            )),
        ]),

        "Decision Tree": DecisionTreeClassifier(
            max_depth=8,
            class_weight="balanced",
            random_state=random_state,
        ),

        "Random Forest": RandomForestClassifier(
            n_estimators=200,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),

        "XGBoost": XGBClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.9,
            colsample_bytree=0.9,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        ),

        "LightGBM": LGBMClassifier(
            n_estimators=200,
            max_depth=-1,
            learning_rate=0.05,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
            verbose=-1,
        ),

        "CatBoost": CatBoostClassifier(
            iterations=200,
            depth=4,
            learning_rate=0.05,
            loss_function="Logloss",
            eval_metric="AUC",
            random_seed=random_state,
            verbose=False,
        ),
    }

    stacking_estimators = [
        ("lr", models["Logistic Regression"]),
        ("rf", RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        )),
        ("xgb", XGBClassifier(
            n_estimators=100,
            max_depth=3,
            learning_rate=0.05,
            eval_metric="logloss",
            random_state=random_state,
            n_jobs=-1,
        )),
    ]

    models["Stacking Ensemble"] = StackingClassifier(
        estimators=stacking_estimators,
        final_estimator=LogisticRegression(max_iter=1000),
        n_jobs=-1,
    )

    return models