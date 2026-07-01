from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier


def make_rf_model(random_state=42):
    return RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight="balanced",
        random_state=random_state,
        n_jobs=-1,
    )


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
