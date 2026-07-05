def get_models(random_state=42):
    from models.logistic import make_fairlearn_dp_model, make_lr_model
    from models.neural import make_mlp_model
    from models.tree_ensemble import (
        make_catboost_model,
        make_lgbm_model,
        make_rf_model,
        make_xgb_model,
    )
    from models.vfae import VFAEClassifier

    return {
        "Logistic Regression": make_lr_model(random_state),
        "Random Forest": make_rf_model(random_state),
        "XGBoost": make_xgb_model(random_state),
        "LightGBM": make_lgbm_model(random_state),
        "CatBoost": make_catboost_model(random_state),
        "MLP": make_mlp_model(random_state),
        "Fairlearn-DP": make_fairlearn_dp_model(random_state),
        "VFAE": VFAEClassifier(random_state=random_state),
    }


def make_non_private_lr(*args, **kwargs):
    from models.dp_model import make_non_private_lr as _make_non_private_lr

    return _make_non_private_lr(*args, **kwargs)


def make_dp_lr(*args, **kwargs):
    from models.dp_model import make_dp_lr as _make_dp_lr

    return _make_dp_lr(*args, **kwargs)
