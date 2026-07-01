from models.logistic import make_lr_model, make_fairlearn_dp_model
from models.tree_ensemble import make_rf_model, make_xgb_model, make_lgbm_model, make_catboost_model
from models.neural import make_mlp_model
from models.dp_model import make_non_private_lr, make_dp_lr
from models.vfae import VFAE, VFAEClassifier, DiabetesTensorDataset, kl_divergence


def get_models(random_state=42):
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
