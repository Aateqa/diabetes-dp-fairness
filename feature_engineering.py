import pandas as pd


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies diabetes-specific feature engineering.

    Required changes:
    - Replace raw MentHlth and PhysHlth with binary burden indicators.
    - Add BMI category and clinical interaction features.
    - Merge Education levels 1 and 2 for fairness grouping.
    """
    df = df.copy()

    if "MentHlth" in df.columns:
        df["any_mental_health_days"] = (df["MentHlth"] > 0).astype("int8")
        df["high_mental_burden"] = (df["MentHlth"] >= 14).astype("int8")
        df = df.drop(columns=["MentHlth"])

    if "PhysHlth" in df.columns:
        df["any_physical_health_days"] = (df["PhysHlth"] > 0).astype("int8")
        df["high_physical_burden"] = (df["PhysHlth"] >= 14).astype("int8")
        df = df.drop(columns=["PhysHlth"])

    df["bmi_category"] = pd.cut(
        df["BMI"],
        bins=[0, 18.5, 25, 30, 35, 40, 60],
        labels=[0, 1, 2, 3, 4, 5],
        include_lowest=True,
    ).astype("int8")

    df["cardiometabolic_risk"] = (
        df["HighBP"]
        + df["HighChol"]
        + df["HeartDiseaseorAttack"]
        + df["Stroke"]
    ).astype("int8")

    df["lifestyle_score"] = (
        df["PhysActivity"]
        + df["Fruits"]
        + df["Veggies"]
        - df["Smoker"]
        - df["HvyAlcoholConsump"]
    ).astype("int8")

    df["bp_chol_combined"] = (df["HighBP"] * df["HighChol"]).astype("int8")

    df["obese_inactive"] = (
        (df["BMI"] >= 30) & (df["PhysActivity"] == 0)
    ).astype("int8")

    df["poor_health_no_access"] = (
        (df["GenHlth"] >= 4) & (df["AnyHealthcare"] == 0)
    ).astype("int8")

    df["age_bmi_risk"] = (
        df["Age"] * (df["BMI"] >= 30).astype("int8")
    ).astype("int16")

    df["cardiovascular_and_bp"] = (
        df["HeartDiseaseorAttack"] * df["HighBP"]
    ).astype("int8")

    df["poor_general_health"] = (df["GenHlth"] >= 4).astype("int8")

    # Merge Education levels 1 and 2 for fairness grouping.
    df["Education_fair_group"] = df["Education"].replace({1: 2}).astype("int16")

    return df