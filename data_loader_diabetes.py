import pandas as pd

from config import (
    RAW_DATA_FILE,
    BALANCED_DATA_FILE,
    TARGET_COLUMN,
    SENSITIVE_COLUMNS,
    PROXY_COLUMNS,
    PROTECTED_ATTRIBUTES,
    BINARY_COLUMNS,
    ORDINAL_COLUMNS,
)

from feature_engineering import feature_engineering


SENSITIVE_DERIVED_COLUMNS = [
    "age_bmi_risk",
]


def create_sex_group(value):
    if value == 0:
        return "female"
    if value == 1:
        return "male"
    return "unknown"


def create_age_group(value):
    if value <= 5:
        return "young"
    if value <= 9:
        return "middle_age"
    return "older"


def create_income_group(value):
    if value <= 3:
        return "low_income"
    if value <= 6:
        return "middle_income"
    return "high_income"


def create_education_group(value):
    """
    Uses merged Education_fair_group, where original levels 1 and 2 are combined.
    """
    if value <= 2:
        return "low_education"
    if value <= 5:
        return "middle_education"
    return "high_education"


def preprocess_dataframe(df):
    """
    Applies required dataset fixes before modelling.

    Fixes:
    - Remove duplicate rows before any processing.
    - Cap BMI at 60.
    - Cast binary columns to int8.
    - Cast ordinal columns to int16.
    - Apply diabetes-specific feature engineering.
    """
    df = df.copy()

    before = len(df)
    df = df.drop_duplicates().reset_index(drop=True)
    removed = before - len(df)
    print(f"Removed duplicate rows: {removed}")

    if "BMI" in df.columns:
        df["BMI"] = df["BMI"].clip(upper=60)

    for col in BINARY_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int8")

    for col in ORDINAL_COLUMNS:
        if col in df.columns:
            df[col] = df[col].astype("int16")

    if TARGET_COLUMN in df.columns:
        df[TARGET_COLUMN] = df[TARGET_COLUMN].astype("int8")

    df = feature_engineering(df)

    return df


def load_diabetes_dataframe(data_file=RAW_DATA_FILE):
    """
    Loads one diabetes CSV file, applies preprocessing, and creates fairness group columns.
    """
    df = pd.read_csv(data_file)

    if TARGET_COLUMN not in df.columns:
        raise ValueError(
            f"Target column '{TARGET_COLUMN}' not found. "
            f"Available columns: {list(df.columns)}"
        )

    required_columns = ["Sex", "Age", "Income", "Education"]

    for col in required_columns:
        if col not in df.columns:
            raise ValueError(
                f"Required column '{col}' not found. "
                f"Available columns: {list(df.columns)}"
            )

    df = preprocess_dataframe(df)

    df["sex_group"] = df["Sex"].apply(create_sex_group)
    df["age_group"] = df["Age"].apply(create_age_group)
    df["income_group"] = df["Income"].apply(create_income_group)

    # Use merged education group for fairness.
    df["education_group"] = df["Education_fair_group"].apply(create_education_group)

    df["intersection_group"] = df["sex_group"] + "_" + df["age_group"]

    return df


def build_feature_sets(df):
    """
    Builds three feature sets:

    1. Original Features:
       Keeps all cleaned and engineered features.

    2. Without Sensitive Attributes:
       Removes direct sensitive attributes and sensitive-derived engineered features.

    3. Without Sensitive Attributes + Proxy-Reduced Features:
       Removes direct sensitive attributes, sensitive-derived engineered features,
       and socioeconomic proxy columns.
    """
    y = df[TARGET_COLUMN].astype(int)

    fairness_df = df[PROTECTED_ATTRIBUTES].copy()

    generated_group_columns = [
        "sex_group",
        "age_group",
        "income_group",
        "education_group",
        "intersection_group",
    ]

    base_drop_columns = [TARGET_COLUMN] + generated_group_columns

    # Feature Set 1: all cleaned original + engineered features.
    X_original = df.drop(
        columns=base_drop_columns,
        errors="ignore",
    )

    # Feature Set 2: remove direct sensitive attributes and features derived from them.
    X_no_sensitive = df.drop(
        columns=base_drop_columns + SENSITIVE_COLUMNS + SENSITIVE_DERIVED_COLUMNS,
        errors="ignore",
    )

    # Feature Set 3: remove sensitive attributes, sensitive-derived columns, and proxies.
    X_proxy_reduced = df.drop(
        columns=(
            base_drop_columns
            + SENSITIVE_COLUMNS
            + SENSITIVE_DERIVED_COLUMNS
            + PROXY_COLUMNS
        ),
        errors="ignore",
    )

    feature_sets = {
        "Original Features": X_original,
        "Without Sensitive Attributes": X_no_sensitive,
        "Without Sensitive Attributes + Proxy-Reduced Features": X_proxy_reduced,
    }

    return feature_sets, y, fairness_df


def print_eda_summary(df, feature_sets, y, fairness_df):
    """
    Prints a quick EDA summary to confirm that loading, preprocessing,
    grouping, and feature-set construction worked correctly.
    """
    print("\n" + "=" * 80)
    print("DIABETES DATASET EDA SUMMARY")
    print("=" * 80)

    print("\nDataset shape:")
    print(df.shape)

    print("\nTarget distribution:")
    print(y.value_counts().sort_index())

    print("\nTarget distribution percentage:")
    print((y.value_counts(normalize=True).sort_index() * 100).round(2))

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0]

    if len(missing) == 0:
        print("No missing values found.")
    else:
        print(missing)

    print("\nProtected attribute counts:")

    for attr in fairness_df.columns:
        print(f"\n{attr}:")
        print(fairness_df[attr].value_counts())

    print("\nFeature set shapes:")

    for name, X in feature_sets.items():
        print(f"{name}: {X.shape}")

    print("\nFeature set columns:")

    for name, X in feature_sets.items():
        print("\n" + "-" * 80)
        print(name)
        print("-" * 80)
        print(list(X.columns))

    print("\n" + "=" * 80)
    print("EDA COMPLETE")
    print("=" * 80)


def load_diabetes_data(data_file=RAW_DATA_FILE, print_summary=True):
    """
    Main function used by other files.

    Returns:
        feature_sets: dict of feature set name -> X dataframe
        y: target labels
        fairness_df: protected group dataframe
        df: full dataframe with generated protected group columns
    """
    df = load_diabetes_dataframe(data_file=data_file)
    feature_sets, y, fairness_df = build_feature_sets(df)

    if print_summary:
        print_eda_summary(df, feature_sets, y, fairness_df)

    return feature_sets, y, fairness_df, df


def load_raw_diabetes_data(print_summary=True):
    return load_diabetes_data(
        data_file=RAW_DATA_FILE,
        print_summary=print_summary,
    )


def load_balanced_diabetes_data(print_summary=True):
    return load_diabetes_data(
        data_file=BALANCED_DATA_FILE,
        print_summary=print_summary,
    )


if __name__ == "__main__":
    load_raw_diabetes_data(print_summary=True)