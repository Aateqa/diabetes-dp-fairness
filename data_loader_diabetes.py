import pandas as pd

from config import (
    DATA_FILE,
    TARGET_COLUMN,
    SENSITIVE_COLUMNS,
    PROXY_COLUMNS,
    PROTECTED_ATTRIBUTES,
)


def create_sex_group(value):
    """
    BRFSS Sex encoding:
    0 = female
    1 = male
    """
    if value == 0:
        return "female"
    if value == 1:
        return "male"
    return "unknown"


def create_age_group(value):
    """
    BRFSS Age is already ordinal-coded from 1 to 13.

    Approximate grouping:
    1-5   = younger adults
    6-9   = middle-aged adults
    10-13 = older adults
    """
    if value <= 5:
        return "young"
    if value <= 9:
        return "middle_age"
    return "older"


def create_income_group(value):
    """
    BRFSS Income is ordinal-coded from 1 to 8.

    Approximate grouping:
    1-3 = low income
    4-6 = middle income
    7-8 = high income
    """
    if value <= 3:
        return "low_income"
    if value <= 6:
        return "middle_income"
    return "high_income"


def create_education_group(value):
    """
    BRFSS Education is ordinal-coded from 1 to 6.

    Approximate grouping:
    1-3 = low education
    4-5 = middle education
    6   = high education
    """
    if value <= 3:
        return "low_education"
    if value <= 5:
        return "middle_education"
    return "high_education"


def load_diabetes_dataframe():
    """
    Loads the diabetes dataset and creates protected group columns.
    """
    df = pd.read_csv(DATA_FILE)

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

    df = df.copy()

    df["sex_group"] = df["Sex"].apply(create_sex_group)
    df["age_group"] = df["Age"].apply(create_age_group)
    df["income_group"] = df["Income"].apply(create_income_group)
    df["education_group"] = df["Education"].apply(create_education_group)
    df["intersection_group"] = df["sex_group"] + "_" + df["age_group"]

    return df


def build_feature_sets(df):
    """
    Builds three feature sets:

    1. Original Features:
       Keeps all original input features.

    2. Without Sensitive Attributes:
       Removes direct sensitive attributes: Sex and Age.

    3. Without Sensitive Attributes + Proxy-Reduced Features:
       Removes Sex, Age, Income, and Education.
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

    # Feature Set 1: Original features
    X_original = df.drop(columns=base_drop_columns, errors="ignore")

    # Feature Set 2: Remove direct sensitive attributes
    X_no_sensitive = df.drop(
        columns=base_drop_columns + SENSITIVE_COLUMNS,
        errors="ignore",
    )

    # Feature Set 3: Remove sensitive attributes and proxy variables
    X_proxy_reduced = df.drop(
        columns=base_drop_columns + SENSITIVE_COLUMNS + PROXY_COLUMNS,
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
    Prints a quick EDA summary to confirm the data loaded correctly.
    """
    print("\n" + "=" * 80)
    print("DIABETES DATASET EDA SUMMARY")
    print("=" * 80)

    print("\nDataset shape:")
    print(df.shape)

    print("\nColumns:")
    print(list(df.columns))

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


def load_diabetes_data(print_summary=True):
    """
    Main function used by other files.

    Returns:
        feature_sets: dict of feature set name -> X dataframe
        y: target labels
        fairness_df: protected group dataframe
        df: full dataframe with generated protected group columns
    """
    df = load_diabetes_dataframe()
    feature_sets, y, fairness_df = build_feature_sets(df)

    if print_summary:
        print_eda_summary(df, feature_sets, y, fairness_df)

    return feature_sets, y, fairness_df, df


if __name__ == "__main__":
    load_diabetes_data(print_summary=True)