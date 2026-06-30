from pathlib import Path

DATA_DIR = Path("data")
RESULTS_DIR = Path("results")
GRAPHS_DIR = Path("graphs")

RAW_DATA_FILE = DATA_DIR / "diabetes_binary_health_indicators_BRFSS2015.csv"
BALANCED_DATA_FILE = DATA_DIR / "diabetes_binary_5050split_health_indicators_BRFSS2015.csv"

TARGET_COLUMN = "Diabetes_binary"

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS = 2

PROTECTED_ATTRIBUTES = [
    "sex_group",
    "age_group",
    "income_group",
    "education_group",
    "intersection_group",
]

SENSITIVE_COLUMNS = [
    "Sex",
    "Age",
]

PROXY_COLUMNS = [
    "Income",
    "Education",
    "Education_fair_group",
]

FEATURE_SET_NAMES = [
    "Original Features",
    "Without Sensitive Attributes",
    "Without Sensitive Attributes + Proxy-Reduced Features",
]

BINARY_COLUMNS = [
    "HighBP",
    "HighChol",
    "CholCheck",
    "Smoker",
    "Stroke",
    "HeartDiseaseorAttack",
    "PhysActivity",
    "Fruits",
    "Veggies",
    "HvyAlcoholConsump",
    "AnyHealthcare",
    "NoDocbcCost",
    "DiffWalk",
    "Sex",
]

ORDINAL_COLUMNS = [
    "Age",
    "GenHlth",
    "Education",
    "Income",
]