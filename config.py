DATA_FILE = "diabetes_binary_health_indicators_BRFSS2015.csv"

RESULTS_DIR = "results"
GRAPHS_DIR = "graphs"

TARGET_COLUMN = "Diabetes_binary"

RANDOM_STATE = 42
TEST_SIZE = 0.25
N_SPLITS = 5

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
]

FEATURE_SET_NAMES = [
    "Original Features",
    "Without Sensitive Attributes",
    "Without Sensitive Attributes + Proxy-Reduced Features",
]