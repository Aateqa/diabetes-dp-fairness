import os
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import networkx as nx

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from dowhy import CausalModel

from config import (
    RESULTS_DIR,
    GRAPHS_DIR,
    RANDOM_STATE,
)

from data_loader_diabetes import load_raw_diabetes_data


def create_binary_bmi_treatment(df):
    """
    Creates a binary BMI treatment variable.

    Treatment:
        1 = obese, BMI >= 30
        0 = not obese, BMI < 30

    This makes the causal effect easier to interpret:
    effect of obesity status on diabetes probability.
    """
    df = df.copy()
    df["BMI_obese"] = (df["BMI"] >= 30).astype(int)
    return df


def compute_naive_correlation(df):
    """
    Computes naive association between obesity and diabetes.

    This is not causal. It simply compares diabetes rates between
    obese and non-obese groups without adjustment.
    """
    grouped = df.groupby("BMI_obese")["Diabetes_binary"].mean()

    non_obese_rate = grouped.loc[0]
    obese_rate = grouped.loc[1]

    naive_difference = obese_rate - non_obese_rate

    return {
        "non_obese_diabetes_rate": non_obese_rate,
        "obese_diabetes_rate": obese_rate,
        "naive_difference": naive_difference,
    }


def estimate_adjusted_effect_with_logistic_regression(df):
    """
    Estimates adjusted obesity effect using logistic regression.

    Diabetes_binary ~ BMI_obese + Age + Income + PhysActivity

    We estimate the average difference in predicted diabetes probability
    when BMI_obese is changed from 0 to 1 while keeping all covariates fixed.
    """
    X = df[["BMI_obese", "Age", "Income", "PhysActivity"]].copy()
    y = df["Diabetes_binary"].astype(int)

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(
            max_iter=1000,
            solver="liblinear",
            random_state=RANDOM_STATE,
        )),
    ])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model.fit(X, y)

    X_control = X.copy()
    X_control["BMI_obese"] = 0

    X_treated = X.copy()
    X_treated["BMI_obese"] = 1

    prob_control = model.predict_proba(X_control)[:, 1]
    prob_treated = model.predict_proba(X_treated)[:, 1]

    adjusted_effect = float(np.mean(prob_treated - prob_control))

    return adjusted_effect


def build_causal_graph_dot():
    """
    Defines the causal DAG in DOT format for DoWhy.

    Required Bari Appi DAG:
        Age -> BMI -> Diabetes
        Income -> PhysActivity -> Diabetes
        Sex -> Age

    Additional direct edges are included because Age and Income can plausibly
    affect diabetes risk directly in this dataset.
    """
    return """
    digraph {
        Sex -> Age;
        Age -> BMI_obese;
        Age -> Diabetes_binary;
        BMI_obese -> Diabetes_binary;
        Income -> PhysActivity;
        Income -> Diabetes_binary;
        PhysActivity -> Diabetes_binary;
    }
    """


def save_dag_image(output_path):
    """
    Saves the DAG as a graph image using networkx/matplotlib.

    This avoids requiring system Graphviz binaries.
    """
    graph = nx.DiGraph()

    edges = [
        ("Sex", "Age"),
        ("Age", "BMI_obese"),
        ("Age", "Diabetes_binary"),
        ("BMI_obese", "Diabetes_binary"),
        ("Income", "PhysActivity"),
        ("Income", "Diabetes_binary"),
        ("PhysActivity", "Diabetes_binary"),
    ]

    graph.add_edges_from(edges)

    plt.figure(figsize=(10, 7))

    pos = nx.spring_layout(
        graph,
        seed=RANDOM_STATE,
        k=1.2,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        node_size=2500,
        alpha=0.9,
    )

    nx.draw_networkx_edges(
        graph,
        pos,
        arrows=True,
        arrowsize=20,
        width=2,
        alpha=0.8,
    )

    nx.draw_networkx_labels(
        graph,
        pos,
        font_size=9,
        font_weight="bold",
    )

    plt.title("Causal DAG for Diabetes Analysis")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved DAG image: {output_path}")


def run_dowhy_causal_estimate(df):
    """
    Runs DoWhy causal estimation.

    Treatment:
        BMI_obese

    Outcome:
        Diabetes_binary

    Main adjustment:
        Age, via the graph backdoor structure.
    """
    causal_graph = build_causal_graph_dot()

    model = CausalModel(
        data=df,
        treatment="BMI_obese",
        outcome="Diabetes_binary",
        graph=causal_graph,
    )

    identified_estimand = model.identify_effect(
        proceed_when_unidentifiable=True,
    )

    estimate = model.estimate_effect(
        identified_estimand,
        method_name="backdoor.linear_regression",
        test_significance=False,
    )

    return identified_estimand, estimate


def run_causal_analysis():
    os.makedirs(RESULTS_DIR / "causal", exist_ok=True)
    os.makedirs(GRAPHS_DIR / "causal", exist_ok=True)

    print("\n" + "=" * 80)
    print("Running causal analysis")
    print("=" * 80)

    feature_sets, y, fairness_df, full_df = load_raw_diabetes_data(print_summary=False)

    required_cols = [
        "Diabetes_binary",
        "BMI",
        "Age",
        "Income",
        "PhysActivity",
        "Sex",
    ]

    missing_cols = [col for col in required_cols if col not in full_df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns for causal analysis: {missing_cols}")

    df = full_df[required_cols].copy()
    df = create_binary_bmi_treatment(df)

    # Keep only columns needed for DoWhy.
    causal_df = df[
        [
            "Diabetes_binary",
            "BMI_obese",
            "Age",
            "Income",
            "PhysActivity",
            "Sex",
        ]
    ].copy()

    print(f"Causal dataframe shape: {causal_df.shape}")
    print("\nTreatment distribution:")
    print(causal_df["BMI_obese"].value_counts().sort_index())

    naive = compute_naive_correlation(causal_df)

    print("\nNaive association:")
    print(f"Non-obese diabetes rate: {naive['non_obese_diabetes_rate']:.4f}")
    print(f"Obese diabetes rate: {naive['obese_diabetes_rate']:.4f}")
    print(f"Naive difference: {naive['naive_difference']:.4f}")

    adjusted_logistic_effect = estimate_adjusted_effect_with_logistic_regression(causal_df)

    print("\nAdjusted logistic estimate:")
    print(f"Average adjusted probability difference: {adjusted_logistic_effect:.4f}")

    print("\nRunning DoWhy backdoor estimate...")

    identified_estimand, dowhy_estimate = run_dowhy_causal_estimate(causal_df)

    dowhy_value = float(dowhy_estimate.value)

    print("\nDoWhy identified estimand:")
    print(identified_estimand)

    print("\nDoWhy estimate:")
    print(dowhy_estimate)
    print(f"\nDoWhy causal estimate value: {dowhy_value:.4f}")

    results_df = pd.DataFrame([
        {
            "analysis": "naive_difference",
            "description": "Difference in diabetes rate between obese and non-obese groups without adjustment",
            "estimate": naive["naive_difference"],
        },
        {
            "analysis": "adjusted_logistic_probability_difference",
            "description": "Average predicted probability difference for BMI_obese, adjusting for Age",
            "estimate": adjusted_logistic_effect,
        },
        {
            "analysis": "dowhy_backdoor_linear_regression",
            "description": "DoWhy backdoor estimate using declared DAG",
            "estimate": dowhy_value,
        },
    ])

    output_csv = RESULTS_DIR / "causal" / "causal_bmi_effect_results.csv"
    output_dag = GRAPHS_DIR / "causal" / "causal_dag.png"

    results_df.to_csv(output_csv, index=False)

    print(f"\nSaved causal results: {output_csv}")

    save_dag_image(output_dag)

    print("\nCausal analysis complete.")

    return results_df


if __name__ == "__main__":
    run_causal_analysis()