"""
run_all.py

Runs every experiment in the correct order with a single command:

    python -u run_all.py

Each step is timed. If a step fails, the error is printed and the run
continues with the next step so you still get partial results.
"""

import time
import traceback


STEPS = [
    "Cross-validation (all models, raw + balanced)",
    "Differential privacy sweep",
    "DP-SGD MLP privacy sweep",
    "Causal analysis (DoWhy backdoor)",
    "Counterfactual fairness analysis",
    "SHAP feature importance",
    "VFAE standalone experiment",
    "Transfer learning / subgroup generalisation",
    "Invariant Risk Minimization",
    "Membership inference attack",
    "Unified comparison plots",
]


def run_step(name):
    print(f"\n  Importing modules for: {name} ...", flush=True)

    if name == "Cross-validation (all models, raw + balanced)":
        from main import main as fn
    elif name == "Differential privacy sweep":
        from dp_training import run_dp_experiment as fn
    elif name == "DP-SGD MLP privacy sweep":
        from dp_sgd_mlp_experiment import run_dp_sgd_mlp_experiment as fn
    elif name == "Causal analysis (DoWhy backdoor)":
        from causal_analysis import run_causal_analysis as fn
    elif name == "Counterfactual fairness analysis":
        from counterfactual_diabetes import run_counterfactual_experiment as fn
    elif name == "SHAP feature importance":
        from shap_analysis import main as fn
    elif name == "VFAE standalone experiment":
        from vfae_experiment import run_vfae_experiment as fn
    elif name == "Transfer learning / subgroup generalisation":
        from transfer_experiment import run_transfer_experiment as fn
    elif name == "Invariant Risk Minimization":
        from irm_experiment import run_irm_experiment as fn
    elif name == "Membership inference attack":
        from membership_inference_experiment import run_membership_inference_experiment as fn
    elif name == "Unified comparison plots":
        from compare_all_results import main as fn

    fn()


def main():
    total_start = time.time()
    print("\n" + "=" * 80, flush=True)
    print("FULL PIPELINE RUN", flush=True)
    print("=" * 80, flush=True)

    for i, name in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {name}", flush=True)
        print("-" * 60, flush=True)
        step_start = time.time()
        try:
            run_step(name)
            elapsed = time.time() - step_start
            print(f"\n  Completed in {elapsed:.1f}s", flush=True)
        except Exception:
            elapsed = time.time() - step_start
            print(f"\n  FAILED after {elapsed:.1f}s:", flush=True)
            traceback.print_exc()

    total = time.time() - total_start
    print("\n" + "=" * 80, flush=True)
    print(f"PIPELINE COMPLETE  -  total time: {total/60:.1f} min", flush=True)
    print("Results saved to:  results/", flush=True)
    print("Plots saved to:    graphs/", flush=True)
    print("=" * 80, flush=True)


if __name__ == "__main__":
    main()
