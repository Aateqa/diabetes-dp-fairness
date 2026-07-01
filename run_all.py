"""
run_all.py

Runs every experiment in the correct order with a single command:

    python run_all.py

Each step is timed. If a step fails, the error is printed and the run
continues with the next step so you still get partial results.
"""

import time
import traceback

from main import main as run_main
from dp_training import run_dp_experiment
from dp_sgd_mlp_experiment import run_dp_sgd_mlp_experiment
from causal_analysis import run_causal_analysis
from counterfactual_diabetes import run_counterfactual_experiment as run_counterfactual_analysis
from shap_analysis import main as run_shap_analysis
from vfae_experiment import run_vfae_experiment
from transfer_experiment import run_transfer_experiment
from irm_experiment import run_irm_experiment
from compare_all_results import main as run_comparison


STEPS = [
    ("Cross-validation (all models, raw + balanced)",   run_main),
    ("Differential privacy sweep",                       run_dp_experiment),
    ("DP-SGD MLP privacy sweep",                         run_dp_sgd_mlp_experiment),
    ("Causal analysis (DoWhy backdoor)",                 run_causal_analysis),
    ("Counterfactual fairness analysis",                 run_counterfactual_analysis),
    ("SHAP feature importance",                          run_shap_analysis),
    ("VFAE standalone experiment",                       run_vfae_experiment),
    ("Transfer learning / subgroup generalisation",      run_transfer_experiment),
    ("Invariant Risk Minimization",                      run_irm_experiment),
    ("Unified comparison plots",                         run_comparison),
]


def main():
    total_start = time.time()
    print("\n" + "=" * 80)
    print("FULL PIPELINE RUN")
    print("=" * 80)

    for i, (name, fn) in enumerate(STEPS, 1):
        print(f"\n[{i}/{len(STEPS)}] {name}")
        print("-" * 60)
        step_start = time.time()
        try:
            fn()
            elapsed = time.time() - step_start
            print(f"\n    Completed in {elapsed:.1f}s")
        except Exception:
            elapsed = time.time() - step_start
            print(f"\n    FAILED after {elapsed:.1f}s:")
            traceback.print_exc()

    total = time.time() - total_start
    print("\n" + "=" * 80)
    print(f"PIPELINE COMPLETE  -  total time: {total/60:.1f} min")
    print("Results saved to:  results/")
    print("Plots saved to:    graphs/")
    print("=" * 80)


if __name__ == "__main__":
    main()
