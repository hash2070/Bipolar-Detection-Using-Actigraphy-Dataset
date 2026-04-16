"""
Master runner script for both experiments with visualizations.
Runs Exp 1, then Exp 2, then generates all visualizations and summary.
"""

import json
import numpy as np
from pathlib import Path
import sys

from train_exp1 import run_experiment_1
from train_exp2 import run_experiment_2
from visualize import generate_all_visualizations


def run_all_experiments():
    """Run both experiments and generate results summary."""
    print("\n" + "=" * 70)
    print("BIPOLAR DETECTION VIA ACTIGRAPHY - COMPLETE PIPELINE")
    print("=" * 70)

    results_dir = Path("results")
    results_dir.mkdir(exist_ok=True)

    # ========== EXPERIMENT 1 ==========
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 1: Healthy vs. Depressed")
    print("=" * 70)

    try:
        exp1_results, exp1_y_true, exp1_y_pred, exp1_y_probs = run_experiment_1()
        exp1_success = True
    except Exception as e:
        print(f"\n[ERROR] Experiment 1 failed: {e}")
        exp1_success = False
        import traceback
        traceback.print_exc()

    # ========== EXPERIMENT 2 ==========
    print("\n" + "=" * 70)
    print("RUNNING EXPERIMENT 2: Bipolar vs. Unipolar")
    print("=" * 70)

    try:
        exp2_results, exp2_y_true, exp2_y_pred, exp2_y_probs = run_experiment_2()
        exp2_success = True
    except Exception as e:
        print(f"\n[ERROR] Experiment 2 failed: {e}")
        exp2_success = False
        import traceback
        traceback.print_exc()

    # ========== GENERATE VISUALIZATIONS ==========
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)

    if exp1_success:
        try:
            generate_all_visualizations(1, str(results_dir))
        except Exception as e:
            print(f"[WARNING] Could not generate Experiment 1 visualizations: {e}")

    if exp2_success:
        try:
            generate_all_visualizations(2, str(results_dir))
        except Exception as e:
            print(f"[WARNING] Could not generate Experiment 2 visualizations: {e}")

    # ========== GENERATE SUMMARY REPORT ==========
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if exp1_success:
        print("\nEXPERIMENT 1 RESULTS (Healthy vs. Depressed):")
        for key, val in exp1_results.items():
            if key != 'confusion_matrix':
                print(f"  {key:15s}: {val:.4f}")

    if exp2_success:
        print("\nEXPERIMENT 2 RESULTS (Bipolar vs. Unipolar):")
        for key, val in exp2_results.items():
            if key != 'confusion_matrix':
                print(f"  {key:15s}: {val:.4f}")

    print("\n" + "=" * 70)
    print(f"Results saved to: {results_dir}")
    print("=" * 70)


if __name__ == "__main__":
    run_all_experiments()
