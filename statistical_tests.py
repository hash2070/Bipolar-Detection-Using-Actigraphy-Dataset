"""
APPROACH 3A: Statistical Significance Testing
Test if bipolar vs unipolar differ statistically on activity variability.
Uses t-tests and effect size (Cohen's d) for rigorous comparison.

Null hypothesis: No difference in variability between groups
Alternative: Bipolar has higher variability than unipolar
"""

import numpy as np
import pandas as pd
from pathlib import Path
import json
from datetime import datetime
from scipy import stats

from data_loader import DepresjonDataLoader


class StatisticalAnalyzer:
    """Perform statistical significance testing on bipolar vs unipolar groups."""

    def __init__(self):
        self.results_dir = Path("findings/3A")
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def extract_variability_metrics(self):
        """Extract variability metrics for each participant."""

        loader = DepresjonDataLoader()

        bipolar_variability = []
        unipolar_variability = []
        bipolar_ids = []
        unipolar_ids = []

        unique_pids = loader.metadata[
            loader.metadata['is_condition']
        ]['participant_id'].unique()

        print(f"\nExtracting variability metrics for {len(unique_pids)} participants...")

        for pid in unique_pids:
            meta = loader.metadata[
                loader.metadata['participant_id'] == pid
            ].iloc[0]

            afftype = meta['afftype']
            afftype_name = "Unipolar" if afftype == 2 else "Bipolar"

            # Load activity
            activity_data = loader._load_activity(pid)
            activity = activity_data['activity'].values.astype(float)

            # Normalize
            activity = (activity - activity.mean()) / (activity.std() + 1e-8)

            # Compute daily statistics
            daily_means = []
            for day in range(0, len(activity) - 1440, 1440):
                day_activity = activity[day:day+1440]
                daily_means.append(day_activity.mean())

            if len(daily_means) < 2:
                continue

            daily_means = np.array(daily_means)
            variability = daily_means.std()

            print(f"  {pid:15s} ({afftype_name:8s}): variability={variability:.4f}")

            if afftype == 2:  # Unipolar
                unipolar_variability.append(variability)
                unipolar_ids.append(pid)
            else:  # Bipolar (afftype 1 or 1.5)
                bipolar_variability.append(variability)
                bipolar_ids.append(pid)

        return {
            'bipolar': np.array(bipolar_variability),
            'unipolar': np.array(unipolar_variability),
            'bipolar_ids': bipolar_ids,
            'unipolar_ids': unipolar_ids,
        }

    def compute_cohens_d(self, group1, group2):
        """Compute Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)

        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0

        return (np.mean(group1) - np.mean(group2)) / pooled_std

    def interpret_cohens_d(self, d):
        """Interpret Cohen's d effect size."""
        d_abs = abs(d)
        if d_abs < 0.2:
            return "negligible", d_abs
        elif d_abs < 0.5:
            return "small", d_abs
        elif d_abs < 0.8:
            return "medium", d_abs
        else:
            return "large", d_abs

    def run_statistical_tests(self, data):
        """Perform t-tests and compute effect sizes."""

        bipolar = data['bipolar']
        unipolar = data['unipolar']

        print(f"\n{'='*70}")
        print(f"APPROACH 3A: Statistical Significance Testing")
        print(f"{'='*70}\n")

        # Descriptive statistics
        print(f"DESCRIPTIVE STATISTICS:")
        print(f"  Bipolar (n={len(bipolar)}):")
        print(f"    Mean variability: {bipolar.mean():.4f}")
        print(f"    Std dev: {bipolar.std():.4f}")
        print(f"    Range: {bipolar.min():.4f} - {bipolar.max():.4f}")
        print(f"\n  Unipolar (n={len(unipolar)}):")
        print(f"    Mean variability: {unipolar.mean():.4f}")
        print(f"    Std dev: {unipolar.std():.4f}")
        print(f"    Range: {unipolar.min():.4f} - {unipolar.max():.4f}")

        # Independent samples t-test
        print(f"\n{'-'*70}")
        print(f"INDEPENDENT SAMPLES T-TEST:")
        print(f"H0: No difference in variability between groups")
        print(f"H1: Bipolar has higher variability than unipolar")
        print(f"{'-'*70}")

        t_stat, p_value_two_tailed = stats.ttest_ind(bipolar, unipolar)
        p_value_one_tailed = p_value_two_tailed / 2 if bipolar.mean() > unipolar.mean() else 1 - p_value_two_tailed / 2

        print(f"\n  t-statistic: {t_stat:.4f}")
        print(f"  p-value (two-tailed): {p_value_two_tailed:.4f}")
        print(f"  p-value (one-tailed): {p_value_one_tailed:.4f}")

        # Interpretation
        alpha = 0.05
        if p_value_two_tailed < alpha:
            print(f"  [YES] Statistically significant at alpha={alpha} level")
        else:
            print(f"  [NO] NOT statistically significant at alpha={alpha} level")

        # Cohen's d
        print(f"\n{'-'*70}")
        print(f"EFFECT SIZE (COHEN'S D):")
        print(f"{'-'*70}")

        cohens_d = self.compute_cohens_d(bipolar, unipolar)
        effect_interpretation, d_abs = self.interpret_cohens_d(cohens_d)

        print(f"\n  Cohen's d: {cohens_d:.4f}")
        print(f"  Effect size: {effect_interpretation.upper()} (d={d_abs:.4f})")

        if d_abs < 0.2:
            print(f"  -> Interpretation: Signal is too weak to detect with current sample size")
        elif d_abs < 0.5:
            print(f"  -> Interpretation: Small but potentially meaningful difference")
        else:
            print(f"  -> Interpretation: Meaningful practical difference")

        # Welch's t-test (doesn't assume equal variance)
        print(f"\n{'-'*70}")
        print(f"WELCH'S T-TEST (more robust to unequal variances):")
        print(f"{'-'*70}")

        t_stat_welch, p_value_welch = stats.ttest_ind(bipolar, unipolar, equal_var=False)
        print(f"\n  t-statistic: {t_stat_welch:.4f}")
        print(f"  p-value: {p_value_welch:.4f}")

        # Additional tests
        print(f"\n{'-'*70}")
        print(f"ADDITIONAL TESTS:")
        print(f"{'-'*70}")

        # Mann-Whitney U (non-parametric alternative)
        u_stat, p_value_mw = stats.mannwhitneyu(bipolar, unipolar, alternative='two-sided')
        print(f"\n  Mann-Whitney U: {u_stat:.4f}, p={p_value_mw:.4f}")
        print(f"  (Non-parametric test, no normality assumption)")

        # Levene's test for equal variances
        levene_stat, p_levene = stats.levene(bipolar, unipolar)
        print(f"\n  Levene's test for equal variances: {levene_stat:.4f}, p={p_levene:.4f}")
        if p_levene < 0.05:
            print(f"  -> Variances are DIFFERENT (Welch's t-test is more appropriate)")
        else:
            print(f"  -> Variances are EQUAL (standard t-test is appropriate)")

        return {
            'bipolar_n': int(len(bipolar)),
            'unipolar_n': int(len(unipolar)),
            'bipolar_mean': float(bipolar.mean()),
            'bipolar_std': float(bipolar.std()),
            'unipolar_mean': float(unipolar.mean()),
            'unipolar_std': float(unipolar.std()),
            't_statistic': float(t_stat),
            't_statistic_welch': float(t_stat_welch),
            'p_value': float(p_value_two_tailed),
            'p_value_welch': float(p_value_welch),
            'p_value_mannwhitney': float(p_value_mw),
            'cohens_d': float(cohens_d),
            'effect_size_interpretation': effect_interpretation,
            'is_significant': bool(p_value_two_tailed < 0.05),
        }

    def save_results(self, results):
        """Save statistical results to JSON."""

        summary = {
            'timestamp': datetime.now().isoformat(),
            'approach': '3A - Statistical Significance Testing',
            'analysis': 'Activity Variability (bipolar vs unipolar)',
            'results': results,
        }

        json_path = self.results_dir / "results_3a.json"
        with open(json_path, 'w') as f:
            json.dump(summary, f, indent=2)

        print(f"\n[SAVED] Results saved to {json_path}")

        return summary


if __name__ == '__main__':
    analyzer = StatisticalAnalyzer()

    # Extract metrics
    data = analyzer.extract_variability_metrics()

    # Run tests
    results = analyzer.run_statistical_tests(data)

    # Save
    summary = analyzer.save_results(results)

    print("\n" + "="*70)
    print("APPROACH 3A COMPLETE")
    print("="*70)
    print(f"Key finding: p-value = {results['p_value']:.4f}, Cohen's d = {results['cohens_d']:.4f}")
    print(f"Statistically significant: {'YES' if results['is_significant'] else 'NO'}")
    print("="*70 + "\n")
