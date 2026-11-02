"""
Saving each run's results to disk.

Printing to the terminal is fine while you're watching it happen, but it's gone the moment you scroll past it or close the window. This module writes the full report (accuracy, classification report, fairness table) to a timestamped file in `results/` instead, so you can open it again later, or compare two runs side by side after changing something in config.yaml.
"""
import os
from datetime import datetime

import pandas as pd


def save_scores(path: str, row: dict) -> pd.DataFrame:
    """Append one run to the scoreboard CSV and return all recorded runs."""
    runs = (
        pd.read_csv(path, dtype={"run": str})
        if os.path.exists(path)
        else pd.DataFrame()
    )
    runs = pd.concat([runs, pd.DataFrame([row])], ignore_index=True)
    parent = os.path.dirname(path)
    if parent:
        os.makedirs(parent, exist_ok=True)
    runs.to_csv(path, index=False)
    return runs


def save_run(
    results_dir: str,
    config: dict,
    report_text: str,
    run_id: str | None = None,
) -> str:
    """
    Writes one run's full report to a timestamped .txt file inside
    `results_dir` (created automatically if it doesn't exist yet) and
    returns the path that was written.
    """
    os.makedirs(results_dir, exist_ok=True)

    timestamp = run_id or datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    path = os.path.join(results_dir, f"run_{timestamp}.txt")
    cv_config = config.get("cv", {})
    tuning_config = config.get("tuning", {})
    holdout_config = config.get("holdout", {})

    header = (
        f"Run: {timestamp}\n"
        f"Model: {config['model']['type']}  params={config['model'].get('params', {})}\n"
        f"Locked test size: {config['test_set']['size']}  "
        f"random_state: {config['test_set']['random_state']}\n"
        + (
            f"Holdout validation size: {holdout_config['size']}  "
            f"random_state: {holdout_config['random_state']}\n"
            if holdout_config
            else ""
        )
        + f"CV: {cv_config.get('n_splits')} folds, "
        f"metrics={cv_config.get('metrics', [cv_config.get('scoring', 'accuracy')])}, "
        f"selection_metric={cv_config.get('selection_metric', cv_config.get('scoring', 'accuracy'))}, "
        f"random_state={cv_config.get('random_state')}\n"
        f"Tuning: {tuning_config.get('enabled', False)}"
        + (f"  settings={tuning_config}\n" if tuning_config.get("enabled", False) else "\n")
        + "=" * 60
        + "\n\n"
    )

    with open(path, "w", encoding="utf-8") as f:
        f.write(header + report_text)

    return path
