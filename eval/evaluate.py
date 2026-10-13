"""Evaluate prediction CSVs from main.py or score_protocols.py.

Functions:
  main(file, subsets) -- print metrics table from a CSV
  subset_metrics(df) -- detection metrics for one group of rows
  get_prefix(sample_id) -- dataset prefix from a `name-001` ID
  parse_args() -- CLI: prediction file + subset prefixes
"""

import argparse

import numpy as np
import pandas as pd
from eval_utils import parse_score_vector

from util.metrics import compute_classification_metrics, compute_eer

COLUMNS = [
    "roc_auc",
    "eer",
    "eer_threshold",
    "accuracy",
    "precision_fake",
    "recall_fake",
    "precision_real",
    "recall_real",
    "f1",
]


def main(filename, subset_list=None):
    """Print pooled and per-prefix metrics for a prediction CSV."""
    df = pd.read_csv(filename)
    # score_protocols.py CSVs key clips on `ID` instead of `sample_id`.
    if "sample_id" not in df.columns and "ID" in df.columns:
        df = df.rename(columns={"ID": "sample_id"})
    missing = {"sample_id", "gold_label_id", "prediction_label_id"} - set(
        df.columns
    )
    if missing:
        raise ValueError(
            "Prediction CSV is missing required columns: "
            + ", ".join(sorted(missing))
        )

    prefixes = df["sample_id"].apply(get_prefix)
    subsets = {"Pooled": df}
    for prefix in subset_list or []:
        subsets[prefix] = df[prefixes == prefix]
    results = {
        name: subset_metrics(sub) for name, sub in subsets.items() if len(sub)
    }
    if not results:
        print("No valid data to display.")
        return

    table = pd.DataFrame.from_dict(results, orient="index")
    table = table[[col for col in COLUMNS if col in table.columns]]
    # Ratio metrics as xx.xx%; eer_threshold stays a raw float.
    for col in table.columns:
        scale, fmt = (1, ".4f") if col == "eer_threshold" else (100, ".2f")
        table[col] = table[col].apply(
            lambda v: f"{v * scale:{fmt}}" if pd.notna(v) else "-"
        )

    print("\n===== METRICS SUMMARY =====")
    print("EER threshold: determined automatically from ROC curve.\n")
    print(table.to_string())


def subset_metrics(df):
    """Classification metrics plus EER for one group of rows."""
    y_true = df["gold_label_id"].astype(int).to_numpy()
    y_pred = df["prediction_label_id"].astype(int).to_numpy()
    score_matrix = None
    if "score_vector" in df.columns:
        scores = df["score_vector"].apply(parse_score_vector)
        if scores.notna().all():
            score_matrix = np.asarray(scores.tolist())

    metrics = compute_classification_metrics(y_true, y_pred, score_matrix)
    if score_matrix is not None and len(np.unique(y_true)) == 2:
        eer, eer_threshold = compute_eer(y_true, score_matrix[:, 1])
        metrics["eer"] = eer
        metrics["eer_threshold"] = eer_threshold
    return metrics


def get_prefix(sample_id):
    """Extract the dataset prefix from a sample ID like `asvspoof-001`."""
    parts = str(sample_id).split("-")
    return parts[0] if len(parts) > 1 else None


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "prediction_file", help="Path to a modular prediction CSV."
    )
    parser.add_argument(
        "subsets",
        nargs="*",
        help="Optional sample-ID prefixes to report separately.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args.prediction_file, args.subsets)
