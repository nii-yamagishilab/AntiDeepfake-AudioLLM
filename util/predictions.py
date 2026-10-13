"""Prediction record accumulation and metric summarization."""

from dataclasses import dataclass
from typing import Any, Optional

import pandas as pd

from util.metrics import (
    compute_classification_metrics,
    compute_text_exact_match,
)


@dataclass
class PredictionRecord:
    """One prediction row as written to the score CSV."""

    sample_id: str
    gold_label: Optional[str]
    gold_label_id: Optional[int]
    prediction_label: Optional[str]
    prediction_label_id: Optional[int]
    prediction_text: Optional[str]
    target_text: Optional[str]
    score_vector: Optional[list[float]]


class PredictionAccumulator:
    """Collect normalized prediction records and derive task-level metrics."""

    def __init__(self):
        self.records: list[PredictionRecord] = []

    def _normalize_label_id(self, value):
        """Convert to int; negative ids mean unlabeled, so None."""
        if value is None:
            return None
        if hasattr(value, "item"):
            value = value.item()
        value = int(value)
        return value if value >= 0 else None

    def add_batch(
        self,
        ids,
        gold_label_ids,
        label_resolver,
        pred_label_ids,
        prediction_texts=None,
        target_texts=None,
        score_matrix=None,
    ):
        """Append one record per sample in the batch."""
        prediction_texts = prediction_texts or [None] * len(ids)
        target_texts = target_texts or [None] * len(ids)
        score_rows = (
            score_matrix.tolist()
            if score_matrix is not None and hasattr(score_matrix, "tolist")
            else score_matrix
            if score_matrix is not None
            else [None] * len(ids)
        )
        for index, sample_id in enumerate(ids):
            gold_label_id = self._normalize_label_id(gold_label_ids[index])
            pred_label_id = self._normalize_label_id(pred_label_ids[index])
            # Label-ID normalization ensures downstream evaluation works
            # regardless of whether predictions came from logits or decoding.
            self.records.append(
                PredictionRecord(
                    sample_id=sample_id,
                    gold_label=label_resolver(gold_label_id)
                    if gold_label_id is not None
                    else None,
                    gold_label_id=gold_label_id,
                    prediction_label=label_resolver(pred_label_id)
                    if pred_label_id is not None
                    else None,
                    prediction_label_id=pred_label_id,
                    prediction_text=prediction_texts[index]
                    if prediction_texts
                    else None,
                    target_text=target_texts[index] if target_texts else None,
                    score_vector=score_rows[index] if score_rows else None,
                )
            )

    def summarize(self) -> dict[str, Any]:
        """Compute metrics over all collected records."""
        y_true = [
            item.gold_label_id
            for item in self.records
            if item.gold_label_id is not None
        ]
        y_pred = [
            item.prediction_label_id
            for item in self.records
            if item.prediction_label_id is not None
        ]
        metrics = {}
        if y_true and len(y_true) == len(y_pred):
            score_matrix = [
                item.score_vector
                for item in self.records
                if item.score_vector is not None
            ]
            score_matrix = (
                score_matrix if len(score_matrix) == len(y_true) else None
            )
            metrics.update(
                compute_classification_metrics(y_true, y_pred, score_matrix)
            )
        target_texts = [
            item.target_text
            for item in self.records
            if item.target_text is not None
        ]
        prediction_texts = [
            item.prediction_text
            for item in self.records
            if item.prediction_text is not None
        ]
        if target_texts and len(target_texts) == len(prediction_texts):
            metrics["exact_match"] = compute_text_exact_match(
                target_texts, prediction_texts
            )
        return metrics

    def to_dataframe(self) -> pd.DataFrame:
        """Return collected records as a DataFrame."""
        return pd.DataFrame([record.__dict__ for record in self.records])
