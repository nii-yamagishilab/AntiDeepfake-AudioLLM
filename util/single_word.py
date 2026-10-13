"""Single-word label verbalizer and parser."""

import re
from typing import Any, Optional


class SingleWordFormatter:
    """Normalize short label verbalizers such as `real` and `fake`."""

    def __init__(
        self,
        labels: dict[str, dict[str, Any]],
        lowercase: bool = True,
        first_word_only: bool = True,
    ):
        self.labels = labels
        self.lowercase = lowercase
        self.first_word_only = first_word_only
        self._id_to_label = {
            label_cfg["id"]: label_name
            for label_name, label_cfg in labels.items()
        }
        self._verbalizer_to_label_id = {}
        for label_name, label_cfg in labels.items():
            label_id = label_cfg["id"]
            verbalizers = label_cfg.get("verbalizers", [label_name])
            normalized = [self._normalize_text(item) for item in verbalizers]
            self.labels[label_name]["verbalizers"] = normalized
            for verbalizer in normalized:
                self._verbalizer_to_label_id[verbalizer] = label_id

    def _normalize_text(self, value: str) -> str:
        """Strip, lowercase, and keep the first word."""
        text = value.strip()
        if self.lowercase:
            text = text.lower()
        if self.first_word_only:
            match = re.search(r"[A-Za-z0-9_+-]+", text)
            text = match.group(0) if match else text.split()[0] if text else ""
        return text

    def ordered_label_names(self) -> list[str]:
        """Return label names sorted by class id."""
        return [
            self._id_to_label[index]
            for index in sorted(self._id_to_label.keys())
        ]

    def primary_verbalizer(self, label_name: str) -> str:
        """Return the first verbalizer word for a label."""
        verbalizers = self.labels[label_name].get("verbalizers", [label_name])
        return verbalizers[0]

    def label_to_id(self, label_name: str) -> int:
        """Map a label name to its class id."""
        label_name = self._normalize_text(str(label_name))
        if label_name in self.labels:
            return self.labels[label_name]["id"]
        raise KeyError(f"Unknown label: {label_name}")

    def id_to_label(self, label_id: int) -> str:
        """Map a class id to its label name."""
        return self._id_to_label[int(label_id)]

    def format_target(self, example: Any) -> str:
        """Return the answer word the LM should generate."""
        label_name = example.label_text or example.label
        if label_name is None:
            raise ValueError("Target formatting requires a label.")
        label_name = self._normalize_text(str(label_name))
        return self.primary_verbalizer(label_name)

    def parse_prediction(self, prediction_text: str) -> dict[str, Any]:
        """Map generated text to a label, if it matches."""
        # Same normalization for gold verbalizers and model output so
        # free-form generations can map back to one class ID.
        normalized = self._normalize_text(prediction_text or "")
        label_id = self._verbalizer_to_label_id.get(normalized)
        label_name = (
            self._id_to_label.get(label_id) if label_id is not None else None
        )
        return {
            "raw_text": prediction_text,
            "normalized_text": normalized,
            "label_id": label_id,
            "label_name": label_name,
        }

    def prediction_to_label_id(
        self,
        parsed_prediction: dict[str, Any],
    ) -> Optional[int]:
        """Return the class id from a parsed prediction."""
        return parsed_prediction.get("label_id")


def build_target_formatter(
    formatter_config: dict[str, Any],
    labels: dict[str, dict[str, Any]],
):
    """Build the label formatter named in the config."""
    formatter_type = formatter_config.get("type", "single_word")
    if formatter_type != "single_word":
        raise ValueError(f"Unsupported target formatter: {formatter_type}")
    return SingleWordFormatter(
        labels=labels,
        lowercase=formatter_config.get("lowercase", True),
        first_word_only=formatter_config.get("first_word_only", True),
    )
