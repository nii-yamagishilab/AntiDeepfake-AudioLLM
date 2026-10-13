"""CSV protocol dataset, batch collator, and DataLoader builders."""

from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
from torch.utils.data import Dataset, WeightedRandomSampler

from dataio.audio_loading import build_waveform_loader, to_absolute_path


class ProtocolDataset(Dataset):
    """CSV/record dataset that emits the shared task-example schema."""

    def __init__(
        self,
        data_path: str,
        dataset_config: dict[str, Any],
        prompt_template,
    ):
        self.data_path = data_path
        self.dataset_config = dataset_config
        self.prompt_template = prompt_template
        self.task_name = dataset_config.get("task_name", "deepfake_detection")
        self.id_key = dataset_config.get("id_key", "ID")
        self.audio_path_key = dataset_config.get("audio_path_key", "Path")
        self.label_key = dataset_config.get("label_key", "Label")
        self.prompt_key = dataset_config.get("prompt_key")
        self.label_text_key = dataset_config.get("label_text_key")
        self.metadata_keys = dataset_config.get("metadata_keys")
        self.data_root = dataset_config.get("data_root")
        self.opensmile_key = dataset_config.get("opensmile_key")
        self.feature_decimals = dataset_config.get("feature_decimals")
        self.waveform_loader = build_waveform_loader(dataset_config)
        self.rows = self._load_rows()

    def _load_rows(self) -> list[dict[str, Any]]:
        data_format = self.dataset_config.get("format", "csv")
        if data_format != "csv":
            raise ValueError(
                f"Unsupported dataset format: {data_format}. "
                "The current modular dataset path expects CSV protocols."
            )
        return pd.read_csv(self.data_path).to_dict(orient="records")

    def _metadata_from_row(self, row: dict[str, Any]) -> dict[str, Any]:
        reserved = {
            self.id_key,
            self.audio_path_key,
            self.label_key,
            self.prompt_key,
            self.label_text_key,
        }
        keys = self.metadata_keys or [key for key in row if key not in reserved]
        return {key: row.get(key) for key in keys if key in row}

    def _normalize_label(self, label: Any) -> Optional[str]:
        if label is None or (isinstance(label, float) and pd.isna(label)):
            return None
        return str(label).strip().lower()

    def _build_prompt(
        self, row: dict[str, Any], metadata: dict[str, Any]
    ) -> str:
        if self.prompt_key and row.get(self.prompt_key):
            return str(row[self.prompt_key]).strip()
        acoustic_text = ""
        if self.opensmile_key:
            val = row.get(self.opensmile_key, "")
            if val and not (isinstance(val, float) and pd.isna(val)):
                acoustic_text = str(val).strip()
                if self.feature_decimals is not None:
                    fmt = f"{{:.{self.feature_decimals}f}}"
                    acoustic_text = ",".join(
                        fmt.format(float(v))
                        for v in acoustic_text.split(",")
                        if v.strip()
                    )
        payload = dict(row)
        payload["metadata"] = metadata
        return self.prompt_template.render(payload, acoustic_text=acoustic_text)

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        """Return one CSV row as an example dict."""
        row = self.rows[index]
        metadata = self._metadata_from_row(row)
        audio_path = to_absolute_path(
            row.get(self.audio_path_key, ""), self.data_root
        )
        label_text = self._normalize_label(row.get(self.label_text_key))
        label = self._normalize_label(row.get(self.label_key))
        # Emit the shared example dict; the collator wraps it per model.
        return {
            "id": str(row.get(self.id_key, index)),
            "audio_path": audio_path,
            "audio": self.waveform_loader.load(audio_path)
            if audio_path
            else None,
            "task": self.task_name,
            "prompt": self._build_prompt(row, metadata),
            "label": label,
            "label_text": label_text or label,
            "metadata": metadata,
        }

    def label_names(self) -> list[Optional[str]]:
        """Return normalized label per row for sampler weights."""
        return [
            self._normalize_label(row.get(self.label_key)) for row in self.rows
        ]


class BatchCollator:
    """Wrap example dicts and delegate packing to the model collator."""

    def __init__(
        self,
        model_collator,
        tokenizer,
        processor=None,
        mode: str = "train",
    ):
        self.model_collator = model_collator
        self.tokenizer = tokenizer
        self.processor = processor
        self.mode = mode

    def __call__(self, batch_examples):
        # SimpleNamespace gives the attribute access that model collators use.
        examples = [SimpleNamespace(**item) for item in batch_examples]
        return self.model_collator.prepare_batch(
            examples,
            tokenizer=self.tokenizer,
            processor=self.processor,
            mode=self.mode,
        )


def build_weighted_sampler(dataset: ProtocolDataset):
    """Create inverse-frequency sampling weights from label names."""
    label_names = dataset.label_names()
    label_counts = {}
    for label_name in label_names:
        if label_name is None:
            continue
        label_counts[label_name] = label_counts.get(label_name, 0) + 1
    if not label_counts:
        return None
    weights = [
        1.0 / label_counts.get(label_name, 1.0) for label_name in label_names
    ]
    return WeightedRandomSampler(
        weights, num_samples=len(weights), replacement=True
    )


def build_loader_kwargs(config, collator, split: str, dataset):
    """Build DataLoader kwargs dict for the given split."""
    training_cfg = config["training"]
    batch_size = (
        training_cfg["batch_size"]
        if split == "train"
        else training_cfg.get("valid_batch_size", 1)
    )
    kwargs = {
        "batch_size": batch_size,
        "num_workers": training_cfg.get("num_workers", 0),
        "pin_memory": True,
        "collate_fn": collator,
    }
    if split == "train":
        sampler = None
        if training_cfg.get("balanced_sampling", True):
            sampler = build_weighted_sampler(dataset)
        if sampler is not None:
            kwargs["sampler"] = sampler
        else:
            kwargs["shuffle"] = True
    return kwargs


def build_datasets(config, prompt_template):
    """Build the configured train/valid/test datasets."""
    dataset_cfg = config["dataset"]
    bundle = {}
    for split in ("train", "valid", "test"):
        split_key = f"{split}_path"
        if dataset_cfg.get(split_key):
            bundle[split] = ProtocolDataset(
                data_path=dataset_cfg[split_key],
                dataset_config=dataset_cfg,
                prompt_template=prompt_template,
            )
        else:
            bundle[split] = None
    return bundle
