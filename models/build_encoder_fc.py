"""Assemble the encoder + FC classifier CMs, no LLM."""

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import ModelComponents
from models.audio_encoders import build_audio_encoder
from models.output_heads import ClassificationHead
from util import build_target_formatter


def build_encoder_fc_components(config: dict[str, Any]) -> ModelComponents:
    """Assemble XLS-R/Whisper: encoder, mean-pool, FC head."""
    target_formatter = build_target_formatter(
        config["target_formatter"], config["labels"]
    )
    audio_encoder_bundle = build_audio_encoder(config.get("audio_encoder"))
    model = EncoderClassificationModel(
        encoder=audio_encoder_bundle.encoder,
        hidden_size=audio_encoder_bundle.output_dim,
        num_classes=config["head"].get("num_classes", 2),
        dropout=config["head"].get("dropout", 0.1),
    )
    model_collator = EncoderFCCollator(
        target_formatter=target_formatter,
        audio_processor=audio_encoder_bundle.processor,
        audio_input_key=audio_encoder_bundle.input_key,
        sample_rate=config["dataset"]
        .get("audio", {})
        .get("sample_rate", 16000),
    )
    return ModelComponents(
        model=model,
        tokenizer=None,
        processor=None,
        model_collator=model_collator,
        head_adapter=ClassificationHead(),
        target_formatter=target_formatter,
        audio_processor=audio_encoder_bundle.processor,
    )


class EncoderClassificationModel(nn.Module):
    """Audio encoder + mean-pool + linear classification head."""

    def __init__(
        self,
        encoder,
        hidden_size: int,
        num_classes: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.encoder = encoder
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, labels=None, **audio_kwargs):
        """Encode audio, mean-pool over time, classify, add loss."""
        # Pass audio inputs (input_values / input_features + attention_mask)
        # to the encoder, which routes them for each architecture.
        hidden = self.encoder(**audio_kwargs)
        pooled = hidden.mean(dim=1)  # [B, D]
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels.long())
        return {"loss": loss, "logits": logits}


class AudioProcessingMixin:
    """Turn raw waveforms into processor tensors for audio collators."""

    audio_processor: Any
    audio_input_key: str
    sample_rate: int

    def _resolve_audio_payloads(self, batch_examples) -> list[Any]:
        """Extract audio from examples as float32 numpy arrays."""
        import numpy as np

        audio_payloads = []
        for example in batch_examples:
            if example.audio is None:
                raise ValueError(
                    "Audio-prefix models require `example.audio` to be "
                    f"populated, but example id='{example.id}' received None."
                    " Set `dataset.audio.load_mode` so waveforms or features"
                    " are loaded before collation."
                )
            audio = example.audio
            if torch.is_tensor(audio):
                audio = audio.detach().cpu().float().numpy()
            elif not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            else:
                audio = audio.astype(np.float32)
            # Squeeze to 1-D (some loaders return shape [1, N] or [N, 1])
            if audio.ndim > 1:
                audio = audio.squeeze()
            audio_payloads.append(audio)
        return audio_payloads

    def _call_audio_processor(self, audio_payloads: list[Any]):
        """Process audio payloads and return a tensor dict."""
        import numpy as np

        if not audio_payloads:
            raise ValueError(
                "Audio-prefix processor expected a non-empty batch of audio "
                "payloads, but received an empty list."
            )
        for index, audio in enumerate(audio_payloads):
            if not isinstance(audio, np.ndarray) or audio.ndim != 1:
                received_shape = getattr(audio, "shape", None)
                received_dtype = getattr(audio, "dtype", None)
                raise TypeError(
                    "Audio processors expect each sample to be a 1-D numpy "
                    f"waveform array, but sample {index} had "
                    f"type={type(audio).__name__}, shape={received_shape}, "
                    f"dtype={received_dtype}."
                )

        common_kwargs = {"return_tensors": "pt", "padding": True}
        if self.audio_input_key == "input_features":
            feature_extractor = getattr(
                self.audio_processor, "feature_extractor", None
            )
            common_kwargs["padding"] = "max_length"
            max_length = getattr(feature_extractor, "n_samples", None)
            if max_length is not None:
                common_kwargs["max_length"] = int(max_length)
        # Each tuple: (positional_args, keyword_args)
        batch_attempts = [
            ([], {"audio": audio_payloads, "sampling_rate": self.sample_rate}),
            ([], {"audios": audio_payloads, "sampling_rate": self.sample_rate}),
            ([audio_payloads], {"sampling_rate": self.sample_rate}),
            ([], {"speech": audio_payloads, "sampling_rate": self.sample_rate}),
            ([], {"audio": audio_payloads}),
            ([], {"audios": audio_payloads}),
            ([audio_payloads], {}),
            ([], {"speech": audio_payloads}),
        ]
        last_batch_error = None
        for args, kwargs in batch_attempts:
            try:
                return dict(
                    self.audio_processor(*args, **common_kwargs, **kwargs)
                )
            except (TypeError, ValueError) as exc:
                last_batch_error = exc

        # ── Per-sample fallback ──────────────────────────────────────────────
        import warnings

        warnings.warn(
            f"Batch audio processing failed "
            f"({type(last_batch_error).__name__}: {last_batch_error}). "
            "Attempting per-sample processing."
        )
        single_results = []
        single_error = None
        for audio in audio_payloads:
            per_sample_attempts = [
                ([], {"audio": audio, "sampling_rate": self.sample_rate}),
                ([audio], {"sampling_rate": self.sample_rate}),
                ([], {"audios": audio, "sampling_rate": self.sample_rate}),
                ([], {"speech": audio, "sampling_rate": self.sample_rate}),
                ([], {"audio": audio}),
                ([audio], {}),
                ([], {"audios": audio}),
                ([], {"speech": audio}),
            ]
            processed = None
            for args, kwargs in per_sample_attempts:
                try:
                    processed = dict(
                        self.audio_processor(*args, **common_kwargs, **kwargs)
                    )
                    break
                except (TypeError, ValueError) as e:
                    single_error = e
            if processed is None:
                break
            single_results.append(processed)

        if len(single_results) == len(audio_payloads):
            # Merge per-sample dicts: pad tensors to the longest in each key.
            merged: dict[str, Any] = {}
            for key in single_results[0]:
                tensors = [r[key] for r in single_results]
                if not all(torch.is_tensor(tensor) for tensor in tensors):
                    raise TypeError(
                        f"Processor key '{key}' returned non-tensor values "
                        "during per-sample audio fallback; expected tensors "
                        "for manual merge."
                    )
                if tensors[0].dim() >= 2:
                    max_len = max(t.shape[-1] for t in tensors)
                    padded = [
                        torch.nn.functional.pad(t, (0, max_len - t.shape[-1]))
                        for t in tensors
                    ]
                    merged[key] = torch.cat(padded, dim=0)
                else:
                    merged[key] = torch.cat(tensors, dim=0)
            return merged

        # ── Last resort: raw waveform stacking ──────────────────────────────
        warnings.warn(
            f"Per-sample audio processing also failed ({single_error}). "
            "Falling back to raw waveform stacking (no feature extraction)."
            " This will NOT work with Whisper (needs log-mel spectrograms)."
        )
        return self._stack_audio_without_processor(audio_payloads)

    def _stack_audio_without_processor(self, audio_payloads: list[Any]):
        """Zero-pad raw waveforms into a batch with attention mask."""
        tensors = [
            torch.as_tensor(item, dtype=torch.float32)
            for item in audio_payloads
        ]
        if not tensors:
            raise ValueError(
                "Audio-prefix batches require at least one audio example."
            )

        # No-processor path: assumes tensors are model-ready features or
        # waveforms. Pad time axis so the encoder gets a dense batch.
        max_length = max(tensor.size(0) for tensor in tensors)
        padded = []
        attention_mask = []
        for tensor in tensors:
            pad_amount = max_length - tensor.size(0)
            if pad_amount > 0:
                pad_shape = (pad_amount, *tuple(tensor.shape[1:]))
                tensor = torch.cat(
                    [tensor, torch.zeros(pad_shape, dtype=tensor.dtype)],
                    dim=0,
                )
            padded.append(tensor)
            attention_mask.append(
                torch.cat(
                    [
                        torch.ones(
                            tensor.size(0) - pad_amount, dtype=torch.long
                        ),
                        torch.zeros(pad_amount, dtype=torch.long),
                    ]
                )
            )

        return {
            self.audio_input_key: torch.stack(padded),
            "attention_mask": torch.stack(attention_mask),
        }

    def _prepare_audio_inputs(
        self, audio_payloads: list[Any]
    ) -> dict[str, Any]:
        """Run the audio processor, or pad raw waveforms."""
        if self.audio_processor is not None:
            return dict(self._call_audio_processor(audio_payloads))
        return self._stack_audio_without_processor(audio_payloads)


class EncoderFCCollator(AudioProcessingMixin):
    """Model collator for encoder + FC models."""

    def __init__(
        self,
        target_formatter,
        audio_processor,
        audio_input_key: str,
        sample_rate: int,
    ):
        self.target_formatter = target_formatter
        self.audio_processor = audio_processor
        self.audio_input_key = audio_input_key
        self.sample_rate = sample_rate

    def prepare_batch(
        self,
        batch_examples,
        tokenizer=None,
        processor=None,
        mode: str = "train",
    ) -> dict[str, Any]:
        """Convert examples into audio tensors and label ids."""
        audio_payloads = self._resolve_audio_payloads(batch_examples)
        audio_inputs = self._prepare_audio_inputs(audio_payloads)

        label_ids = [
            self.target_formatter.label_to_id(str(example.label))
            if example.label is not None
            else -1
            for example in batch_examples
        ]
        ordered_label_names = self.target_formatter.ordered_label_names()
        ordered_label_ids = torch.tensor(
            [
                self.target_formatter.label_to_id(name)
                for name in ordered_label_names
            ],
            dtype=torch.long,
        )

        return {
            "ids": [example.id for example in batch_examples],
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            # ClassificationHead.compute_forward spreads model_inputs as
            # **forward_kwargs and adds labels=label_ids before the model call.
            "model_inputs": dict(audio_inputs),
            "audio": audio_inputs,
            "ordered_label_names": ordered_label_names,
            "ordered_label_ids": ordered_label_ids,
            "candidate_token_ids": None,
        }
