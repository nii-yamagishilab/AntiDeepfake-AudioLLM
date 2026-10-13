"""Audio encoder for composed audio-text models."""

from dataclasses import dataclass
from typing import Any, Optional

import torch.nn as nn
from transformers import AutoFeatureExtractor, AutoModel, AutoProcessor

from models.utils import infer_hidden_size


@dataclass
class AudioEncoderBundle:
    """Audio encoder with processor, input key, and metadata."""

    encoder: nn.Module
    processor: Optional[Any]
    input_key: str
    output_dim: Optional[int]
    is_active: bool


class IdentityAudioEncoder(nn.Module):
    """Return the already-loaded audio tensor / features unchanged."""

    def forward(
        self,
        audio_features=None,
        input_values=None,
        input_features=None,
        **kwargs,
    ):
        features = audio_features
        if features is None:
            features = input_features
        if features is None:
            features = input_values
        if features is None:
            raise ValueError(
                "IdentityAudioEncoder expected one of `audio_features`, "
                "`input_features`, or `input_values`, but all were None."
            )
        return features


class TransformersAudioEncoder(nn.Module):
    """Wrap a HuggingFace encoder for uniform sequence output."""

    # Keys understood by seq2seq encoders (Whisper)
    SEQ2SEQ_KEYS = frozenset({"input_features", "attention_mask"})
    # Keys understood by direct waveform encoders (Wav2Vec2, WavLM, HuBERT …)
    DIRECT_KEYS = frozenset({"input_values", "attention_mask"})

    def __init__(self, model):
        super().__init__()
        self.model = model
        self.config = model.config

    @property
    def _is_seq2seq(self) -> bool:
        """True when model has both encoder and decoder."""
        return hasattr(self.model, "encoder") and hasattr(self.model, "decoder")

    def forward(self, **audio_inputs):
        """Run the encoder and return last_hidden_state [B, T, D]."""
        if self._is_seq2seq:
            encoder_model = self.model.encoder
            safe_keys = self.SEQ2SEQ_KEYS
        else:
            encoder_model = self.model
            safe_keys = self.DIRECT_KEYS

        filtered = {k: v for k, v in audio_inputs.items() if k in safe_keys}

        if not filtered:
            model_type = getattr(self.config, "model_type", "unknown")
            expected = "input_features" if self._is_seq2seq else "input_values"
            raise ValueError(
                f"TransformersAudioEncoder ({model_type}) received no "
                f"recognised audio key. Got: {list(audio_inputs.keys())}. "
                f"Expected '{expected}' (+ optional 'attention_mask')."
            )

        outputs = encoder_model(**filtered, return_dict=True)
        hidden_states = getattr(outputs, "last_hidden_state", None)
        if (
            hidden_states is None
            and isinstance(outputs, (tuple, list))
            and outputs
        ):
            hidden_states = outputs[0]
        if hidden_states is None:
            raise ValueError(
                "Audio encoder did not return a 'last_hidden_state' tensor. "
                "Check that the encoder config points to an encoder-only or "
                "seq2seq model whose encoder produces hidden states."
            )
        return hidden_states


#: All type strings that resolve to a Transformers-based encoder.
TRANSFORMERS_ENCODER_TYPES = {
    "hf_auto",
    "transformers_auto",
    "transformers_audio_encoder",
    # Explicit model families for clarity in configs
    "whisper",
    "wav2vec2",
    "wav2vec2_bert",
    "wavlm",
    "hubert",
    "data2vec_audio",
    "unispeech",
    "unispeech_sat",
    "sew",
    "sew_d",
}


def build_audio_encoder(config: dict[str, Any]) -> AudioEncoderBundle:
    """Build the optional audio encoder from config."""
    encoder_config = config or {}
    encoder_type = encoder_config.get("type", "none")

    if encoder_type == "none":
        return AudioEncoderBundle(
            encoder=IdentityAudioEncoder(),
            processor=None,
            input_key="audio_features",
            output_dim=None,
            is_active=False,
        )

    if encoder_type == "identity":
        return _build_identity_bundle(encoder_config)

    if encoder_type in TRANSFORMERS_ENCODER_TYPES:
        return _build_transformers_bundle(encoder_config)

    raise ValueError(
        f"Unsupported audio encoder type: '{encoder_type}'. "
        "Use one of: none, identity, "
        f"{', '.join(sorted(TRANSFORMERS_ENCODER_TYPES))}."
    )


def _freeze_module(module: nn.Module):
    for p in module.parameters():
        p.requires_grad = False


def _load_audio_processor(encoder_config: dict[str, Any]):
    name = encoder_config.get("processor_name") or encoder_config["model_name"]
    kwargs = dict(encoder_config.get("processor_kwargs", {}))
    kwargs.setdefault(
        "trust_remote_code", encoder_config.get("trust_remote_code", True)
    )
    kwargs.setdefault("cache_dir", encoder_config.get("cache_dir"))
    try:
        return AutoProcessor.from_pretrained(name, **kwargs)
    except Exception:
        return AutoFeatureExtractor.from_pretrained(name, **kwargs)


def _build_identity_bundle(
    encoder_config: dict[str, Any],
) -> AudioEncoderBundle:
    return AudioEncoderBundle(
        encoder=IdentityAudioEncoder(),
        processor=None,
        input_key=encoder_config.get("input_key", "audio_features"),
        output_dim=encoder_config.get("output_dim"),
        is_active=True,
    )


def _build_transformers_bundle(
    encoder_config: dict[str, Any],
) -> AudioEncoderBundle:
    model_kwargs = dict(encoder_config.get("load_kwargs", {}))
    model_kwargs.setdefault(
        "trust_remote_code", encoder_config.get("trust_remote_code", True)
    )
    model_kwargs.setdefault("cache_dir", encoder_config.get("cache_dir"))

    model = AutoModel.from_pretrained(
        encoder_config["model_name"], **model_kwargs
    )
    if encoder_config.get("freeze", True):
        _freeze_module(model)

    # Determine the correct input key automatically from the model architecture.
    # Seq2seq models (Whisper) → input_features; direct encoders → input_values.
    is_seq2seq = hasattr(model, "encoder") and hasattr(model, "decoder")
    auto_input_key = "input_features" if is_seq2seq else "input_values"
    input_key = encoder_config.get("input_key", auto_input_key)
    try:
        output_dim = infer_hidden_size(model.config)
    except ValueError as exc:
        model_name = encoder_config["model_name"]
        model_type = getattr(
            model.config, "model_type", type(model.config).__name__
        )
        raise ValueError(
            f"Could not infer output_dim for audio encoder '{model_name}' "
            f"(model_type='{model_type}'). Set `audio_encoder.output_dim` "
            "explicitly or use a config with a standard hidden-size field."
        ) from exc

    return AudioEncoderBundle(
        encoder=TransformersAudioEncoder(model),
        processor=_load_audio_processor(encoder_config),
        input_key=input_key,
        output_dim=output_dim,
        is_active=True,
    )
