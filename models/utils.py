"""Small shared tensor and config introspection helpers."""

from typing import Any, Optional

import torch


def last_real_token_positions(attention_mask: torch.Tensor) -> torch.Tensor:
    """Return last real token index per row."""
    seq_len = attention_mask.size(1)
    return (
        seq_len - 1 - attention_mask.long().flip(dims=[1]).argmax(dim=1)
    ).clamp(min=0)


def get_output_field(outputs: Any, key: str, default: Optional[Any] = None):
    """Read a field from dict or object attributes."""
    if isinstance(outputs, dict):
        return outputs.get(key, default)
    return getattr(outputs, key, default)


def infer_hidden_size(model_or_config: Any) -> int:
    """Infer hidden size from a backbone or encoder."""
    config = getattr(model_or_config, "config", model_or_config)
    attrs = (
        "hidden_size",
        "n_embd",
        "d_model",
        "output_hidden_size",
        "encoder_embed_dim",
        "dim",
    )
    # Check top-level config first, then fall back to nested sub-configs
    # (e.g. Qwen2AudioConfig nests the LM hidden_size inside text_config)
    configs_to_try = [config] + [
        getattr(config, sub, None)
        for sub in (
            "text_config",
            "audio_config",
            "language_config",
            "vision_config",
        )
    ]
    for cfg in configs_to_try:
        if cfg is None:
            continue
        for attr in attrs:
            value = getattr(cfg, attr, None)
            if value is not None:
                return int(value)
    config_type = type(config).__name__
    raise ValueError(
        f"Could not infer hidden size from config type '{config_type}'. "
        f"Tried attributes: {', '.join(attrs)}."
    )
