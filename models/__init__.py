"""Model assembly entrypoint: ModelComponents + family dispatch."""

from dataclasses import dataclass
from typing import Any


@dataclass
class ModelComponents:
    """All assembled components for training and inference."""

    model: Any
    tokenizer: Any
    processor: Any
    model_collator: Any
    head_adapter: Any
    target_formatter: Any
    audio_processor: Any


def build_model_components(config: dict[str, Any]) -> ModelComponents:
    """Assemble the model stack, dispatching by model family."""
    family = config.get("model", {}).get("family", "auto_causal_lm")
    # Local imports keep the two family builders from importing each other.
    if family == "encoder_fc":
        from models.build_encoder_fc import build_encoder_fc_components

        return build_encoder_fc_components(config)
    from models.build_llm import build_llm_components

    return build_llm_components(config)


__all__ = ["ModelComponents", "build_model_components"]
