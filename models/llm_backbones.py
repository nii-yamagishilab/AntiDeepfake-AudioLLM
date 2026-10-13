"""Transformers-based backbone loading for processor-backed audio LLMs."""

from dataclasses import dataclass
from typing import Any

from transformers import AutoProcessor, AutoTokenizer

from models.utils import infer_hidden_size
from util.normalization import PROCESSOR_AUDIO_FAMILIES

# PROCESSOR_AUDIO_FAMILIES lives in util.normalization (defined there) and is
# imported above so the schema and the model builder stay in sync.


@dataclass
class BackboneBundle:
    """Backbone with all components for model building."""

    model: Any
    tokenizer: Any
    processor: Any
    hidden_size: int


def build_transformers_backbone(model_config: dict[str, Any]) -> BackboneBundle:
    """Load a Hugging Face audio LLM and its processor."""
    model_name = model_config["model_name"]
    family = model_config.get("family", "auto_causal_lm")
    model_kwargs = dict(model_config.get("load_kwargs", {}))
    model_kwargs.setdefault(
        "trust_remote_code", model_config.get("trust_remote_code", True)
    )
    model_kwargs.setdefault(
        "device_map", model_config.get("device_map", "auto")
    )
    model_kwargs.setdefault("cache_dir", model_config.get("cache_dir"))

    if family not in PROCESSOR_AUDIO_FAMILIES:
        raise ValueError(
            f"Unsupported model family '{family}'. "
            f"Expected one of: {', '.join(sorted(PROCESSOR_AUDIO_FAMILIES))}."
        )

    processor_name = model_config.get("processor_name") or model_name
    processor_kwargs = dict(model_config.get("processor_kwargs", {}))
    processor_kwargs.setdefault(
        "trust_remote_code",
        model_config.get("trust_remote_code", True),
    )
    processor_kwargs.setdefault("cache_dir", model_config.get("cache_dir"))
    processor = AutoProcessor.from_pretrained(
        processor_name,
        **processor_kwargs,
    )
    # Many multimodal processors already expose a tokenizer. Loading the
    # processor first avoids AutoTokenizer failures on audio-only repos
    # that package tokenization inside the processor object.
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is None:
        tokenizer_name = model_config.get("tokenizer_name") or processor_name
        tokenizer_config = dict(model_config)
        tokenizer_config["tokenizer_name"] = tokenizer_name
        tokenizer = _load_tokenizer(tokenizer_config)
    elif getattr(tokenizer, "pad_token", None) is None:
        eos_token = getattr(tokenizer, "eos_token", None)
        if eos_token is None:
            raise ValueError(
                f"Processor tokenizer for '{processor_name}' has no "
                "`pad_token` and no `eos_token`, so padding cannot be "
                "configured automatically."
            )
        tokenizer.pad_token = eos_token

    model = _load_audio_model(model_name, model_kwargs)
    model = _maybe_apply_lora(model, model_config.get("lora"))
    return BackboneBundle(
        model=model,
        tokenizer=tokenizer,
        processor=processor,
        hidden_size=infer_hidden_size(model.config),
    )


def _maybe_apply_lora(model, lora_config: dict[str, Any]):
    """Apply LoRA when requested; keeps plain inference dependency-light."""
    if not lora_config or not lora_config.get("enabled", False):
        return model

    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

    if lora_config.get("prepare_for_kbit_training", False):
        model = prepare_model_for_kbit_training(
            model,
            use_gradient_checkpointing=lora_config.get(
                "use_gradient_checkpointing",
                False,
            ),
        )

    # Default to "all-linear" so PEFT auto-selects target layers when the user
    # doesn't specify them. An empty list would cause PEFT to raise an error.
    target_modules = lora_config.get("target_modules") or "all-linear"
    peft_config = LoraConfig(
        r=lora_config.get("r", 8),
        lora_alpha=lora_config.get("lora_alpha", 16),
        target_modules=target_modules,
        lora_dropout=lora_config.get("lora_dropout", 0.1),
        bias=lora_config.get("bias", "none"),
        task_type=lora_config.get("task_type", "CAUSAL_LM"),
    )
    return get_peft_model(model, peft_config, autocast_adapter_dtype=False)


def _load_tokenizer(model_config: dict[str, Any]):
    """Load a tokenizer from Hugging Face model hub."""
    tokenizer_kwargs = dict(model_config.get("tokenizer_kwargs", {}))
    tokenizer_kwargs.setdefault(
        "trust_remote_code",
        model_config.get("trust_remote_code", True),
    )
    tokenizer_kwargs.setdefault(
        "padding_side",
        model_config.get("padding_side", "right"),
    )
    tokenizer_kwargs.setdefault(
        "use_fast",
        model_config.get("use_fast_tokenizer", True),
    )
    tokenizer_kwargs.setdefault("cache_dir", model_config.get("cache_dir"))
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["tokenizer_name"],
        **tokenizer_kwargs,
    )
    if tokenizer.pad_token is None:
        if tokenizer.eos_token is None:
            raise ValueError(
                f"Tokenizer '{model_config['tokenizer_name']}' has no "
                "`pad_token` and no `eos_token`, so padding cannot be "
                "configured automatically."
            )
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _load_audio_model(model_name: str, model_kwargs: dict[str, Any]):
    """Load a native audio LLM for transformers 4.x/5.x."""
    import transformers

    # Strategy 1: transformers 4.x auto class
    auto_cls = getattr(transformers, "AutoModelForConditionalGeneration", None)
    if auto_cls is not None:
        try:
            return auto_cls.from_pretrained(model_name, **model_kwargs)
        except (ValueError, AttributeError):
            pass

    # Strategy 2: transformers 5.x — derive class name from AutoConfig
    from transformers import AutoConfig

    safe_kwargs = {
        k: model_kwargs[k]
        for k in ("trust_remote_code", "cache_dir")
        if k in model_kwargs
    }
    cfg = AutoConfig.from_pretrained(model_name, **safe_kwargs)
    base = type(cfg).__name__.replace("Config", "")
    for suffix in ("ForConditionalGeneration", "ForCausalLM", ""):
        cls = getattr(transformers, base + suffix, None)
        if cls is not None:
            return cls.from_pretrained(model_name, **model_kwargs)

    raise ValueError(
        f"Cannot find a model class for {type(cfg).__name__} in transformers "
        f"{transformers.__version__}. Tried: {base}ForConditionalGeneration, "
        f"{base}ForCausalLM, {base}."
    )
