"""Modular-config detection and default normalization."""

import copy
import os
from typing import Any

# Model families with an integrated audio processor (native speech-LLMs) that
# loads BOTH audio and text jointly. Defined here; re-imported by models/.
PROCESSOR_AUDIO_FAMILIES = {
    "auto_audio_causal_lm",
    "transformers_audio_causal_lm",
    "audio_causal_lm",
    # Named aliases so configs are self-documenting
    "qwen_audio",  # Qwen-Audio-Chat, Qwen2-Audio-Instruct
    "kimi_audio",  # Kimi-Audio (moonshot-ai)
    "glm4_voice",  # GLM-4-Voice
    "native_audio_llm",  # Generic alias for any native speech-LLM
}

# Families that have no LLM backbone and therefore do not require model_name.
# The audio encoder feeds an FC head directly, with no LLM and no projector.
ENCODER_ONLY_FAMILIES = {"encoder_fc"}


def is_modular_config(hparams: dict[str, Any]) -> bool:
    """Return True if config sets framework type modular."""
    framework_cfg = hparams.get("framework", {})
    if (
        isinstance(framework_cfg, dict)
        and framework_cfg.get("type") == "modular"
    ):
        return True
    return hparams.get("framework_type") == "modular"


def normalize_modular_config(hparams: dict[str, Any]) -> dict[str, Any]:
    """Fill config with defaults for a normalized schema."""
    config = copy.deepcopy(hparams)
    config.setdefault("framework", {})
    config.setdefault("prompt", {})
    config.setdefault("target_formatter", {"type": "single_word"})
    config.setdefault(
        "generation", {"max_new_tokens": 4, "generate_text": True}
    )
    config.setdefault("training", {})
    config.setdefault("evaluation", {})
    config.setdefault("dataset", {})
    config.setdefault("head", {"type": "generative"})
    config.setdefault("labels", {})
    config.setdefault("model", {})
    config.setdefault("audio_encoder", {})
    config.setdefault("projector", {})

    base_path = config.get("base_path", ".")
    exp_name = config.get("exp_name", "exp")
    head_type = config["head"].get("type", "generative")
    config["output_folder"] = config.get(
        "output_folder",
        _default_output_folder(base_path, exp_name, head_type),
    )
    config["save_folder"] = config.get(
        "save_folder",
        os.path.join(config["output_folder"], "save"),
    )
    config["score_folder"] = config.get(
        "score_folder",
        os.path.join(config["output_folder"], "score"),
    )
    config["score_path"] = config.get(
        "score_path",
        os.path.join(config["score_folder"], "test.csv"),
    )
    config["train_stage_log"] = config.get(
        "train_stage_log",
        os.path.join(config["output_folder"], "exp_log.txt"),
    )
    config["train_steps_log"] = config.get(
        "train_steps_log",
        os.path.join(config["output_folder"], "train_steps_log"),
    )

    dataset_cfg = config["dataset"]
    dataset_cfg.setdefault("format", "csv")
    dataset_cfg.setdefault(
        "task_name", config.get("task", {}).get("name", "deepfake_detection")
    )
    dataset_cfg.setdefault("data_root", config.get("data_folder"))
    dataset_cfg.setdefault("train_path", config.get("train_csv"))
    dataset_cfg.setdefault("valid_path", config.get("valid_csv"))
    dataset_cfg.setdefault("test_path", config.get("test_csv"))
    dataset_cfg.setdefault("id_key", "ID")
    dataset_cfg.setdefault("audio_path_key", "Path")
    dataset_cfg.setdefault("label_key", "Label")
    dataset_cfg.setdefault("audio", {"load_mode": "none"})
    dataset_cfg.setdefault("opensmile_key", None)

    prompt_cfg = config["prompt"]
    if not prompt_cfg.get("template_body") and not prompt_cfg.get("template"):
        prompt_cfg["template"] = (
            "Determine whether the following speech audio is real or fake. "
            "Answer with only one word: real or fake."
        )
    prompt_cfg.setdefault("template_body", "")
    prompt_cfg.setdefault("template_suffix", "")
    prompt_cfg.setdefault("style", "plain")
    prompt_cfg.setdefault("system_message", "You are a helpful assistant.")

    model_cfg = config["model"]
    model_cfg.setdefault("family", "auto_causal_lm")
    model_cfg.setdefault("max_length", config.get("max_len", 512))
    if "model_name" not in model_cfg:
        model_cfg["model_name"] = config.get("model_name")
    if (
        model_cfg["model_name"] is None
        and model_cfg.get("family") not in ENCODER_ONLY_FAMILIES
    ):
        raise ValueError("Modular config requires model.model_name.")

    audio_encoder_cfg = config["audio_encoder"]
    audio_encoder_cfg.setdefault("type", "none")

    projector_cfg = config["projector"]
    user_projector_cfg = hparams.get("projector", {})
    # ENCODER_ONLY_FAMILIES use the audio encoder directly into a FC head; they
    # have no LLM to project into, so the auto-linear-projector must not fire.
    if (
        audio_encoder_cfg.get("type", "none") != "none"
        and model_cfg.get("family") not in ENCODER_ONLY_FAMILIES
    ):
        if (
            projector_cfg.get("type", "none") == "none"
            and "type" not in user_projector_cfg
        ):
            projector_cfg["type"] = "linear"
        projector_cfg.setdefault("prefix_length", 8)
    else:
        projector_cfg.setdefault("type", "none")

    if (
        model_cfg["family"] in PROCESSOR_AUDIO_FAMILIES
        or audio_encoder_cfg.get("type", "none") != "none"
    ):
        if dataset_cfg["audio"].get("load_mode", "none") in ("none", None):
            dataset_cfg["audio"]["load_mode"] = "waveform"
        dataset_cfg["audio"].setdefault("sample_rate", 16000)

    training_cfg = config["training"]
    training_cfg.setdefault("batch_size", config.get("batch_size", 2))
    training_cfg.setdefault(
        "valid_batch_size", config.get("valid_batch_size", 1)
    )
    training_cfg.setdefault("num_workers", config.get("num_workers", 0))
    training_cfg.setdefault("learning_rate", config.get("lr", 4e-5))
    training_cfg.setdefault("weight_decay", config.get("weight_decay", 0.01))
    training_cfg.setdefault(
        "number_of_epochs", config.get("number_of_epochs", 1)
    )
    training_cfg.setdefault("log_interval", config.get("log_interval", 100))
    training_cfg.setdefault("valid_step", config.get("valid_step", 1000))
    training_cfg.setdefault("balanced_sampling", True)

    evaluation_cfg = config["evaluation"]
    evaluation_cfg.setdefault(
        "run_test_after_training", bool(dataset_cfg.get("test_path"))
    )

    if config["head"]["type"] == "classification":
        config["head"].setdefault("num_classes", len(config["labels"]))

    return config


def _default_output_folder(
    base_path: str, exp_name: str, head_type: str
) -> str:
    # EXP_OUTPUT_ROOT overrides the default <base_path>/Log/exps root when set
    # (used to redirect reproduction runs); folder name is unchanged.
    root = os.environ.get("EXP_OUTPUT_ROOT") or os.path.join(
        base_path, "Log", "exps"
    )
    return os.path.join(root, f"modular_{head_type}_{exp_name}")
