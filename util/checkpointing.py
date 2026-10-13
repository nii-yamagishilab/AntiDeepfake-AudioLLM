"""Checkpoint I/O and recovery utilities."""

import logging
import os
import re
import shutil
import zipfile

import torch

_logger = logging.getLogger(__name__)

_VALID_LOSS_RE = re.compile(r"steps:\s*(\d+),.*?valid loss:\s*([0-9.eE+-]+)")


def checkpoint_ok(ckpt_path):
    """Return True if the path is a non-truncated zip archive."""
    if not os.path.isfile(ckpt_path):
        return False
    try:
        with zipfile.ZipFile(ckpt_path):
            return True
    except (zipfile.BadZipFile, OSError):
        return False


def best_valid_step_from_log(log_path, save_folder):
    """Find lowest-valid-loss step with checkpoint on disk."""
    if not os.path.isfile(log_path):
        return None
    best = None  # (loss, step)
    with open(log_path, encoding="utf-8") as handle:
        for line in handle:
            m = _VALID_LOSS_RE.search(line)
            if not m:
                continue
            step, loss = int(m.group(1)), float(m.group(2))
            if not checkpoint_ok(
                os.path.join(save_folder, str(step), "model.ckpt")
            ):
                continue
            if (
                best is None
                or loss < best[0]
                or (loss == best[0] and step > best[1])
            ):
                best = (loss, step)
    return best[1] if best else None


def load_checkpoint_if_available(model, config):
    """Load best available checkpoint into model; return state or None."""
    checkpoint_path = (
        config.get("checkpoint_path")
        or config.get("evaluation", {}).get("checkpoint_path")
        or _find_latest_checkpoint(config.get("save_folder", ""))
    )
    if not (checkpoint_path and os.path.isfile(checkpoint_path)):
        return None
    _logger.info(f"Loading checkpoint: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu")
    result = model.load_state_dict(state["model_state_dict"], strict=False)
    _log_checkpoint_recovery(result.missing_keys, result.unexpected_keys, model)
    return state


def _find_latest_checkpoint(save_folder):
    if not os.path.isdir(save_folder):
        return None
    best = None
    for name in os.listdir(save_folder):
        ckpt = os.path.join(save_folder, name, "model.ckpt")
        if name.isdigit() and checkpoint_ok(ckpt):
            step = int(name)
            if best is None or step > best[0]:
                best = (step, ckpt)
    return best[1] if best else None


def _log_checkpoint_recovery(missing_keys, unexpected_keys, model):
    component_prefixes = {
        "audio_encoder": "audio encoder",
        "projector": "projector",
        "backbone": "LLM backbone",
        "classifier": "classification head",
    }
    trainable_components = set()
    for name, param in model.named_parameters():
        if param.requires_grad:
            prefix = name.split(".")[0]
            # unwrap one level for composed wrappers (backbone.audio_encoder.*)
            if prefix == "backbone":
                sub = name.split(".")[1] if name.count(".") >= 1 else ""
                if sub in component_prefixes:
                    trainable_components.add(sub)
                else:
                    trainable_components.add(prefix)
            else:
                trainable_components.add(prefix)

    missing_prefixes = {k.split(".")[0] for k in missing_keys}
    for prefix, label in component_prefixes.items():
        if prefix in trainable_components:
            status = (
                "NOT recovered (missing keys)"
                if prefix in missing_prefixes
                else "recovered"
            )
            _logger.info(f"  {label}: {status}")

    if unexpected_keys:
        _logger.warning(
            f"Checkpoint had {len(unexpected_keys)} unexpected keys (ignored): "
            f"{unexpected_keys[:5]}{'...' if len(unexpected_keys) > 5 else ''}"
        )
    if missing_keys:
        _logger.warning(
            f"Checkpoint was missing {len(missing_keys)} keys (left at init): "
            f"{missing_keys[:5]}{'...' if len(missing_keys) > 5 else ''}"
        )


def prune_checkpoints(save_folder, current_step, train_stage_log):
    """Keep only best-valid and latest checkpoints."""
    latest = str(current_step)
    if not checkpoint_ok(os.path.join(save_folder, latest, "model.ckpt")):
        return
    keep = {latest}
    best = best_valid_step_from_log(train_stage_log, save_folder)
    if best is not None:
        keep.add(str(best))
    for name in os.listdir(save_folder):
        if name.isdigit() and name not in keep:
            shutil.rmtree(os.path.join(save_folder, name), ignore_errors=True)
