"""Shared helpers for the eval/ scripts.

Importing this module adds the repo root to ``sys.path`` so the core
packages (dataio, models, util, main) resolve from ``python eval/<script>``.

Functions:
  load_experiment(folder_or_ckpt, ...) -- build model + load checkpoint
  predict_frame(exp, df, dataset_cls, ...) -- run the model over a DataFrame
  parse_score_vector(value) -- score-vector cell to list or None
  assign_components(df) -- derive CompSpoof V2 component/class labels

Experiment:
  unload() -- move model off-GPU and free memory
"""

import ast
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import torch

# Lives in ``eval/``; add repo root so core packages are importable.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from hyperpyyaml import load_hyperpyyaml

from dataio import BatchCollator, build_loader_kwargs
from dataio.dataset import ProtocolDataset
from main import ModularLLMBrain
from models import build_model_components
from util import build_prompt_template, normalize_modular_config
from util.checkpointing import (
    best_valid_step_from_log,
    load_checkpoint_if_available,
)

__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"


class Experiment:
    """A loaded model bundle for one Log/exps experiment folder."""

    def __init__(
        self, name, config, components, collator, brain, real_id, fake_id
    ):
        self.name = name
        self.config = config
        self.components = components
        self.collator = collator
        self.brain = brain
        self.real_id = real_id
        self.fake_id = fake_id

    def unload(self):
        """Move model off-GPU and free memory."""
        self.components.model.cpu()
        del self.brain, self.components
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def load_experiment(
    exp_folder,
    device="cpu",
    batch_size=8,
    checkpoint="latest",
    data_root=None,
):
    """Build model from an exp folder or a model.ckpt file."""
    if os.path.isfile(exp_folder):
        # main.py stores the normalized training config inside model.ckpt,
        # so a lone checkpoint file is enough; `checkpoint` is ignored.
        state = torch.load(exp_folder, map_location="cpu", mmap=True)
        hparams = state["config"]
        name = os.path.basename(hparams["output_folder"].rstrip("/"))
        name = f"{name}_step{state['step']}"
        ckpt_path = exp_folder
        del state
    else:
        yaml_path = os.path.join(exp_folder, "hyperparams.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(
                f"hyperparams.yaml not found in {exp_folder}"
            )
        with open(yaml_path, encoding="utf-8") as handle:
            hparams = load_hyperpyyaml(handle)
        name = os.path.basename(exp_folder.rstrip("/"))
        ckpt_path = _resolve_checkpoint_path(exp_folder, checkpoint)

    config = normalize_modular_config(hparams)
    config["training"]["valid_batch_size"] = batch_size
    config["training"]["num_workers"] = 0
    # Scoring must not append to the training run's exp_log.txt.
    config["train_stage_log"] = os.devnull
    if data_root is not None:
        config["dataset"]["data_root"] = data_root
    if ckpt_path is not None:
        config["checkpoint_path"] = ckpt_path

    labels = config.get("labels", {})
    real_id = labels.get(config.get("word_for_real", "real"), {}).get("id", 0)
    fake_id = labels.get(config.get("word_for_fake", "fake"), {}).get("id", 1)

    components = build_model_components(config)
    components.model.to(device)
    load_checkpoint_if_available(components.model, config)
    components.model.eval()

    collator = BatchCollator(
        model_collator=components.model_collator,
        tokenizer=components.tokenizer,
        processor=components.processor,
        mode="predict",
    )
    brain = ModularLLMBrain(
        modules={"model": components.model},
        opt_class=torch.optim.AdamW,
        hparams={},
        run_opts={"device": device},
        checkpointer=None,
        config=config,
        components=components,
    )
    return Experiment(
        name=name,
        config=config,
        components=components,
        collator=collator,
        brain=brain,
        real_id=real_id,
        fake_id=fake_id,
    )


def predict_frame(exp, df, dataset_cls=ProtocolDataset, **dataset_kwargs):
    """Score protocol rows; return metrics and predictions."""
    tmp = tempfile.NamedTemporaryFile(suffix=".csv", delete=False, mode="w")
    df.to_csv(tmp.name, index=False)
    tmp.close()
    try:
        dataset = dataset_cls(
            data_path=tmp.name,
            dataset_config=exp.config["dataset"],
            prompt_template=build_prompt_template(exp.config["prompt"]),
            **dataset_kwargs,
        )
        loader_kwargs = build_loader_kwargs(
            exp.config, exp.collator, "test", dataset
        )
        loader_kwargs["pin_memory"] = False  # bf16 tensors can't be pinned
        return exp.brain.evaluate(
            dataset, loader_kwargs=loader_kwargs, output_path=None
        )
    finally:
        os.unlink(tmp.name)


def parse_score_vector(value):
    """Score-vector cell (list, string, or NaN) to list."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return None
    if isinstance(value, str):
        return ast.literal_eval(value)
    return value


def assign_components(df):
    """Derive CompSpoof V2 component labels from source columns."""
    out = df.copy()
    is_original = out["Path"].astype(str).str.contains("/original_audio/")

    def _component(method_column):
        values = out[method_column].astype(str).str.strip()
        return np.where(
            values.isin(["", "-", "nan", "None"]), "bonafide", "spoof"
        )

    speech = _component("speech_generation_mothed")
    event = _component("env_generation_mothed")
    out["speech"] = np.where(is_original, "bonafide", speech)
    out["event"] = np.where(is_original, "bonafide", event)
    out["Class"] = np.where(
        is_original,
        "original",
        pd.Series(speech, index=out.index)
        + "_"
        + pd.Series(event, index=out.index),
    )
    return out


# ── Private helpers ───────────────────────────────────────────────────────────


def _resolve_checkpoint_path(exp_folder, checkpoint):
    """Map checkpoint selector to a model.ckpt path or None."""
    if checkpoint in (None, "latest"):
        return None
    if checkpoint == "best_valid":
        step = best_valid_step_from_log(
            os.path.join(exp_folder, "exp_log.txt"),
            os.path.join(exp_folder, "save"),
        )
        if step is None:
            raise RuntimeError(
                f"no valid-loss checkpoint on disk in {exp_folder}"
            )
    else:
        step = int(checkpoint)
    path = os.path.join(exp_folder, "save", str(step), "model.ckpt")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"requested checkpoint not found: {path}")
    return path
