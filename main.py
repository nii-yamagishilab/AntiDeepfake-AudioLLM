"""Training entrypoint and brain for modular deepfake detection."""

import logging
import os
import random
import sys
import time
from typing import Optional

import numpy as np
import speechbrain as sb
import torch
from hyperpyyaml import load_hyperpyyaml
from speechbrain.dataio.dataloader import LoopedLoader
from speechbrain.nnet.schedulers import WarmAndExpDecayLRSchedule
from speechbrain.utils.train_logger import FileTrainLogger, TensorboardLogger
from torch.utils.data import DataLoader
from tqdm.contrib import tqdm

logging.getLogger("httpx").setLevel(logging.WARNING)

from dataio import (
    BatchCollator,
    build_datasets,
    build_loader_kwargs,
)
from models import build_model_components
from util import (
    build_prompt_template,
    is_modular_config,
    move_batch_to_device,
    normalize_modular_config,
)
from util.checkpointing import (
    load_checkpoint_if_available,
    prune_checkpoints,
)
from util.predictions import PredictionAccumulator

_logger = logging.getLogger(__name__)


class ModularLLMBrain(sb.core.Brain):
    """SpeechBrain brain for modular deepfake detection."""

    def __init__(
        self, *args, config: dict, components, resume_state=None, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.config = config
        self.components = components
        self._resume_state = resume_state
        self._train_steps_logger = None
        self.train_stage_logger = FileTrainLogger(
            save_file=self.config["train_stage_log"],
        )
        self.lr_scheduler = WarmAndExpDecayLRSchedule(
            lr=self.config["training"]["learning_rate"],
            n_warmup_steps=self.config["training"].get("warmup_steps", 1000),
            total_steps=self.config["training"].get(
                "optimizer_step_limit", 30000
            ),
            decay_factor=self.config["training"].get("decay_factor", 0.01),
        )

    @property
    def train_steps_logger(self):
        """TensorBoard logger, created lazily on first use."""
        # Lazy so scoring-only brains (eval/) never write tfevents files.
        if self._train_steps_logger is None:
            self._train_steps_logger = TensorboardLogger(
                save_dir=self.config["train_steps_log"],
            )
        return self._train_steps_logger

    def init_optimizers(self):
        """Create AdamW over all model parameters."""
        self.optimizer = self.opt_class(
            self.modules.model.parameters(),
            lr=self.config["training"]["learning_rate"],
            weight_decay=self.config["training"]["weight_decay"],
        )

    def on_fit_start(self):
        """Restore optimizer state and step when resuming."""
        super().on_fit_start()
        if self._resume_state is None:
            return
        state = self._resume_state
        self._resume_state = None
        resumed_step = state.get("step", 0)
        try:
            self.optimizer.load_state_dict(state["optimizer_state_dict"])
        except Exception as exc:
            _logger.warning(f"Could not restore optimizer state: {exc}")
        self.optimizer_step = resumed_step
        _logger.info(
            f"Resumed from checkpoint - step: {resumed_step},"
            f" epoch: {state.get('epoch', '?')}"
        )

    def compute_forward(self, batch, stage):
        """Move batch to device and run the head's forward."""
        # Every collator returns a shared batch dict plus `model_inputs`,
        # the exact tensors passed as keyword arguments to the model.
        batch = move_batch_to_device(batch, self.device)
        outputs = self.components.head_adapter.compute_forward(
            self.modules.model,
            batch,
        )
        return outputs, batch

    def compute_objectives(self, forward_outputs, batch, stage):
        """Return loss; outside training also record predictions."""
        outputs, device_batch = forward_outputs
        loss = self.components.head_adapter.compute_loss(outputs, device_batch)
        if stage != sb.Stage.TRAIN:
            # Validation/test metrics are computed from normalized label IDs so
            # both generative and classification-head modes share one evaluator.
            postprocessed = self.components.head_adapter.postprocess(
                outputs,
                device_batch,
            )
            self.prediction_accumulator.add_batch(
                ids=device_batch["ids"],
                gold_label_ids=device_batch["label_ids"]
                .detach()
                .cpu()
                .tolist(),
                label_resolver=self.components.target_formatter.id_to_label,
                pred_label_ids=postprocessed["pred_label_ids"].tolist(),
                prediction_texts=postprocessed.get("prediction_texts"),
                target_texts=device_batch.get("target_texts"),
                score_matrix=postprocessed.get("score_matrix"),
            )
        return {"loss": loss}

    def fit(
        self,
        epoch_counter,
        train_set,
        valid_set=None,
        progressbar=None,
        train_loader_kwargs=None,
        valid_loader_kwargs=None,
    ):
        """Run training epochs, then a final validation pass."""
        train_loader_kwargs = train_loader_kwargs or {}
        valid_loader_kwargs = valid_loader_kwargs or {}

        if not isinstance(train_set, (DataLoader, LoopedLoader)):
            train_set = self.make_dataloader(
                train_set,
                stage=sb.Stage.TRAIN,
                **train_loader_kwargs,
            )
        if valid_set is not None and not isinstance(
            valid_set, (DataLoader, LoopedLoader)
        ):
            valid_set = self.make_dataloader(
                valid_set,
                stage=sb.Stage.VALID,
                ckpt_prefix=None,
                **valid_loader_kwargs,
            )

        self.on_fit_start()
        if progressbar is None:
            progressbar = not self.noprogressbar
        enable = progressbar and sb.utils.distributed.if_main_process()

        last_epoch = None
        for epoch in epoch_counter:
            last_epoch = epoch
            self._fit_train_customized(
                train_set=train_set,
                valid_set=valid_set,
                epoch=epoch,
                enable=enable,
                valid_step=self.config["training"]["valid_step"],
            )

        if valid_set is not None and last_epoch is not None:
            self._fit_valid_customized(valid_set=valid_set, epoch=last_epoch)

    def _fit_train_customized(
        self,
        train_set,
        valid_set,
        epoch,
        enable,
        valid_step=100,
    ):
        """Run one epoch, validating every `valid_step` batches."""
        logger = sb.core.logger
        self.on_stage_start(sb.Stage.TRAIN, epoch)
        self.modules.train()
        self.zero_grad()
        self.nonfinite_count = 0

        with tqdm(
            train_set,
            initial=self.step,
            dynamic_ncols=True,
            disable=not enable,
            colour=self.tqdm_barcolor["train"],
        ) as iterator:
            for batch in iterator:
                if self._optimizer_step_limit_exceeded:
                    logger.info("Train iteration limit exceeded")
                    break
                self.step += 1
                loss = self.fit_batch(batch)
                self.avg_train_loss = self.update_average(
                    loss, self.avg_train_loss
                )
                iterator.set_postfix(train_loss=self.avg_train_loss)

                if valid_set is not None and self.step % valid_step == 0:
                    self._fit_valid_customized(valid_set, epoch)

        self.zero_grad(set_to_none=True)
        self.on_stage_end(sb.Stage.TRAIN, self.avg_train_loss, epoch)
        self.avg_train_loss = 0.0

    def _fit_valid_customized(self, valid_set, epoch):
        """Run full validation pass; on_stage_end saves a checkpoint."""
        if valid_set is None:
            return
        self.on_stage_start(sb.Stage.VALID, epoch)
        self.modules.eval()
        avg_valid_loss = 0.0
        valid_steps = 0
        with torch.no_grad():
            for batch in valid_set:
                valid_steps += 1
                loss = self.evaluate_batch(batch, stage=sb.Stage.VALID)
                # Plain mean over valid batches. sb's update_average()
                # divides by self.step (global *train* step), which shrinks
                # the reported valid loss as training progresses and biases
                # best-valid selection toward late checkpoints.
                avg_valid_loss += (float(loss) - avg_valid_loss) / valid_steps
        self.valid_epoch_loss = avg_valid_loss if valid_steps else 0.0
        self.on_stage_end(sb.Stage.VALID, avg_valid_loss, epoch)
        self.modules.train()
        return

    def fit_batch(self, batch):
        """Forward/backward one batch; step optimizer after accumulation."""
        should_step = (self.step % self.grad_accumulation_factor) == 0
        with self.no_sync(not should_step):
            # training_ctx is sb's autocast for --precision (fp32 = no-op).
            with self.training_ctx:
                predictions = self.compute_forward(batch, sb.Stage.TRAIN)
                objectives = self.compute_objectives(
                    predictions, batch, sb.Stage.TRAIN
                )
            self.scaler.scale(
                objectives["loss"] / self.grad_accumulation_factor
            ).backward()
            objectives["total_loss"] = objectives["loss"].detach()

        if should_step:
            self.optimizers_step()
            self.on_fit_batch_end(objectives)
        return objectives["loss"].detach()

    def on_fit_batch_end(self, objectives):
        """Update learning rate and log every `log_interval` steps."""
        self.lr_scheduler(self.optimizer)
        log_interval = self.config["training"].get("log_interval", 100)
        if (
            self.optimizer_step % log_interval == 0
            and sb.utils.distributed.if_main_process()
        ):
            log_dct = {
                key: (
                    value.item() if isinstance(value, torch.Tensor) else value
                )
                for key, value in objectives.items()
            }
            log_dct["steps"] = self.optimizer_step
            log_dct["lr"] = self.optimizer.param_groups[0]["lr"]
            log_dct["avg_loss"] = self.avg_train_loss
            if hasattr(self, "time_last_log"):
                log_dct["run_time"] = time.time() - self.time_last_log
            self.time_last_log = time.time()
            self.train_steps_logger.log_stats(stats_meta=log_dct)

    def evaluate_batch(self, batch, stage):
        """Compute loss for one validation batch."""
        forward_outputs = self.compute_forward(batch, stage=stage)
        objectives = self.compute_objectives(
            forward_outputs, batch, stage=stage
        )
        return objectives["loss"].detach().cpu()

    def on_stage_start(self, stage, epoch):
        """Reset the prediction accumulator before valid/test."""
        if stage != sb.Stage.TRAIN:
            self.prediction_accumulator = PredictionAccumulator()
            if stage == sb.Stage.VALID:
                self.valid_epoch_loss = 0.0

    def _save_checkpoint(self, epoch):
        """Write save/<step>/model.ckpt plus tokenizer/processor files."""
        checkpoint_dir = os.path.join(
            self.config["save_folder"],
            str(self.optimizer_step),
        )
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, "model.ckpt")
        # Write to a temp file then os.replace() so model.ckpt is always
        # complete: a kill or disk error mid-write leaves the .tmp, not a
        # truncated model.ckpt, so the previous good checkpoint stays intact.
        tmp_path = checkpoint_path + ".tmp"
        torch.save(
            {
                "step": self.optimizer_step,
                "epoch": epoch,
                "model_state_dict": self.modules.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "config": self.config,
            },
            tmp_path,
        )
        os.replace(tmp_path, checkpoint_path)
        if hasattr(self.components.tokenizer, "save_pretrained"):
            self.components.tokenizer.save_pretrained(
                os.path.join(checkpoint_dir, "tokenizer"),
            )
        if hasattr(self.components.processor, "save_pretrained"):
            self.components.processor.save_pretrained(
                os.path.join(checkpoint_dir, "processor"),
            )
        if hasattr(self.components.audio_processor, "save_pretrained"):
            self.components.audio_processor.save_pretrained(
                os.path.join(checkpoint_dir, "audio_processor"),
            )

    def _save_valid_scores(self):
        """Write per-clip valid predictions to score/valid_<step>.csv."""
        score_folder = self.config.get("score_folder")
        if not score_folder:
            return
        os.makedirs(score_folder, exist_ok=True)
        csv_path = os.path.join(
            score_folder, f"valid_{self.optimizer_step}.csv"
        )
        self.prediction_accumulator.to_dataframe().to_csv(csv_path, index=False)

    def _prune_checkpoints(self):
        """Delete checkpoints except best-valid and latest."""
        prune_checkpoints(
            self.config["save_folder"],
            self.optimizer_step,
            self.config["train_stage_log"],
        )

    @staticmethod
    def _format_stats_pct(stats):
        """Format ratio metrics as percentage strings for logging."""
        _PCT_KEYS = {
            "accuracy",
            "precision_fake",
            "recall_fake",
            "precision_real",
            "recall_real",
            "f1",
            "roc_auc",
            "exact_match",
        }
        return {
            k: f"{v * 100:.2f}%"
            if k in _PCT_KEYS and isinstance(v, float)
            else v
            for k, v in stats.items()
        }

    def on_stage_end(self, stage, stage_loss, epoch=None):
        """Log metrics; after validation also checkpoint and prune."""
        if stage == sb.Stage.TRAIN:
            self.train_stats = {"loss": stage_loss}
            return

        # Test runs predict only, so they log metrics without a loss.
        stage_stats = {} if stage_loss is None else {"loss": stage_loss}
        stage_stats.update(self.prediction_accumulator.summarize())
        current_lr = (
            self.optimizer.param_groups[0]["lr"]
            if hasattr(self, "optimizer")
            else None
        )
        display_stats = self._format_stats_pct(stage_stats)
        self.train_stage_logger.log_stats(
            stats_meta={
                "epoch": epoch,
                "steps": self.optimizer_step,
                "lr": current_lr,
            },
            train_stats=None if stage != sb.Stage.TRAIN else self.train_stats,
            valid_stats=display_stats if stage == sb.Stage.VALID else None,
            test_stats=display_stats if stage == sb.Stage.TEST else None,
        )
        if stage == sb.Stage.VALID and sb.utils.distributed.if_main_process():
            tb_stats = {"steps": self.optimizer_step}
            tb_stats.update(
                {
                    k: (v.item() if isinstance(v, torch.Tensor) else v)
                    for k, v in stage_stats.items()
                    if isinstance(v, (int, float, torch.Tensor))
                }
            )
            self.train_steps_logger.log_stats(stats_meta=tb_stats)
            self._save_checkpoint(epoch)
            self._save_valid_scores()
            self._prune_checkpoints()

    def evaluate(
        self, dataset, loader_kwargs=None, output_path: Optional[str] = None
    ):
        """Predict a dataset; return metrics and per-clip DataFrame."""
        loader_kwargs = loader_kwargs or {}
        if not isinstance(dataset, (DataLoader, LoopedLoader)):
            dataset = self.make_dataloader(
                dataset,
                sb.Stage.TEST,
                ckpt_prefix=None,
                **loader_kwargs,
            )

        self.on_stage_start(sb.Stage.TEST, epoch=0)
        self.modules.eval()

        with torch.no_grad():
            for batch in tqdm(dataset):
                batch = move_batch_to_device(batch, self.device)
                # `predict` returns normalized label IDs plus optional free-form
                # text generations so downstream evaluation can report both
                # classification metrics and exact-match text accuracy.
                predictions = self.components.head_adapter.predict(
                    self.modules.model,
                    batch,
                )
                self.prediction_accumulator.add_batch(
                    ids=batch["ids"],
                    gold_label_ids=batch["label_ids"].detach().cpu().tolist(),
                    label_resolver=self.components.target_formatter.id_to_label,
                    pred_label_ids=predictions["pred_label_ids"].tolist(),
                    prediction_texts=predictions.get("prediction_texts"),
                    target_texts=batch.get("target_texts"),
                    score_matrix=predictions.get("score_matrix"),
                )

        metrics = self.prediction_accumulator.summarize()
        predictions_df = self.prediction_accumulator.to_dataframe()
        if output_path and sb.utils.distributed.if_main_process():
            predictions_df.to_csv(output_path, index=False)
        self.on_stage_end(sb.Stage.TEST, stage_loss=None, epoch=0)
        return metrics, predictions_df


def run_modular_experiment(hparams_file, run_opts, overrides, config):
    """Build and run a full training experiment."""
    normalized = config
    sb.create_experiment_directory(
        experiment_directory=normalized["output_folder"],
        hyperparams_to_save=hparams_file,
        overrides=overrides,
    )
    os.makedirs(normalized["save_folder"], exist_ok=True)
    os.makedirs(normalized["score_folder"], exist_ok=True)

    prompt_template = build_prompt_template(normalized["prompt"])
    components = build_model_components(normalized)
    datasets = build_datasets(normalized, prompt_template)
    resume_state = load_checkpoint_if_available(components.model, normalized)

    train_collator = BatchCollator(
        model_collator=components.model_collator,
        tokenizer=components.tokenizer,
        processor=components.processor,
        mode="train",
    )
    valid_collator = BatchCollator(
        model_collator=components.model_collator,
        tokenizer=components.tokenizer,
        processor=components.processor,
        mode="eval",
    )
    predict_collator = BatchCollator(
        model_collator=components.model_collator,
        tokenizer=components.tokenizer,
        processor=components.processor,
        mode="predict",
    )

    train_loader_kwargs = (
        build_loader_kwargs(
            normalized, train_collator, "train", datasets["train"]
        )
        if datasets["train"] is not None
        else None
    )
    valid_loader_kwargs = (
        build_loader_kwargs(
            normalized, valid_collator, "valid", datasets["valid"]
        )
        if datasets["valid"] is not None
        else None
    )
    test_loader_kwargs = (
        build_loader_kwargs(
            normalized, predict_collator, "test", datasets["test"]
        )
        if datasets["test"] is not None
        else None
    )

    brain = ModularLLMBrain(
        modules={"model": components.model},
        opt_class=torch.optim.AdamW,
        # sb.Brain reads optimizer_step_limit from run_opts/hparams only,
        # so pass the config value through (CLI override still wins).
        hparams={
            "optimizer_step_limit": normalized["training"].get(
                "optimizer_step_limit"
            ),
        },
        run_opts=run_opts,
        checkpointer=None,
        config=normalized,
        components=components,
        resume_state=resume_state,
    )

    start_epoch = (resume_state["epoch"] + 1) if resume_state else 1
    epochs = range(start_epoch, normalized["training"]["number_of_epochs"] + 1)
    if datasets["train"] is not None:
        # The same experiment object handles training and evaluation; only the
        # collator mode changes how prompts are supervised or generated.
        brain.fit(
            epochs,
            datasets["train"],
            datasets["valid"],
            train_loader_kwargs=train_loader_kwargs,
            valid_loader_kwargs=valid_loader_kwargs,
        )

    test_metrics = None
    predictions_df = None
    if (
        normalized["evaluation"].get("run_test_after_training", False)
        and datasets["test"] is not None
    ):
        test_metrics, predictions_df = brain.evaluate(
            datasets["test"],
            loader_kwargs=test_loader_kwargs,
            output_path=normalized["score_path"],
        )

    return {
        "brain": brain,
        "components": components,
        "datasets": datasets,
        "test_metrics": test_metrics,
        "predictions": predictions_df,
        "predict_collator": predict_collator,
        "test_loader_kwargs": test_loader_kwargs,
    }


def set_seed(seed):
    """Seed Python, NumPy, and Torch RNGs."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    _logger.info(f"Seeded RNGs with seed={seed}")


def load_hparams(argv):
    """Parse CLI arguments and load a HyperPyYAML experiment config."""
    hparams_file, run_opts, overrides = sb.parse_arguments(argv)
    sb.utils.distributed.ddp_init_group(run_opts)
    with open(hparams_file, encoding="utf-8") as handle:
        hparams = load_hyperpyyaml(handle, overrides)
    hparams.update(run_opts)
    return hparams_file, run_opts, overrides, hparams


def main(argv=None):
    """CLI entrypoint: load config, seed, then train/evaluate."""
    # Load the YAML config, fill in schema defaults, then build and run
    # the experiment (training, plus test scoring if configured).
    hparams_file, run_opts, overrides, hparams = load_hparams(
        argv or sys.argv[1:]
    )
    if hparams.get("seed") is not None:
        set_seed(int(hparams["seed"]))
    if not is_modular_config(hparams):
        raise ValueError(
            "Only modular configs are supported. Please use a config with "
            "`framework.type: modular`."
        )

    config = normalize_modular_config(hparams)
    run_modular_experiment(
        hparams_file=hparams_file,
        run_opts=run_opts,
        overrides=overrides,
        config=config,
    )


if __name__ == "__main__":
    main()
