"""Head orchestrators: loss/postprocess/predict for each CM head type."""

from typing import Any, Optional

import torch

from models.utils import get_output_field


class ClassificationHead:
    """Adapter for classification-head training and inference."""

    def compute_forward(self, model, batch: dict[str, Any]):
        """Forward pass through the classifier."""
        forward_kwargs = dict(batch["model_inputs"])
        label_ids = batch.get("label_ids")
        if label_ids is not None and torch.all(label_ids >= 0):
            forward_kwargs["labels"] = label_ids
        return model(**forward_kwargs)

    def compute_loss(self, outputs, batch):
        """Extract loss from model outputs."""
        return outputs["loss"]

    def postprocess(self, outputs, batch):
        """Convert classifier logits to probabilities and class IDs."""
        logits = outputs["logits"]
        # The classifier head produces one logit per ordered label. Softmax
        # turns those logits into normalized class probabilities.
        # fp32 softmax: bf16 CLS logits quantize to ~1/256 steps (EER ties).
        probabilities = torch.softmax(logits.float(), dim=-1)
        predicted_indices = probabilities.argmax(dim=-1)
        ordered_label_ids = batch["ordered_label_ids"].to(logits.device)
        predicted_label_ids = ordered_label_ids[predicted_indices]
        return {
            "pred_label_ids": predicted_label_ids.detach().cpu(),
            "score_matrix": probabilities.detach().cpu(),
            "prediction_texts": [
                batch["ordered_label_names"][index]
                for index in predicted_indices.detach().cpu().tolist()
            ],
        }

    def predict(self, model, batch):
        """Run full prediction pipeline: forward + postprocess."""
        outputs = self.compute_forward(model, batch)
        return self.postprocess(outputs, batch)


class GenerativeHead:
    """Train/evaluate a causal LM by verbalizing labels as generated text."""

    def __init__(
        self,
        tokenizer,
        target_formatter,
        generation_config: Optional[dict[str, Any]] = None,
    ):
        """Initialize the generative head adapter."""
        self.tokenizer = tokenizer
        self.target_formatter = target_formatter
        self.generation_config = generation_config or {}

    def compute_forward(self, model, batch: dict[str, Any]):
        """Forward pass through the causal LM."""
        forward_kwargs = dict(batch["model_inputs"])
        forward_kwargs["labels"] = batch.get("labels")
        return model(**forward_kwargs)

    def compute_loss(self, outputs, batch):
        """Extract language modeling loss from model outputs."""
        return get_output_field(outputs, "loss")

    def postprocess(self, outputs, batch):
        """Convert LM logits to label scores and predicted text."""
        logits = get_output_field(outputs, "logits")
        pred_label_ids, score_matrix, prediction_texts = (
            self._candidate_scores_from_logits(
                logits,
                batch,
                outputs,
            )
        )
        return {
            "pred_label_ids": pred_label_ids,
            "score_matrix": score_matrix,
            "prediction_texts": prediction_texts,
        }

    def predict(self, model, batch):
        """Run full prediction pipeline: forward + postprocess."""
        outputs = self.compute_forward(model, batch)
        predictions = self.postprocess(outputs, batch)
        if not self.generation_config.get("generate_text", True):
            return predictions

        generation_kwargs = {
            "max_new_tokens": self.generation_config.get("max_new_tokens", 4),
            "do_sample": self.generation_config.get("do_sample", False),
        }
        if self.generation_config.get("num_beams"):
            generation_kwargs["num_beams"] = self.generation_config["num_beams"]

        generate_kwargs = {**batch["model_inputs"], **generation_kwargs}

        if hasattr(model, "generate"):
            generated_ids = model.generate(**generate_kwargs)
            generated_texts = self._decode_generated_sequences(
                generated_ids,
                batch["input_ids"],
            )
            # Parse free-form text through target formatter so "fake", "Fake",
            # or longer variants normalize to one label ID.
            parsed_predictions = [
                self.target_formatter.parse_prediction(text)
                for text in generated_texts
            ]
            parsed_ids = [
                self.target_formatter.prediction_to_label_id(item)
                for item in parsed_predictions
            ]
            for index, label_id in enumerate(parsed_ids):
                if label_id is not None:
                    predictions["pred_label_ids"][index] = label_id
            predictions["prediction_texts"] = generated_texts

        return predictions

    def _candidate_scores_from_logits(self, logits, batch, outputs):
        """Softmax over verbalizer-token logits at the answer position."""
        batch_index = torch.arange(logits.size(0), device=logits.device)
        position_offset = int(
            get_output_field(outputs, "logit_position_offset", 0)
        )
        positions = (
            batch["answer_logit_positions"].to(logits.device) + position_offset
        )
        # Take the logits that predict the answer word, keep only the
        # verbalizer tokens (e.g. "real", "fake"), and softmax over them.
        # The result is a [B, num_labels] probability matrix, same shape as
        # the classification head's, so EER is computed the same way.
        answer_logits = logits[batch_index, positions, :]
        candidate_token_ids = batch["candidate_token_ids"].to(logits.device)
        candidate_scores = answer_logits.index_select(-1, candidate_token_ids)
        # fp32 softmax: bf16 logits quantize to ~1/256 steps (EER ties).
        candidate_probs = torch.softmax(candidate_scores.float(), dim=-1)
        candidate_order = batch["ordered_label_ids"].to(logits.device)
        predicted_indices = candidate_probs.argmax(dim=-1)
        predicted_label_ids = candidate_order[predicted_indices]
        prediction_texts = [
            self.target_formatter.primary_verbalizer(
                self.target_formatter.id_to_label(label_id)
            )
            for label_id in predicted_label_ids.detach().cpu().tolist()
        ]
        return (
            predicted_label_ids.detach().cpu(),
            candidate_probs.detach().cpu(),
            prediction_texts,
        )

    def _decode_generated_sequences(self, generated_ids, input_ids):
        """Decode only the newly generated tokens after each prompt."""
        decoded = []
        for prompt_ids, full_ids in zip(input_ids, generated_ids):
            prompt_length = int(
                prompt_ids.ne(self.tokenizer.pad_token_id).sum().item()
            )
            continuation = full_ids[prompt_length:]
            decoded.append(
                self.tokenizer.decode(
                    continuation,
                    skip_special_tokens=True,
                    errors="ignore",
                ).strip()
            )
        return decoded
