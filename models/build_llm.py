"""Assemble LLM-backbone CMs (Qwen2-Audio) and their collator."""

import inspect
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers.trainer_pt_utils import LabelSmoother

from models import ModelComponents
from models.audio_encoders import build_audio_encoder
from models.llm_backbones import build_transformers_backbone
from models.output_heads import ClassificationHead, GenerativeHead
from models.utils import (
    get_output_field,
    infer_hidden_size,
    last_real_token_positions,
)
from util import build_target_formatter

IGNORE_TOKEN_ID = LabelSmoother.ignore_index


def build_llm_components(config: dict[str, Any]) -> ModelComponents:
    """Assemble a Transformers LLM backbone + classification/generative head."""
    target_formatter = build_target_formatter(
        config["target_formatter"],
        config["labels"],
    )
    backbone_bundle = build_transformers_backbone(config["model"])

    # Pre-flight check: for generative heads each verbalizer must be a single
    # token so that candidate scoring can do a direct logit lookup.
    if config["head"]["type"] == "generative":
        _check_verbalizer_tokenization(
            config["labels"], backbone_bundle.tokenizer
        )

    audio_encoder_bundle = build_audio_encoder(config.get("audio_encoder"))

    # Can't use both processor-backed audio AND a separate encoder.
    if audio_encoder_bundle.is_active:
        family = config["model"].get("family", "auto_audio_causal_lm")
        encoder_type = config.get("audio_encoder", {}).get("type", "none")
        raise ValueError(
            "Processor-backed audio LMs already handle audio internally. "
            f"Received model.family='{family}' with "
            f"audio_encoder.type='{encoder_type}'. "
            "Set `audio_encoder.type: none` for processor-backed families "
            "such as `qwen_audio`, `kimi_audio`, `glm4_voice`, "
            "or `native_audio_llm`."
        )

    base_model = backbone_bundle.model
    collator_kwargs = {
        "target_formatter": target_formatter,
        "system_message": config["prompt"].get(
            "system_message",
            "You are a helpful assistant.",
        ),
        "max_length": config["model"].get("max_length", 512),
        "head_type": config["head"]["type"],
        "prompt_style": config["prompt"].get("style", "plain"),
    }

    # The backbone processor (e.g. Qwen2-Audio) handles audio + text jointly.
    model_collator = TransformersAudioCollator(
        **collator_kwargs,
        sample_rate=config["dataset"]
        .get("audio", {})
        .get("sample_rate", 16000),
    )

    model, head_adapter = _build_head(
        base_model,
        config,
        tokenizer=backbone_bundle.tokenizer,
        target_formatter=target_formatter,
    )

    return ModelComponents(
        model=model,
        tokenizer=backbone_bundle.tokenizer,
        processor=backbone_bundle.processor,
        model_collator=model_collator,
        head_adapter=head_adapter,
        target_formatter=target_formatter,
        audio_processor=None,
    )


class TransformersAudioCollator:
    """Prepare batches for processor-backed audio causal LMs."""

    def __init__(
        self,
        target_formatter,
        system_message: str,
        max_length: int,
        sample_rate: int,
        head_type: str = "generative",
        prompt_style: str = "plain",
    ):
        """Initialize the processor-backed audio collator."""
        self.target_formatter = target_formatter
        self.system_message = system_message
        self.max_length = max_length
        self.sample_rate = sample_rate
        self.head_type = head_type
        self.prompt_style = prompt_style

    def _ordered_label_payload(self, tokenizer):
        """Label names, class ids, and first verbalizer token ids."""
        label_names = self.target_formatter.ordered_label_names()
        label_ids = []
        token_ids = []
        for label_name in label_names:
            verbalizer = self.target_formatter.primary_verbalizer(label_name)
            encoded = tokenizer.encode(verbalizer, add_special_tokens=False)
            if not encoded:
                raise ValueError(
                    f"Could not tokenize label verbalizer: {verbalizer}"
                )
            label_ids.append(self.target_formatter.label_to_id(label_name))
            token_ids.append(encoded[0])
        return (
            label_names,
            torch.tensor(label_ids, dtype=torch.long),
            torch.tensor(token_ids, dtype=torch.long),
        )

    def _plain_prompt(self, example) -> str:
        if self.prompt_style == "instruction":
            return (
                f"System: {self.system_message}\n"
                f"User: {example.prompt}\n"
                "Assistant:"
            )
        return f"{self.system_message}\n\n{example.prompt}".strip()

    @staticmethod
    def _user_content(processor, prompt_text: str):
        """Return user message content for the chat template."""
        if hasattr(processor, "audio_token"):
            return [{"type": "audio"}, {"type": "text", "text": prompt_text}]
        return prompt_text

    def _render_prompt(
        self,
        tokenizer,
        example,
        include_target: bool,
        target_text: Optional[str] = None,
        processor=None,
    ):
        messages = []
        if self.system_message:
            messages.append({"role": "system", "content": self.system_message})
        messages.append(
            {
                "role": "user",
                "content": self._user_content(processor, example.prompt),
            }
        )

        # Use the processor's apply_chat_template when it carries its own
        # template (e.g. Qwen2AudioProcessor), because the raw tokenizer
        # template doesn't handle list-typed message content.
        template_caller = (
            processor
            if (
                processor is not None
                and hasattr(processor, "audio_token")
                and hasattr(processor, "apply_chat_template")
            )
            else tokenizer
        )
        if self.prompt_style == "chat" and hasattr(
            template_caller, "apply_chat_template"
        ):
            prompt_text = template_caller.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
            if include_target:
                full_text = template_caller.apply_chat_template(
                    [*messages, {"role": "assistant", "content": target_text}],
                    tokenize=False,
                    add_generation_prompt=False,
                )
                return prompt_text, full_text
            return prompt_text, prompt_text

        prompt_text = self._plain_prompt(example)
        if include_target:
            return prompt_text, f"{prompt_text} {target_text}".strip()
        return prompt_text, prompt_text

    def _resolve_audio_payload(self, example):
        """Return audio waveform as a float32 numpy array."""
        import numpy as np

        if example.audio is None:
            raise ValueError(
                "Processor-backed audio models require `example.audio` to be "
                f"populated, but example id='{example.id}' received None. "
                "Set `dataset.audio.load_mode` so the waveform is loaded "
                "before collation."
            )
        audio = example.audio
        if torch.is_tensor(audio):
            audio = audio.detach().cpu().float().numpy()
        elif not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)
        else:
            audio = audio.astype(np.float32)
        if audio.ndim > 1:
            audio = audio.squeeze()
        if audio.ndim != 1:
            raise ValueError(
                "Processor-backed audio models expect each resolved audio "
                "payload to be a 1-D waveform array, but example "
                f"id='{example.id}' produced shape={audio.shape}."
            )
        return audio

    @staticmethod
    def _audio_kwarg_for(processor) -> str:
        """Return the audio keyword name for this processor's __call__."""
        try:
            params = inspect.signature(processor.__call__).parameters
        except (TypeError, ValueError):
            params = {}
        for name in ("audios", "audio", "speech"):
            if name in params:
                return name
        return "audios"  # best-guess fallback

    def _call_processor(self, processor, texts: list[str], audios: list[Any]):
        """Call the backbone processor with text and audio."""
        audio_key = self._audio_kwarg_for(processor)
        uses_modality_routing = hasattr(processor, "audio_token")
        if uses_modality_routing:
            # Do NOT pass truncation/max_length for audio-routing processors.
            # The processor expands each <|AUDIO|> placeholder into hundreds of
            # tokens (formula: (frames-2)//2+1), then tokenizes. Truncating the
            # expanded sequence clips audio tokens and triggers a mismatch check
            # inside _check_special_mm_tokens. The plain text portion alone is
            # never near max_length, so truncation here only causes breakage.
            # padding=True is still needed for different-length audio batches.
            common_kwargs = {
                "text_kwargs": {"padding": True, "return_tensors": "pt"},
            }
        else:
            common_kwargs = {
                "return_tensors": "pt",
                "padding": True,
                "truncation": True,
                "max_length": self.max_length,
            }
        attempts = [
            {
                "text": texts,
                audio_key: audios,
                "sampling_rate": self.sample_rate,
            },
            {"text": texts, audio_key: audios},
        ]
        last_error = None
        for kwargs in attempts:
            try:
                return processor(**common_kwargs, **kwargs)
            except (TypeError, ValueError) as exc:
                last_error = exc
        processor_name = type(processor).__name__
        raise TypeError(
            f"Processor '{processor_name}' did not accept audio keyword "
            f"'{audio_key}' (with or without sampling_rate). "
            f"Received {len(texts)} text and {len(audios)} audio items. "
            f"Last error: {last_error}"
        ) from last_error

    def prepare_batch(
        self,
        batch_examples,
        tokenizer,
        processor=None,
        mode: str = "train",
    ) -> dict[str, Any]:
        """Convert audio+text examples into processor model tensors."""
        if processor is None:
            raise ValueError(
                "Audio collators require a Hugging Face processor "
                "in `ModelComponents`."
            )

        include_target = self.head_type == "generative" and mode != "predict"
        prompt_texts: list[str] = []
        full_texts: list[str] = []
        target_texts: list[str] = []
        audio_payloads: list[Any] = []
        label_ids: list[int] = []

        for example in batch_examples:
            target_text = None
            if example.label is not None:
                target_text = self.target_formatter.format_target(example)
                label_ids.append(
                    self.target_formatter.label_to_id(str(example.label))
                )
            else:
                label_ids.append(-1)

            prompt_text, full_text = self._render_prompt(
                tokenizer,
                example,
                include_target=include_target and target_text is not None,
                target_text=target_text,
                processor=processor,
            )
            prompt_texts.append(prompt_text)
            full_texts.append(full_text)
            target_texts.append(target_text or "")
            audio_payloads.append(self._resolve_audio_payload(example))

        model_inputs = dict(
            self._call_processor(processor, full_texts, audio_payloads)
        )
        if "input_ids" not in model_inputs:
            raise ValueError(
                "Processor-backed causal LM batches must include `input_ids`,"
                f" but processor returned keys: {sorted(model_inputs.keys())}."
            )
        if "attention_mask" not in model_inputs:
            raise ValueError(
                "Processor-backed causal LM batches must include "
                "`attention_mask`, but processor returned keys: "
                f"{sorted(model_inputs.keys())}."
            )

        ordered_label_names, ordered_label_ids, candidate_token_ids = (
            self._ordered_label_payload(tokenizer)
        )
        batch = {
            "ids": [example.id for example in batch_examples],
            "examples": batch_examples,
            "prompts": prompt_texts,
            "target_texts": target_texts,
            "label_ids": torch.tensor(label_ids, dtype=torch.long),
            # `model_inputs` is the exact tensor payload forwarded into the
            # Transformers model. Keeping it explicit makes the multimodal
            # handoff easy to trace during debugging.
            "model_inputs": model_inputs,
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "ordered_label_names": ordered_label_names,
            "ordered_label_ids": ordered_label_ids,
            "candidate_token_ids": candidate_token_ids,
        }

        if include_target:
            # Generative training: mask every token up to the answer, so the
            # loss covers the answer and any template tokens after it.
            # answer_logit_positions is the token just before the answer; its
            # logits predict the answer word and are what gets scored.
            prompt_only = dict(
                self._call_processor(processor, prompt_texts, audio_payloads)
            )
            labels = model_inputs["input_ids"].clone()
            full_mask = model_inputs["attention_mask"]
            prompt_mask = prompt_only["attention_mask"]
            target_suffix_lens = (
                full_mask.sum(dim=1).long() - prompt_mask.sum(dim=1).long()
            )
            last_real = last_real_token_positions(full_mask)
            answer_logit_positions = (last_real - target_suffix_lens).clamp(
                min=0
            )
            for batch_index in range(labels.size(0)):
                target_start = (
                    int(answer_logit_positions[batch_index].item()) + 1
                )
                labels[batch_index, :target_start] = IGNORE_TOKEN_ID
            batch["labels"] = labels
            batch["answer_logit_positions"] = answer_logit_positions
        else:
            # No answer in the input: the last real token predicts it.
            batch["labels"] = None
            batch["answer_logit_positions"] = last_real_token_positions(
                model_inputs["attention_mask"]
            )

        return batch


class CausalLMClassificationModel(nn.Module):
    """Causal LM with a linear classifier over final hidden state."""

    def __init__(
        self,
        backbone,
        num_classes: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone = backbone
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(infer_hidden_size(backbone), num_classes)

        # Get backbone device robustly (works with wrapped models)
        # Try backbone.device first, fall back to getting device from parameters
        try:
            backbone_device = backbone.device
        except AttributeError:
            # For wrapped models without .device attribute,
            # get device from the first parameter
            backbone_device = next(backbone.parameters()).device

        self.classifier.to(backbone_device)

    @property
    def device(self):
        """Device of the backbone, else of the classifier."""
        try:
            return self.backbone.device
        except AttributeError:
            return self.classifier.weight.device

    def _transform_labels(self, labels, reference_tensor):
        """Transform labels to correct shape for cross-entropy loss."""
        if labels is None:
            return None

        # Ensure labels are on the same device as the classifier logits
        if labels.device != reference_tensor.device:
            labels = labels.to(reference_tensor.device)

        # Handle shape transformation
        if labels.dim() == 1:
            # Already correct shape (batch_size,)
            return labels.long()
        elif labels.dim() == 2:
            # Language modeling format (batch_size, seq_length)
            # For classification, use the first token's label as the class label
            return labels[:, 0].long()
        else:
            raise ValueError(
                "Expected labels to be 1D (batch_size,) or 2D "
                f"(batch_size, seq_length), got shape {tuple(labels.shape)}."
            )

    def forward(
        self,
        input_ids,
        attention_mask=None,
        labels=None,
        **kwargs,
    ):
        """Pool last real token hidden state, then classify."""
        # We ask the backbone for hidden states so the classifier head can pool
        # the final token representation instead of using next-token logits.
        backbone_kwargs = dict(kwargs)
        backbone_kwargs.pop("labels", None)
        outputs = self.backbone(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            use_cache=False,
            **backbone_kwargs,
        )
        hidden_states = get_output_field(outputs, "hidden_states")
        if hidden_states is None:
            raise ValueError(
                "Classification model expected the backbone to return "
                "`hidden_states` (output_hidden_states=True), but got None."
            )
        hidden_states = hidden_states[-1]
        sequence_attention_mask = get_output_field(
            outputs,
            "sequence_attention_mask",
            attention_mask,
        )
        if sequence_attention_mask is not None:
            # Handles left padding too (the Qwen2-Audio tokenizer pads left),
            # where `mask.sum() - 1` would point into the middle of the prompt.
            last_index = last_real_token_positions(sequence_attention_mask)
        else:
            last_index = torch.full(
                (hidden_states.size(0),),
                hidden_states.size(1) - 1,
                device=hidden_states.device,
                dtype=torch.long,
            )
        batch_index = torch.arange(
            hidden_states.size(0), device=hidden_states.device
        )
        pooled = hidden_states[batch_index, last_index, :]
        if (
            pooled.device != self.classifier.weight.device
            or pooled.dtype != self.classifier.weight.dtype
        ):
            self.classifier = self.classifier.to(
                device=pooled.device, dtype=pooled.dtype
            )
        logits = self.classifier(self.dropout(pooled))
        loss = None
        if labels is not None:
            # Transform labels to correct shape and ensure correct device
            transformed_labels = self._transform_labels(labels, logits)
            # `labels` here are class IDs (for example 0=real, 1=fake), so the
            # head uses standard cross-entropy over the classifier logits.
            loss = F.cross_entropy(logits, transformed_labels)
        return {
            "loss": loss,
            "logits": logits,
            "pooled_output": pooled,
            "backbone_outputs": outputs,
            "sequence_attention_mask": sequence_attention_mask,
        }


def _check_verbalizer_tokenization(
    labels_config: dict[str, Any], tokenizer
) -> None:
    """Raise if any verbalizer is not a single token."""
    failures = []
    for label_name, label_def in labels_config.items():
        for word in label_def.get("verbalizers", []):
            resolved_form = None
            for candidate in [word, " " + word]:
                ids = tokenizer.encode(candidate, add_special_tokens=False)
                if len(ids) == 1:
                    resolved_form = candidate
                    break
            if resolved_form is None:
                # Report the token breakdown of the bare word for clarity
                bare_ids = tokenizer.encode(word, add_special_tokens=False)
                bare_tokens = [tokenizer.decode([i]) for i in bare_ids]
                failures.append(
                    f"  label '{label_name}': verbalizer '{word}' → "
                    f"{bare_tokens} ({len(bare_ids)} tokens)"
                )

    if failures:
        raise ValueError(
            "Verbalizer vocabulary check failed — the following words are not "
            "single tokens in this tokenizer and cannot be scored reliably by "
            "the generative head:\n"
            + "\n".join(failures)
            + "\n\nPlease choose words that resolve to exactly one token.  "
            "You can find valid single-token words by checking:\n"
            "  tokenizer.encode('<your_word>', add_special_tokens=False)\n"
            "A single-element list means the word is usable.  "
            "If the bare form is multi-token, try adding a leading space "
            "(' <your_word>') — BPE tokenizers often encode common words as "
            "a single token in that form."
        )


def _build_head(model, config: dict[str, Any], tokenizer, target_formatter):
    """Build classification or generative head and adapter."""
    head_type = config["head"]["type"]
    if head_type == "classification":
        # Classification-head mode: consumes hidden states, predicts logits
        # directly, rather than teaching the LM to generate label text.
        return (
            CausalLMClassificationModel(
                backbone=model,
                num_classes=config["head"]["num_classes"],
                dropout=config["head"].get("dropout", 0.1),
            ),
            ClassificationHead(),
        )
    if head_type == "generative":
        # Generative mode: keeps the causal LM unchanged and supervises it
        # with the target formatter's verbalized label string (e.g. "fake").
        return (
            model,
            GenerativeHead(
                tokenizer=tokenizer,
                target_formatter=target_formatter,
                generation_config=config.get("generation", {}),
            ),
        )
    raise ValueError(
        f"Unsupported head type '{head_type}'. Expected 'classification' or "
        "'generative'."
    )
