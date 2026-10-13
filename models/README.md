# models/ — model assembly (encoders, backbones, heads, collators)

Part of the modular deepfake-detection countermeasure (CM) pipeline. The full
flow (shown in every folder's README so you can see where this folder sits):

```
 protocol CSV  (ID, Path, Label)                 [ config/*.yaml declares the run ]
        |
        |  dataio:  ProtocolDataset -> example dict { audio, prompt, label, meta }
        |           BatchCollator    -> model_collator.prepare_batch -> model_inputs
        v
 +----------------------------------+       +----------------------------------+
 | models: encoder + FC             |       | models: Qwen2-Audio + LoRA       |
 |  XLS-R / Whisper encoder         |       |  processor loads audio + prompt  |
 |  -> mean-pool -> linear (CLS)    |       |  -> CLS head  OR  LM head        |
 +----------------------------------+       +----------------------------------+
        |                                          |
        v                                          v
 models/output_heads:  ClassificationHead   /   GenerativeHead
        |
        |  train:   compute_loss
        |  predict: score_matrix + predicted label
        v
 main.py  ModularLLMBrain   (train / valid loop, checkpointing)
        |
        |   util:  config schema, prompt template, label verbalizer,
        |          metrics, checkpoint I/O, tensor moves
        v
 eval:  score_protocols | subband_eval | evaluate
        -> per-clip scores, EER, ROC-AUC, accuracy
```

**This folder** is the middle stage: from a normalized config it builds the
model, the matching collator (packs `dataio` example dicts into `model_inputs`),
and the head adapter. `build_model_components` dispatches on `model.family`.

## Files

- **`__init__.py`** — used by `main.py` and `eval/*` to assemble the model stack,
  dispatching on `model.family` (`build_model_components`, `ModelComponents`).
- **`build_encoder_fc.py`** — used by `build_model_components` (family
  `encoder_fc`) to build the encoder+FC model, its collator, and CLS head.
- **`build_llm.py`** — used by `build_model_components` (LLM families) to build
  the Qwen2-Audio model, its collator, and CLS/LM head.
- **`audio_encoders.py`** — used by both builders to construct the audio encoder
  (`build_audio_encoder`, `AudioEncoderBundle`).
- **`llm_backbones.py`** — used by `build_llm.py` to load the HF causal LM +
  processor with LoRA (`build_transformers_backbone`).
- **`output_heads.py`** — used by `main.py`/`eval` (via `head_adapter`) to
  compute loss and predictions (`ClassificationHead`, `GenerativeHead`).
- **`utils.py`** — used by the builders and heads for small tensor/config
  helpers (`last_real_token_positions`, `get_output_field`, `infer_hidden_size`).

## Editing guide

Where to edit for each experiment change, and each file's functions.

### `__init__.py`

```text
Edit here to:
  - add a new model family -> build_model_components
  - pick a model family -> config `model.family` (no code change)
```

### `build_encoder_fc.py`

```text
Edit here to:
  - swap XLS-R / Whisper -> config `encoder_name` (no code change)
  - change dropout -> config `head.dropout` (no code change)
  - change pooling (mean over time) or loss (cross-entropy) ->
    EncoderClassificationModel.forward
  - change the classifier layers -> EncoderClassificationModel.__init__

Functions:
  build_encoder_fc_components(cfg) -- assemble encoder+FC ModelComponents

EncoderClassificationModel:
  forward(labels, **audio) -- encode, mean-pool, classify (+loss)

AudioProcessingMixin:
  -- turn raw waveforms into processor tensors for audio collators

EncoderFCCollator(AudioProcessingMixin):
  prepare_batch(batch, ...) -- build text-free audio-classification batch
```

`EncoderClassificationModel`: `audio_encoder.freeze` (default true) controls whether the encoder
trains; the paper config sets it false to fine-tune.

`EncoderFCCollator`: reuses the audio loading/processing machinery from `AudioProcessingMixin`
(waveform resolution, processor fallbacks, padding) and prepares a text-free batch.

### `build_llm.py`

```text
Edit here to:
  - pick CLS or LM head -> config `head_type` (no code change)
  - swap the backbone -> config `llm_name` (no code change)
  - change CLS pooling (last real token) or CLS loss ->
    CausalLMClassificationModel.forward
  - change chat formatting -> TransformersAudioCollator._render_prompt
  - change which tokens the LM loss supervises ->
    TransformersAudioCollator.prepare_batch (prompt is masked)
  - add a new head type -> _build_head

Functions:
  build_llm_components(cfg) -- assemble backbone + head ModelComponents

TransformersAudioCollator:
  prepare_batch(batch, ..., processor) -- processor audio+text batch tensors

CausalLMClassificationModel:
  forward(...) -- LM final hidden state -> linear classifier logits/loss
  device -- backbone device
```

### `audio_encoders.py`

```text
Edit here to:
  - swap the encoder -> config `encoder_name` (no code change)
  - fine-tune or freeze -> config `audio_encoder.freeze` (no code change)
  - register a new encoder type -> TRANSFORMERS_ENCODER_TYPES
  - use other features (e.g. a mid layer) ->
    TransformersAudioEncoder.forward (now last_hidden_state)

Functions:
  build_audio_encoder(cfg) -- build AudioEncoderBundle from config

AudioEncoderBundle:
  -- dataclass: encoder + processor + input key + output dim + flag

IdentityAudioEncoder:
  forward(...) -- return preloaded audio tensor/features unchanged

TransformersAudioEncoder:
  forward(**inputs) -- run HF encoder for uniform sequence output
```

### `llm_backbones.py`

```text
Edit here to:
  - change LoRA rank / targets -> config `model.lora.*` (no code change)
  - use another PEFT method -> _maybe_apply_lora
  - load a new audio-LLM class -> _load_audio_model

Functions:
  build_transformers_backbone(cfg) -- load audio LLM + processor

BackboneBundle:
  -- dataclass: model + tokenizer + processor + hidden size
```

### `output_heads.py`

```text
Edit here to:
  - change how CLS logits become scores -> ClassificationHead.postprocess
  - change verbalizer scoring (Score_real / Score_fake) ->
    GenerativeHead._candidate_scores_from_logits
  - decode free text at test ->
    config `generation.generate_text` (no code change)

ClassificationHead:
  compute_forward / compute_loss / postprocess / predict
    -- classifier-head training and inference

GenerativeHead:
  compute_forward / compute_loss / postprocess / predict
    -- verbalized-label LM training and candidate scoring
```

### `utils.py`

```text
Functions:
  last_real_token_positions(mask) -- last non-pad token index per row
  get_output_field(outputs, key, default) -- read dict key or attribute
  infer_hidden_size(model_or_config) -- find hidden size across configs
```
