# util/ — shared helpers (config, prompts, labels, metrics, checkpoints)

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

**This folder** is the cross-cutting toolbox used by every stage: it turns a raw
config into a validated schema, renders prompts, maps labels ↔ verbalizers,
computes metrics, and handles checkpoints.

## Files

- **`normalization.py`** — used by `main.py` and `eval/*` to normalize and
  validate a config into the modular schema (also holds the model-family tables
  re-imported by `models/`).
- **`prompts_templates.py`** — used by `dataio` datasets and the model collators
  to render each example's prompt.
- **`single_word.py`** — used by the heads and collators in `models/` to map
  labels ↔ `real`/`fake` verbalizers and parse generated text.
- **`metrics.py`** — used by `eval/` and `predictions.py` to compute EER and
  classification metrics.
- **`predictions.py`** — used by `main.py`/`eval` to collect predictions into
  metrics and the score CSV.
- **`checkpointing.py`** — used by `main.py` and `eval/*` to save, load, and
  prune checkpoints.
- **`tensors.py`** — used by the `main.py` trainer to move a batch to the device.
- **`__init__.py`** — used by callers (`from util import ...`) to expose the
  config / prompt / label / tensor helpers.

## Editing guide

Where to edit for each experiment change, and each file's functions.

### `normalization.py`

```text
Edit here to:
  - change defaults for keys missing from YAML ->
    normalize_modular_config
  - move experiment outputs -> env EXP_OUTPUT_ROOT (no code change)
  - register a new audio-LLM family name -> PROCESSOR_AUDIO_FAMILIES

Functions:
  is_modular_config(hparams) -- True for a modular-framework config
  normalize_modular_config(hparams) -- fill config with schema defaults
```

### `prompts_templates.py`

```text
Edit here to:
  - change prompt wording -> config `prompt.system_message` /
    `template_body` / `template_suffix` (no code change)
  - use a CSV column in the prompt -> write {Column} (no code change)
  - change how placeholders are filled -> PromptTemplate.render

Functions:
  build_prompt_template(cfg) -- build PromptTemplate from prompt config

PromptTemplate:
  render(payload, acoustic_text) -- format body/suffix or legacy template
```

### `single_word.py`

```text
Edit here to:
  - change label words -> config `word_for_real` / `_fake` (no code change)
  - change how generated text maps to a label ->
    _normalize_text / parse_prediction
  - add a new formatter type -> build_target_formatter

Functions:
  build_target_formatter(cfg, labels) -- build a SingleWordFormatter

SingleWordFormatter:
  ordered_label_names() -- label names in fixed id order
  primary_verbalizer(name) -- primary surface word for a label
  label_to_id(name) / id_to_label(id) -- label <-> class-id map
  format_target(example) -- gold label verbalized as target text
  parse_prediction(text) -- map free-form text to label + id
  prediction_to_label_id(parsed) -- parsed prediction -> class id
```

### `metrics.py`

```text
Edit here to:
  - add a metric to valid/test logs -> compute_classification_metrics
  - change the EER computation (used by eval/) -> compute_eer

Functions:
  compute_eer(y_true, scores) -- Equal Error Rate + threshold
  compute_text_exact_match(targets, preds) -- exact-match rate
  compute_classification_metrics(y_true, y_pred, scores)
    -- accuracy, precision/recall, F1, ROC-AUC
```

### `predictions.py`

```text
Edit here to:
  - add a column to the score CSV -> PredictionRecord + add_batch
  - change which metrics are summarized -> summarize

PredictionRecord:
  -- dataclass: one prediction row (labels, text, score vector)

PredictionAccumulator:
  add_batch(...) -- append normalized records from a batch
  summarize() -- task-level metrics over collected records
  to_dataframe() -- collected records as a pandas DataFrame
```

### `checkpointing.py`

```text
Edit here to:
  - load a specific checkpoint -> config `checkpoint_path` (no code change)
  - change best-checkpoint rule (lowest valid loss) ->
    best_valid_step_from_log
  - change which checkpoints stay on disk -> prune_checkpoints

Functions:
  load_checkpoint_if_available(model, cfg) -- load best available ckpt
  best_valid_step_from_log(log, save) -- lowest-valid-loss step on disk
  checkpoint_ok(path) -- True if a valid (non-truncated) zip archive
  prune_checkpoints(save, step, log) -- keep only best-valid + latest
```

### `tensors.py`

```text
Functions:
  move_batch_to_device(payload, device) -- recursively move tensors
```
