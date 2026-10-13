# config/ — run declarations (HyperPyYAML)

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

**This folder** sits at the very top: each YAML fully declares one run (data,
prompt, labels, model family, head, training schedule). Override any key from
the CLI, e.g. `--seed 2`, `--encoder_name ...`, `--head_type ...`.

## Files

- **`encoder_fc_classification.yaml`** — used by `main.py` (loaded via
  HyperPyYAML, normalized by `util/normalization.py`) to build and train the
  encoder + FC CM: XLS-R/Whisper → mean-pool → linear CLS head.
- **`modular_transformers_audio.yaml`** — used by `main.py` to build and train
  the Qwen2-Audio + LoRA CM, with a CLS or LM head (`--head_type`).
