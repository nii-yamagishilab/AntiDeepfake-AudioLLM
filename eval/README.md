# eval/ — scoring trained checkpoints

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

**This folder** is the last stage: rebuild a trained model from its experiment
folder (reusing `dataio` + `models` + `util`), run it over a protocol, and turn
predictions into metrics / per-clip score CSVs. Run as `python eval/<script>.py`.

## Files

- **`eval_utils.py`** — imported by the three scripts below: adds the repo
  root to `sys.path`, `load_experiment` (exp folder or `model.ckpt`),
  `predict_frame` (run a model over a protocol DataFrame), `parse_score_vector`,
  `assign_components` (CompSpoof `Class`/`speech`/`event`).
- **`score_protocols.py`** — run by the user to score experiments × protocols
  into per-clip score CSVs (paper Tables III–IV). `--exp` takes an experiment
  folder or a single `model.ckpt`; `--data-root` re-points relative audio paths.
- **`subband_eval.py`** — run by the user to do sub-band band-stop probing
  (Table V, RQ2); re-exports `load_experiment` for notebook use.
- **`evaluate.py`** — run by the user to compute metrics (EER, ROC-AUC, accuracy,
  F1) from a prediction CSV, optionally per sample-ID prefix.
