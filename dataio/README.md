# dataio/ — protocol CSV → model-ready batches

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

**This folder** owns the first stage: read the protocol CSV, emit one uniform
example dict per row, and collate a batch. It is model-agnostic — the per-model
tensor packing is delegated to the collator supplied by `models/`.

## Files

- **`dataset.py`** — used by `main.py` and `eval/*` to turn the protocol CSV into
  example dicts and collate batches (`ProtocolDataset`, `BatchCollator`,
  `build_datasets`, `build_loader_kwargs`).
- **`audio_loading.py`** — used by `dataset.py` to ffmpeg-decode each clip to a
  16 kHz mono waveform and crop it (`WaveformLoader`, `load_audio`).
- **`__init__.py`** — used by callers (`from dataio import ...`) to expose the
  dataset, collator, and DataLoader builders.

## Editing guide

Where to edit for each experiment change, and each file's functions.

### `dataset.py`

```text
Edit here to:
  - use your own protocol CSVs / column names ->
    config `dataset.*_path` / `*_key` (no code change)
  - toggle class-balanced sampling ->
    config `training.balanced_sampling` (no code change)
  - add fields to each example -> ProtocolDataset.__getitem__
  - change sampling weights -> build_weighted_sampler

ProtocolDataset:
  __getitem__(i) -- CSV row -> example dict (id/audio/prompt/label)
  label_names() -- normalized label per row (for sampler weighting)

BatchCollator:
  __call__(batch) -- wrap dicts as namespaces, call model collator

Functions:
  build_weighted_sampler(ds) -- inverse-frequency sampler
  build_loader_kwargs(cfg, collator, split, ds) -- DataLoader kwargs
  build_datasets(cfg, template) -- train/valid/test dataset bundle
```

### `audio_loading.py`

```text
Edit here to:
  - change sample rate -> config `dataset.audio.sample_rate` (no code change)
  - change crop length (default 30 s, 0 disables) ->
    config `dataset.audio.max_seconds` (no code change)
  - change crop policy (random chunk, every split) -> _random_crop
  - change decoding or add waveform preprocessing -> load_audio

Functions:
  load_audio(path, sr) -- ffmpeg-decode to float32 mono waveform
  to_absolute_path(path, root) -- expand $ROOT/relative to absolute path
  build_waveform_loader(cfg) -- build WaveformLoader from audio config

WaveformLoader:
  load(path) -- decode waveform via ffmpeg, with optional crop
```
