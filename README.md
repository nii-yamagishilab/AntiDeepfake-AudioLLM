<!-- markdownlint-disable first-line-h1 -->
<!-- markdownlint-disable html -->
<!-- markdownlint-disable no-duplicate-header -->

<div align="center">
  <h1>What Do Deepfake Audio Detectors Hear? From Self-Supervised Models to Audio LLMs</p>
</div>

<div align="center" style="line-height: 1;">
  <a href="https://huggingface.co/nii-yamagishilab/slt2026-checkpoints"><img alt="Hugging Face"
    src="https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Checkpoints-ffc107?color=ffc107&logoColor=white"/></a>
  <a href="./LICENSE"><img alt="Code License"
    src="https://img.shields.io/badge/Code_License-BSD%203--Clause-f5de53?&color=f5de53"/></a>
  <a href="./LICENSE-CHECKPOINT"><img alt="Model License"
    src="https://img.shields.io/badge/Checkpoint_License-CC BY%20NC 4.0-f5de53?&color=f5de53"/></a>

  <br>
  <a href="https://arxiv.org/abs/2610.17617"><b>Paper Link</b></a>
</div>

---

## 1. Introduction

This repository contains scripts to reproduce our paper *What Do Deepfake Audio Detectors Hear? From Self-Supervised Models to Audio LLMs* published in IEEE Spoken Language Technology Workshop 2026.

---

## 2. Installation

```bash
conda create -n YOUR_ENV_NAME python=3.10 -y
conda activate YOUR_ENV_NAME
conda install -c conda-forge ffmpeg -y
pip install --upgrade pip
python3 -m pip install -r requiments.txt
```

---

## 3. Data

Detectors are trained and evaluated on the **CompSpoof V2** database, available here:

- 🤗 [XuepingZhang/ESDD2-CompSpoof-V2](https://huggingface.co/datasets/XuepingZhang/ESDD2-CompSpoof-V2)

Each mixed clip is one of four classes:

| Class | Speech | Event | Label |
|-------|--------|-------|-------|
| `bonafide_bonafide` | real | real | **real** |
| `bonafide_spoof` | real | fake | fake |
| `spoof_bonafide` | fake | real | fake |
| `spoof_spoof` | fake | fake | fake |

The all-real mixture is **real**; any fake component makes it **fake**. The unmixed `original`
class is excluded to avoid a shortcut on the mixing operation.

Generate `train.csv` / `valid.csv` / `test.csv` from the CompSpoof V2 metadata:

```bash
python protocols/generate_compspoofv2_protocol.py --help
```

Our dataloader is designed to load three columns from a given CSV protocol:

- `ID` — unique clip identifier
- `Path` — audio file path, read relative to `<base_path>/Data`
- `Label` — the real/fake label

Any additional columns (e.g. `Attack`)
are carried through as metadata and used only for per-class scoring. You can refer to our previous
scripts to generate similar CSV protocols for other databases:
[nii-yamagishilab/AntiDeepfake/protocols](https://github.com/nii-yamagishilab/AntiDeepfake/tree/main/protocols).

---

## 4. Training

`main.py` loads a config and accepts SpeechBrain / HyperPyYAML overrides. The configs already carry
the paper's exact hyperparameters — no tuning needed. Point at your data with `--base_path`
(audio lives under `<base_path>/Data`) and `--protocol_folder`. The script loads the
`train.csv` / `valid.csv` / `test.csv` protocols from the folder given by `--protocol_folder`.

```bash
# XLS-R + FC
python main.py config/encoder_fc_classification.yaml \
  --base_path /path/to/root --protocol_folder ./protocols --run_name xlsr_fc

# Whisper + FC
python main.py config/encoder_fc_classification.yaml \
  --base_path /path/to/root --protocol_folder ./protocols \
  --encoder_name openai/whisper-large-v2 --run_name whisper_fc

# Qwen2-Audio, discriminative
python main.py config/modular_transformers_audio.yaml \
  --base_path /path/to/root --protocol_folder ./protocols \
  --head_type classification --run_name qwen_cls

# Qwen2-Audio, generative
python main.py config/modular_transformers_audio.yaml \
  --base_path /path/to/root --protocol_folder ./protocols --run_name qwen_gen
```

Please refer to the two `config/*.yaml` files (commented inline) for the training settings and
hyperparameters.

**Choosing verbalizer words.** You may replace `word_for_real: real` / `word_for_fake: fake` with other words. For the **generative** head, each word must be a **single token** in the backbone's tokenizer; the classification head has no such constraint. Check a word before training:

```python
from transformers import AutoProcessor

tok = AutoProcessor.from_pretrained("Qwen/Qwen2-Audio-7B-Instruct").tokenizer
tok.encode("apple", add_special_tokens=False)      # [22377]        -> 1 token: OK
tok.encode("pineapple", add_special_tokens=False)  # [38038, 22377] -> 'pine' + 'apple': avoid this case
```

A multi-token word still trains correctly, because the loss covers every token of the word. Scoring and inference do not: they read only the word's first token, so `pineapple` is scored as `pine`.

---

## 5. Evaluation

**Per-clip scoring** — writes a CSV with `Score_real` / `Score_fake` and the CompSpoof `Class`:

```bash
python eval/score_protocols.py \
  --exp /path/to/exps/<experiment-folder> \
  --protocol /path/to/CompSpoofV2/test.csv \
  --out-dir scores/ --checkpoint best_valid --components
```

`--exp` also accepts a single checkpoint file (`.../save/<step>/model.ckpt`): the training config
is stored inside it, so the experiment folder is not needed and `--checkpoint` is ignored. If your
protocol uses relative or `$ROOT` audio paths, add `--data-root /path/to/audio/root`.

**EER** — target is always `bonafide_bonafide`; scored once per fake condition
(`bonafide_spoof` = fake event, `spoof_bonafide` = fake speech, `spoof_spoof` = both).

**Sub-band probing** — zero one of six 0–8 kHz bands with a Butterworth band-stop and re-measure EER:

```python
import sys; sys.path.insert(0, "eval")
import pandas as pd
from subband_eval import load_experiment, run_conditions, BANDS

exp = load_experiment("/path/to/exps/<experiment-folder>", device="cuda", checkpoint="best_valid")
df = pd.read_csv("/path/to/CompSpoofV2/test.csv")
scores = run_conditions(exp, df, bands=BANDS)  # {"clean": df, "stop_0-0.8k": df, ...}
```

---

## 6. Acknowledgments

This study is supported by JSPS KAKENHI Grant (25K24398). This study was carried out on the
TSUBAME4.0 supercomputer at the Institute of Science Tokyo.

Codes are based on the implementations of
[wav2vec 2.0 pretraining with SpeechBrain](https://github.com/speechbrain/speechbrain/tree/develop/recipes/LibriSpeech/self-supervised-learning/wav2vec2),
[qwen_audio_for_add](https://github.com/ucas-hao/qwen_audio_for_add), and
[AntiDeepfake](https://github.com/nii-yamagishilab/AntiDeepfake/tree/main).

The code in this repository is released under the BSD 3-Clause License. See [`LICENSE`](LICENSE).
The model checkpoints are released under CC BY-NC 4.0. See [`LICENSE-CHECKPOINT`](LICENSE-CHECKPOINT).

---

## 7. Citation

If you find this repository useful, please consider citing:

```bibtex
@inproceedings{ge2026what,
  title     = {What Do Deepfake Audio Detectors Hear? From Self-Supervised Models to Audio LLMs},
  author    = {Ge, Wanying and Wang, Xin and Yamagishi, Junichi},
  booktitle = {IEEE Spoken Language Technology Workshop (SLT)},
  year      = {2026}
}
```
