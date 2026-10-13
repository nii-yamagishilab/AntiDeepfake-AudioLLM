"""Sub-band band-stop probing for RQ2 analysis.

Functions:
  run_conditions(exp, df, bands) -- score clean + each band-stop cond
  apply_bandstop(wav, band) -- filter out one sub-band (None = clean)
  band_label(band) -- condition label ('clean' or 'stop_LO-HIk')

BandStopDataset(ProtocolDataset):
  __getitem__(i) -- example with one audio sub-band muted
"""

import numpy as np
import pandas as pd
import torch
from eval_utils import (
    load_experiment,  # noqa: F401  re-exported for notebook use
    parse_score_vector,
    predict_frame,
)
from scipy.signal import butter, sosfiltfilt

from dataio.audio_loading import SAMPLE_RATE
from dataio.dataset import ProtocolDataset

__author__ = "Wanying Ge, Xin Wang"
__email__ = "gewanying@nii.ac.jp, wangxin@nii.ac.jp"
__copyright__ = "Copyright 2025, National Institute of Informatics"

# Six sub-bands spanning 0-8 kHz (Nyquist), muted one at a time (paper §III.D).
BANDS = [
    (0, 800),
    (800, 2400),
    (2400, 4000),
    (4000, 5600),
    (5600, 7200),
    (7200, 8000),
]


def run_conditions(
    exp, df, bands=BANDS, merge_cols=("Class", "speech", "event")
):
    """Score df under clean and each band-stop condition."""
    meta = df.set_index(df["ID"].astype(str))
    results = {}
    for band in [None, *bands]:
        _, pred_df = predict_frame(exp, df, BandStopDataset, band=band)
        results[band_label(band)] = _scores_to_frame(
            pred_df, exp, meta, merge_cols
        )
    return results


def apply_bandstop(wav, band):
    """Suppress one sub-band via zero-phase Butterworth filter."""
    # band=None is the clean (unfiltered) condition.
    if band is None:
        return wav
    x = wav.detach().cpu().numpy().astype(np.float64)
    x = sosfiltfilt(_bandstop_sos(*band), x)
    return torch.from_numpy(np.ascontiguousarray(x)).to(dtype=wav.dtype)


def band_label(band):
    """Condition label: None -> 'clean', else stop_LO-HIk."""
    if band is None:
        return "clean"
    lo, hi = (f"{hz / 1000:g}" for hz in band)
    return f"stop_{lo}-{hi}k"


class BandStopDataset(ProtocolDataset):
    """ProtocolDataset that mutes one sub-band per item."""

    def __init__(self, *args, band=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.band = band

    def __getitem__(self, index):
        d = super().__getitem__(index)
        if d.get("audio") is not None:
            d["audio"] = apply_bandstop(d["audio"], self.band)
        return d


# ── Private helpers ───────────────────────────────────────────────────────────


def _bandstop_sos(lo, hi, sr=SAMPLE_RATE, order=8):
    """SOS coefficients for a band-stop filter at lo-hi Hz."""
    # Edge bands become high-pass (lo=0) or low-pass (hi=Nyquist) filters,
    # since a band-stop cutoff cannot sit at 0 Hz or at Nyquist.
    nyq = sr / 2
    if lo <= 0:
        return butter(order, hi / nyq, btype="highpass", output="sos")
    if hi >= nyq:
        return butter(order, lo / nyq, btype="lowpass", output="sos")
    return butter(order, [lo / nyq, hi / nyq], btype="bandstop", output="sos")


def _scores_to_frame(pred_df, exp, meta, merge_cols):
    """Join brain.evaluate output with per-clip metadata."""
    records = []
    for _, row in pred_df.iterrows():
        score = parse_score_vector(row["score_vector"])
        if score is None:
            continue
        sid = str(row["sample_id"])
        record = {
            "ID": sid,
            "Label": row["gold_label"],
            "Score_real": float(score[exp.real_id]),
            "Score_fake": float(score[exp.fake_id]),
        }
        if sid in meta.index:
            source = meta.loc[sid]
            for col in merge_cols:
                if col in source:
                    record[col] = source[col]
        records.append(record)
    return pd.DataFrame(records)
