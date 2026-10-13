"""Audio decoding and dataset-side waveform loading."""

import os
import random
from subprocess import CalledProcessError, run
from typing import Any, Optional

import numpy as np
import torch

SAMPLE_RATE = 16000
_MAX_AUDIO_SECONDS = 30


def load_audio(file_path: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """Decode audio with ffmpeg, return float32 waveform."""
    command = [
        "ffmpeg",
        "-nostdin",
        "-threads",
        "0",
        "-i",
        file_path,
        "-f",
        "s16le",
        "-ac",
        "1",
        "-acodec",
        "pcm_s16le",
        "-ar",
        str(sr),
        "-",
    ]
    try:
        output = run(command, capture_output=True, check=True).stdout
    except CalledProcessError as exc:
        raise RuntimeError(
            f"Failed to load audio: {exc.stderr.decode()}"
        ) from exc
    return (
        np.frombuffer(output, np.int16).flatten().astype(np.float32) / 32768.0
    )


class WaveformLoader:
    """Resolve optional waveform/feature loading for dataset examples."""

    def __init__(
        self,
        load_mode: str = "none",
        sample_rate: int = SAMPLE_RATE,
        max_seconds: float = _MAX_AUDIO_SECONDS,
    ):
        self.load_mode = load_mode
        self.sample_rate = sample_rate
        self.max_samples = (
            int(max_seconds * sample_rate) if max_seconds > 0 else 0
        )

    def load(self, audio_path: str) -> Optional[Any]:
        """Load waveform randomly cropped to max_seconds, or None."""
        if self.load_mode in ("none", None):
            return None
        if self.load_mode == "waveform":
            return self._crop(
                torch.tensor(load_audio(audio_path, sr=self.sample_rate))
            )
        raise ValueError(f"Unsupported audio load mode: {self.load_mode}")

    def _crop(self, waveform: torch.Tensor) -> torch.Tensor:
        if self.max_samples > 0:
            return _random_crop(waveform, self.max_samples)
        return waveform


def to_absolute_path(audio_path: str, data_root: Optional[str]) -> str:
    """Expand `$ROOT` and relative paths into absolute audio file paths."""
    if audio_path is None:
        return ""
    resolved = str(audio_path)
    if data_root:
        normalized_root = os.path.abspath(data_root)
        resolved = resolved.replace("$ROOT", normalized_root)
        if not os.path.isabs(resolved):
            resolved = os.path.join(normalized_root, resolved)
    return os.path.abspath(resolved)


def build_waveform_loader(dataset_config: dict[str, Any]) -> WaveformLoader:
    """Create the dataset-side audio loader described by the config."""
    audio_cfg = dataset_config.get("audio", {})
    return WaveformLoader(
        load_mode=audio_cfg.get("load_mode", "none"),
        sample_rate=audio_cfg.get("sample_rate", SAMPLE_RATE),
        max_seconds=audio_cfg.get("max_seconds", _MAX_AUDIO_SECONDS),
    )


def _random_crop(waveform: torch.Tensor, max_samples: int) -> torch.Tensor:
    """Return a random contiguous chunk of at most max_samples."""
    if waveform.shape[-1] <= max_samples:
        return waveform
    start = random.randint(0, waveform.shape[-1] - max_samples)
    return waveform[..., start : start + max_samples]
