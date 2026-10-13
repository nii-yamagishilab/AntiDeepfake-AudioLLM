"""Tensor helpers for the trainer."""

from typing import Any

import torch


def move_batch_to_device(payload: Any, device):
    """Recursively move all tensors in a batch to device."""
    if torch.is_tensor(payload):
        return payload.to(device=device, non_blocking=True)
    if isinstance(payload, dict):
        return {
            key: move_batch_to_device(value, device)
            for key, value in payload.items()
        }
    if isinstance(payload, list):
        return [move_batch_to_device(item, device) for item in payload]
    if isinstance(payload, tuple):
        return tuple(move_batch_to_device(item, device) for item in payload)
    return payload
