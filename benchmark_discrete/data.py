import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from .configs import CaseConfig


@dataclass
class DataBundle:
    x_train: torch.Tensor
    y_train: torch.Tensor
    x_val: torch.Tensor
    y_val: torch.Tensor
    x_test: torch.Tensor
    y_test: torch.Tensor
    grid: torch.Tensor
    ignore_prefix_steps: int
    meta: Dict[str, Any]


def _slice_count(tensor: torch.Tensor, count: Optional[int]) -> torch.Tensor:
    if count is None:
        return tensor
    return tensor[:count]


def load_case_data(
    case: CaseConfig,
    max_train_samples: Optional[int] = None,
    max_val_samples: Optional[int] = None,
    max_test_samples: Optional[int] = None,
) -> DataBundle:
    raw = np.load(case.data_path)
    meta_path = Path(case.data_path).with_name("meta.json")
    data_meta: Dict[str, Any] = {}
    if meta_path.exists():
        data_meta = json.loads(meta_path.read_text(encoding="utf-8"))
    ignore_prefix_steps = int(data_meta.get("metric_ignore_prefix_steps", 0))

    def to_tensor(key: str) -> torch.Tensor:
        return torch.from_numpy(raw[key].astype(np.float32))

    x_train = to_tensor("x_train")
    y_train = to_tensor("y_train")
    x_val = to_tensor("x_val")
    y_val = to_tensor("y_val")
    x_test = to_tensor("x_test")
    y_test = to_tensor("y_test")
    grid = to_tensor("grid").reshape(-1)

    return DataBundle(
        x_train=_slice_count(x_train, max_train_samples),
        y_train=_slice_count(y_train, max_train_samples),
        x_val=_slice_count(x_val, max_val_samples),
        y_val=_slice_count(y_val, max_val_samples),
        x_test=_slice_count(x_test, max_test_samples),
        y_test=_slice_count(y_test, max_test_samples),
        grid=grid,
        ignore_prefix_steps=ignore_prefix_steps,
        meta=data_meta,
    )


def build_dataloaders(data: DataBundle, batch_size: int, *, seed: Optional[int] = None):
    generator = None
    if seed is not None:
        generator = torch.Generator()
        generator.manual_seed(seed)
    train_loader = DataLoader(
        TensorDataset(data.x_train, data.y_train),
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
    )
    val_loader = DataLoader(
        TensorDataset(data.x_val, data.y_val),
        batch_size=batch_size,
        shuffle=False,
    )
    test_loader = DataLoader(
        TensorDataset(data.x_test, data.y_test),
        batch_size=1,
        shuffle=False,
    )
    return train_loader, val_loader, test_loader
