from dataclasses import dataclass
from typing import Optional

import numpy as np
import scipy.io
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
    raw = scipy.io.loadmat(case.data_path)

    def to_tensor(key: str) -> torch.Tensor:
        return torch.from_numpy(raw[key].astype(np.float32))

    x_train = to_tensor("f_train").reshape(-1, case.sequence_length, 1)
    y_train = to_tensor("u_train").reshape(-1, case.sequence_length, 1)
    x_val = to_tensor("f_vali").reshape(-1, case.sequence_length, 1)
    y_val = to_tensor("u_vali").reshape(-1, case.sequence_length, 1)
    x_test = to_tensor("f_test").reshape(-1, case.sequence_length, 1)
    y_test = to_tensor("u_test").reshape(-1, case.sequence_length, 1)
    grid = to_tensor("x_train").reshape(case.sequence_length)

    return DataBundle(
        x_train=_slice_count(x_train, max_train_samples),
        y_train=_slice_count(y_train, max_train_samples),
        x_val=_slice_count(x_val, max_val_samples),
        y_val=_slice_count(y_val, max_val_samples),
        x_test=_slice_count(x_test, max_test_samples),
        y_test=_slice_count(y_test, max_test_samples),
        grid=grid,
    )


def build_dataloaders(
    data: DataBundle,
    batch_size: int,
    *,
    seed: Optional[int] = None,
):
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
