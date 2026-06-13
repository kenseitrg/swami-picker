"""Custom collate function for Phase 4 picking batches.

The default PyTorch collate cannot handle ``None`` cluster embeddings
mixed with tensors.  This module provides a collate function that stacks
tensors and leaves ``None`` / string fields as lists.
"""

from __future__ import annotations

from typing import Any

import torch


def picking_collate(batch: list[tuple[Any, ...]]) -> tuple[Any, ...]:
    """Collate a list of picking dataset samples into a batch.

    The dataset returns a 7-tuple where the sixth element may be ``None``
    (no cluster embedding) or a tensor.  The seventh element is a string
    spectrum id.  This collate stacks all tensor fields and returns lists
    for ``None`` and string fields.

    Args:
        batch: List of samples from ``FKPickingDataset``.

    Returns:
        A 7-tuple suitable for ``PickingTrainer``.
    """
    transposed: list[list[Any]] = [[] for _ in range(len(batch[0]))]
    for sample in batch:
        for i, item in enumerate(sample):
            transposed[i].append(item)

    output: list[Any] = []
    for col_idx, values in enumerate(transposed):
        if all(v is None for v in values):
            output.append(None)
        elif col_idx == 6:
            # spectrum_id strings
            output.append(values)
        elif isinstance(values[0], torch.Tensor):
            output.append(torch.stack(values, dim=0))
        else:
            output.append(values)

    return tuple(output)
