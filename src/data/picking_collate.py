"""Custom collate function for Phase 4 picking batches.

Handles variable ``cluster_embedding`` presence and string spectrum IDs.
"""

from __future__ import annotations

import torch


def picking_collate(batch: list) -> tuple:
    """Collate a list of picking dataset items into tensors.

    Args:
        batch: List of tuples returned by ``FKPickingDataset.__getitem__``.

    Returns:
        Tuple of batched tensors and a list of spectrum IDs.
    """
    spectra, pick_targets, direct_masks, confidences, cluster_labels, spectrum_ids = (
        zip(*batch)
    )
    return (
        torch.stack(spectra, dim=0),
        torch.stack(pick_targets, dim=0),
        torch.stack(direct_masks, dim=0),
        torch.stack(confidences, dim=0),
        torch.stack(cluster_labels, dim=0),
        list(spectrum_ids),
    )
