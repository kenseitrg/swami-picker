from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from numpy.typing import NDArray

logger = logging.getLogger(__name__)


@dataclass
class AnnotationRecord:
    """A single spectrum's expert annotation.

    Attributes:
        spectrum_id: Canonical spectrum identifier.
        wavenumber_picks: Dense wavenumber index per frequency column,
            shape ``(256,)``, dtype ``int16``.  ``-1`` = no pick.
        direct_mask: ``True`` at frequencies explicitly clicked by the
            expert, shape ``(256,)``, dtype ``bool``.
        confidence: Per-frequency confidence score in ``[0, 1]``,
            shape ``(256,)``, dtype ``float32``.
        timestamp: ISO-8601 UTC timestamp of last save.
        version: Incremented on every re-save (starts at 1).
    """

    spectrum_id: str
    wavenumber_picks: NDArray[np.int16]
    direct_mask: NDArray[np.bool_]
    confidence: NDArray[np.float32]
    timestamp: str
    version: int = 1

    def __post_init__(self) -> None:
        """Validate array shapes and dtypes."""
        expected = (256,)
        for name, arr in (
            ("wavenumber_picks", self.wavenumber_picks),
            ("direct_mask", self.direct_mask),
            ("confidence", self.confidence),
        ):
            if arr.shape != expected:
                msg = f"{name} must have shape {expected}, got {arr.shape}"
                raise ValueError(msg)

        if self.wavenumber_picks.dtype != np.int16:
            msg = f"wavenumber_picks must be int16, got {self.wavenumber_picks.dtype}"
            raise ValueError(msg)
        if self.direct_mask.dtype != bool:
            msg = f"direct_mask must be bool, got {self.direct_mask.dtype}"
            raise ValueError(msg)
        if self.confidence.dtype != np.float32:
            msg = f"confidence must be float32, got {self.confidence.dtype}"
            raise ValueError(msg)


def compute_confidence(
    wavenumber_picks: NDArray[np.int16],
    direct_mask: NDArray[np.bool_],
) -> NDArray[np.float32]:
    """Compute default confidence scores from pick state.

    Args:
        wavenumber_picks: Dense picks of shape ``(256,)``.
        direct_mask: Direct-pick mask of shape ``(256,)``.

    Returns:
        Confidence array of shape ``(256,)`` with:
        ``1.0`` for direct picks, ``0.5`` for interpolated regions,
        ``0.0`` for unpicked columns (``-1``).
    """
    confidence = np.zeros(256, dtype=np.float32)
    confidence[direct_mask] = 1.0
    interpolated = (~direct_mask) & (wavenumber_picks != -1)
    confidence[interpolated] = 0.5
    return confidence


def save_annotation(record: AnnotationRecord, path: Path) -> None:
    """Persist an annotation record as an ``.npz`` file.

    Args:
        record: The annotation to save.
        path: Destination ``.npz`` path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        wavenumber_picks=record.wavenumber_picks,
        direct_mask=record.direct_mask,
        confidence=record.confidence,
        spectrum_id=np.array(record.spectrum_id, dtype=object),
        timestamp=np.array(record.timestamp, dtype=object),
        version=np.array(record.version),
    )
    logger.info("Saved annotation for %s to %s", record.spectrum_id, path)


def load_annotation(path: Path) -> AnnotationRecord:
    """Load an annotation record from an ``.npz`` file.

    Args:
        path: Path to the ``.npz`` annotation file.

    Returns:
        Reconstructed ``AnnotationRecord``.

    Raises:
        FileNotFoundError: If *path* does not exist.
        KeyError: If required keys are missing from the ``.npz``.
    """
    if not path.exists():
        raise FileNotFoundError(f"Annotation file not found: {path}")

    data = np.load(path, allow_pickle=True)
    try:
        record = AnnotationRecord(
            spectrum_id=str(data["spectrum_id"].item()),
            wavenumber_picks=data["wavenumber_picks"],
            direct_mask=data["direct_mask"],
            confidence=data["confidence"],
            timestamp=str(data["timestamp"].item()),
            version=int(data["version"].item()),
        )
    except KeyError as exc:
        msg = f"Missing key in annotation file {path}: {exc}"
        raise KeyError(msg) from exc
    finally:
        data.close()

    return record


def create_session_manifest(
    session_id: str,
    annotator: str | None,
    percentage: float,
    query_strategy: str,
    per_cluster_target: dict[int, int],
    spectra_ordered: list[str],
    annotations_dir: Path,
) -> dict[str, Any]:
    """Build a session manifest dictionary.

    Args:
        session_id: Unique session identifier (e.g. ``"2026-06-10_iter0"``).
        annotator: Name of the annotator, or ``None``.
        percentage: Coverage percentage used for this session.
        query_strategy: Query strategy name (e.g.
            ``"centroid_boundary"``).
        per_cluster_target: Mapping ``cluster_id -> target_count``.
        spectra_ordered: Ordered list of spectrum IDs to annotate.
        annotations_dir: Directory where per-spectrum ``.npz`` files
            are stored.

    Returns:
        JSON-serialisable manifest dictionary.
    """
    now = datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")
    return {
        "session_id": session_id,
        "created": now,
        "annotator": annotator,
        "percentage_per_cluster": percentage,
        "total_target": sum(per_cluster_target.values()),
        "per_cluster_target": {str(k): v for k, v in per_cluster_target.items()},
        "spectra_ordered": spectra_ordered,
        "query_strategy": query_strategy,
        "annotations_dir": str(annotations_dir),
    }


def save_session_manifest(manifest: dict[str, Any], path: Path) -> None:
    """Write a session manifest to JSON.

    Args:
        manifest: Manifest dictionary.
        path: Destination ``.json`` path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    logger.info("Saved session manifest to %s", path)


def load_session_manifest(path: Path) -> dict[str, Any]:
    """Load a session manifest from JSON.

    Args:
        path: Path to the manifest ``.json`` file.

    Returns:
        Manifest dictionary.

    Raises:
        FileNotFoundError: If *path* does not exist.
    """
    if not path.exists():
        raise FileNotFoundError(f"Session manifest not found: {path}")

    with open(path) as fh:
        manifest: dict[str, Any] = json.load(fh)

    return manifest
