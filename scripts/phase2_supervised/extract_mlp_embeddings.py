"""Extract MLP penultimate-layer embeddings for the full dataset.

Loads the trained MLP classifier and runs inference on all 1,392 spectra's
20-D descriptor features, producing 128-D embeddings for Phase 3 clustering.

Usage:
    python scripts/phase2_supervised/extract_mlp_embeddings.py \
        --checkpoint experiments/phase2c-mlp-11cl-final-100ep/checkpoints/best_model.pt \
        --output data/processed/mlp_embeddings_phase3.npz
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.models.pseudo_label_classifier import MLPClassifier


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Extract MLP embeddings for Phase 3")
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the trained MLP checkpoint .pt file.",
    )
    parser.add_argument(
        "--features",
        type=str,
        default="data/processed/features/features_descriptors.npz",
        help="Path to descriptor features .npz file.",
    )
    parser.add_argument(
        "--labels",
        type=str,
        default="experiments/2026-06-07_phase2c-descriptor-umap5-mindist0/pseudo_labels_merged.npz",
        help="Path to merged pseudo-labels .npz file.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed/mlp_embeddings_phase3.npz",
        help="Output path for embeddings .npz file.",
    )
    args = parser.parse_args(argv)

    # Load checkpoint
    checkpoint = torch.load(
        Path(args.checkpoint), map_location="cpu", weights_only=False
    )
    state_dict = checkpoint["model"]
    config = checkpoint["config"]

    # Determine architecture from state dict keys
    # MLPClassifier net: Linear(0) → ReLU(1) → Dropout(2) → Linear(3) → ReLU(4) → Dropout(5) → Linear(6)
    input_dim = int(state_dict["net.0.weight"].shape[1])
    h1 = int(state_dict["net.0.weight"].shape[0])
    h2 = (
        int(state_dict["net.3.weight"].shape[0])
        if "net.3.weight" in state_dict
        else None
    )
    hidden_dims = [h1, h2] if h2 is not None else [h1]
    # Find final Linear layer (highest even index in state_dict)
    linear_keys = [k for k in state_dict if "weight" in k and k.startswith("net.")]
    final_key = max(linear_keys, key=lambda k: int(k.split(".")[1]))
    num_classes = int(state_dict[final_key].shape[0])

    model = MLPClassifier(
        input_dim=input_dim,
        hidden_dims=hidden_dims,
        num_classes=num_classes,
        dropout=config.get("mlp_dropout", 0.2),
    )
    model.load_state_dict(state_dict)
    model.eval()

    # Load features
    feat_data = np.load(Path(args.features))
    features = np.array(feat_data["features"])
    spectrum_ids = np.array(feat_data["spectrum_ids"])
    feat_data.close()

    # Load labels (including noise = -1)
    label_data = np.load(Path(args.labels))
    labels = np.array(label_data["labels"])
    label_spectrum_ids = np.array(label_data["spectrum_ids"])
    label_data.close()

    # Align labels with features by spectrum_id
    id_to_label = {sid: int(lbl) for sid, lbl in zip(label_spectrum_ids, labels)}
    aligned_labels = np.array([id_to_label.get(sid, -1) for sid in spectrum_ids])

    # Extract embeddings
    with torch.no_grad():
        x = torch.from_numpy(features).float()
        embeddings = model.extract_embedding(x).numpy()

    # Save
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        embeddings=embeddings,
        spectrum_ids=spectrum_ids,
        labels=aligned_labels,
    )
    print(
        f"Saved {len(embeddings)} embeddings of shape {embeddings.shape} to {output_path}"
    )
    print(f"  Spectrum IDs: {len(spectrum_ids)}")
    print(
        f"  Labels: {len(aligned_labels)} (including {(aligned_labels == -1).sum()} noise)"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
