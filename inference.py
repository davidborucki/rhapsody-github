import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks


class HighlightTransformer(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        num_layers: int = 4,
        num_heads: int = 8,
        dropout: float = 0.1,
        max_positions: int = 2048,
    ) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Embedding(max_positions, hidden_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def _ensure_positional_capacity(self, seq_len: int, device: torch.device) -> None:
        if seq_len <= self.pos_encoder.num_embeddings:
            if self.pos_encoder.weight.device != device:
                self.pos_encoder = self.pos_encoder.to(device)
            return

        old_weight = self.pos_encoder.weight.data
        old_count, hidden = old_weight.shape
        new_count = seq_len

        if old_count == 1:
            new_weight = old_weight.expand(new_count, -1).clone()
        else:
            weight_3d = old_weight.t().unsqueeze(0)
            new_weight = F.interpolate(
                weight_3d,
                size=new_count,
                mode="linear",
                align_corners=True,
            ).squeeze(0).t().contiguous()

        embedding = nn.Embedding(new_count, hidden)
        embedding.weight.data.copy_(new_weight)
        self.pos_encoder = embedding.to(device)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        device = x.device

        self._ensure_positional_capacity(seq_len, device)
        projected = self.input_proj(x)
        positions = torch.arange(seq_len, device=device).unsqueeze(0).expand(batch_size, -1)
        encoded = projected + self.pos_encoder(positions)
        transformed = self.transformer(encoded)
        logits = self.classifier(transformed).squeeze(-1)
        return torch.sigmoid(logits)


def postprocess_predictions(
    probs: np.ndarray,
    chunk_duration: float = 2.0,
    threshold: float = 0.5,
    smoothing_sigma: float = 2.0,
    min_gap: float = 4.0,
    min_duration: float = 4.0,
) -> list:
    probs_smooth = gaussian_filter1d(probs, sigma=smoothing_sigma)
    min_distance_chunks = max(1, int(min_gap / max(chunk_duration, 1e-6)))
    peaks, _ = find_peaks(probs_smooth, height=threshold, distance=min_distance_chunks)

    highlights = []
    for peak in peaks:
        start_idx = peak
        while start_idx > 0 and probs_smooth[start_idx - 1] >= threshold * 0.7:
            start_idx -= 1

        end_idx = peak
        last_index = len(probs_smooth) - 1
        while end_idx < last_index and probs_smooth[end_idx + 1] >= threshold * 0.7:
            end_idx += 1

        start_time = start_idx * chunk_duration
        end_time = (end_idx + 1) * chunk_duration
        duration = end_time - start_time
        if duration < min_duration:
            continue

        window_probs = probs_smooth[start_idx:end_idx + 1]
        highlights.append(
            {
                "start": round(start_time, 2),
                "end": round(end_time, 2),
                "duration": round(duration, 2),
                "confidence": round(float(window_probs.mean()), 3),
                "peak_time": round(peak * chunk_duration, 2),
                "peak_confidence": round(float(probs_smooth[peak]), 3),
            }
        )

    return highlights


def _load_numpy(features_path: Path) -> Tuple[np.ndarray, float]:
    if features_path.suffix.lower() == ".npz":
        with np.load(features_path, allow_pickle=False) as data:
            if "features" not in data:
                raise KeyError(f"Expected 'features' array in {features_path}")
            features = data["features"]
            chunk_duration = float(data.get("chunk_duration", 2.0))
    else:
        features = np.load(features_path, allow_pickle=False)
        chunk_duration = 2.0
    return np.asarray(features, dtype=np.float32), chunk_duration


def load_model_state(model_path: Path, device: torch.device) -> Dict[str, torch.Tensor]:
    checkpoint = torch.load(model_path, map_location=device)
    if isinstance(checkpoint, dict):
        for key in ("state_dict", "model_state_dict", "model"):
            if key in checkpoint and isinstance(checkpoint[key], dict):
                checkpoint = checkpoint[key]
                break
    if not isinstance(checkpoint, dict):
        raise ValueError(f"Unsupported checkpoint structure in {model_path}")

    if any(k.startswith("module.") for k in checkpoint):
        checkpoint = {k.replace("module.", "", 1): v for k, v in checkpoint.items()}

    return checkpoint


def build_model(
    input_dim: int,
    checkpoint: Dict[str, torch.Tensor],
    device: torch.device,
    max_positions: int,
) -> HighlightTransformer:
    hidden_dim = checkpoint.get("input_proj.weight").shape[0] if "input_proj.weight" in checkpoint else 256
    pos_weight = checkpoint.get("pos_encoder.weight")
    checkpoint_positions = pos_weight.shape[0] if pos_weight is not None else None
    num_positions = checkpoint_positions if checkpoint_positions is not None else max_positions
    model = HighlightTransformer(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        num_layers=4,
        num_heads=8,
        dropout=0.1,
        max_positions=num_positions,
    )
    model.load_state_dict(checkpoint, strict=False)
    return model.to(device)


def run_inference(
    features_path: str,
    model_path: str,
    output_path: str,
    threshold: float = 0.5,
    smoothing: float = 2.0,
    min_gap: float = 4.0,
    min_duration: float = 4.0,
    device: str = "cpu",
    max_positions: int = 2048,
    chunk_duration_override: float = None,
) -> None:
    features_file = Path(features_path)
    model_file = Path(model_path)
    output_file = Path(output_path)

    if not features_file.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")

    print(f"Loading features from {features_file}...")
    features, chunk_duration = _load_numpy(features_file)
    if features.ndim != 2:
        raise ValueError(f"Expected 2D feature matrix, got shape {features.shape}")
    chunk_duration = chunk_duration_override or chunk_duration
    print(f"  features shape: {features.shape}")
    print(f"  chunk duration: {chunk_duration:.3f}s")

    torch_device = torch.device(device)
    print(f"Loading model from {model_file}...")
    state_dict = load_model_state(model_file, torch_device)
    model = build_model(features.shape[1], state_dict, torch_device, max_positions)
    model.eval()
    print("  model ready for inference")

    with torch.no_grad():
        tensor = torch.from_numpy(features).unsqueeze(0).to(torch_device)
        probs = model(tensor).squeeze(0).cpu().numpy()

    print(f"Generated {len(probs)} per-chunk probabilities")

    highlights = postprocess_predictions(
        probs,
        chunk_duration=chunk_duration,
        threshold=threshold,
        smoothing_sigma=smoothing,
        min_gap=min_gap,
        min_duration=min_duration,
    )
    print(f"Detected {len(highlights)} highlight spans")

    results = {
        "metadata": {
            "num_chunks": len(features),
            "total_duration": round(len(features) * chunk_duration, 2),
            "threshold": threshold,
            "num_highlights": len(highlights),
        },
        "highlights": highlights,
    }

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with output_file.open("w", encoding="utf-8") as fh:
        json.dump(results, fh, indent=2)

    print(f"Results saved to {output_file}")
    if highlights:
        print("\nHIGHLIGHTS")
        print("=" * 60)
        for idx, span in enumerate(highlights, start=1):
            print(
                f"{idx}. {span['start']}s - {span['end']}s "
                f"({span['duration']}s) | avg conf {span['confidence']:.3f}"
            )
        print("=" * 60)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run highlight inference on feature sequences")
    parser.add_argument("features", help="Path to feature matrix (.npy or .npz)")
    parser.add_argument("--model", default="highlight_model.pt", help="Path to the trained model")
    parser.add_argument("--output", default="highlights.json", help="Where to write the output JSON")
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold for highlights")
    parser.add_argument("--smoothing", type=float, default=2.0, help="Gaussian smoothing sigma")
    parser.add_argument("--min-gap", type=float, default=4.0, help="Minimum gap between highlights in seconds")
    parser.add_argument("--min-duration", type=float, default=4.0, help="Minimum highlight duration in seconds")
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu", help="Inference device")
    parser.add_argument("--max-positions", type=int, default=2048, help="Starting size for positional embeddings")
    parser.add_argument(
        "--chunk-duration",
        type=float,
        default=None,
        help="Override chunk duration (seconds) if not stored alongside features",
    )
    return parser.parse_args()


if __name__ == "__main__":
    ARGS = parse_args()
    run_inference(
        features_path=ARGS.features,
        model_path=ARGS.model,
        output_path=ARGS.output,
        threshold=ARGS.threshold,
        smoothing=ARGS.smoothing,
        min_gap=ARGS.min_gap,
        min_duration=ARGS.min_duration,
        device=ARGS.device,
        max_positions=ARGS.max_positions,
        chunk_duration_override=ARGS.chunk_duration,
    )
