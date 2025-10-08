import argparse
from pathlib import Path

import numpy as np
import opensmile
import torch
import torchaudio
from torchaudio.transforms import Resample


DEFAULT_FILENAME = "dva_emb_2s.npz"  # embeddings + timestamps
TARGET_RATE = 16000
CHUNK_SECONDS = 2.0


def resolve_output_path(destination: str) -> Path:
    dest_path = Path(destination)
    if dest_path.is_dir() or destination.endswith(("/", "\\")):
        return dest_path / DEFAULT_FILENAME
    return dest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract 2s DVA embeddings with OpenSMILE emobase (+ timestamps)")
    parser.add_argument("source", help="Path to the source audio file")
    parser.add_argument("destination", help="Output file or directory for the embeddings")
    return parser.parse_args()

def stream_resampled_chunks(path: Path):
    waveform, sr = torchaudio.load(str(path))
    waveform = waveform.mean(dim=0, keepdim=True)  # mono
    if sr != TARGET_RATE:
        waveform = Resample(orig_freq=sr, new_freq=TARGET_RATE)(waveform)

    mono = waveform.squeeze(0)
    chunk_frames = int(TARGET_RATE * CHUNK_SECONDS)

    for i in range(0, mono.shape[0] // chunk_frames):
        start = i * chunk_frames
        end = start + chunk_frames
        yield mono[start:end].numpy()


def compute_dva_embeddings(path: Path):
    smile = opensmile.Smile(
        feature_set=opensmile.FeatureSet.emobase,
        feature_level=opensmile.FeatureLevel.Functionals,
    )

    feature_dim = None
    chunks = []
    timestamps = []

    idx = 0
    for chunk in stream_resampled_chunks(path):
        feats = smile.process_signal(chunk, TARGET_RATE).to_numpy().mean(axis=0)
        if feature_dim is None:
            feature_dim = feats.shape[0]
        chunks.append(feats)

        # strict 2s windows → timestamps are exact
        start_t = idx * CHUNK_SECONDS
        end_t = start_t + CHUNK_SECONDS
        timestamps.append((start_t, end_t))
        idx += 1

    if not chunks:
        return (
            np.empty((0, feature_dim or 0), dtype=np.float32),
            np.empty((0, 2), dtype=np.float32),
        )

    stacked = np.vstack(chunks).astype(np.float32, copy=False)
    ts = np.array(timestamps, dtype=np.float32)
    return stacked, ts


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {source_path}")

    output_path = resolve_output_path(args.destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Processing audio from {source_path}…")
    embeddings, timestamps = compute_dva_embeddings(source_path)

    if embeddings.size == 0:
        print("⚠️ No 2s chunks extracted. No file written.")
        return

    np.savez(output_path, embeddings=embeddings, timestamps=timestamps)
    print(f"✅ Saved {embeddings.shape[0]} chunks to {output_path} "
          f"(emb {embeddings.shape}, ts {timestamps.shape})")


if __name__ == "__main__":
    main()

