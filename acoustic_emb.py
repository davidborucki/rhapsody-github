import argparse
from pathlib import Path

import numpy as np
import torch
import torchaudio
from transformers import Wav2Vec2Model, Wav2Vec2Processor


MODEL_NAME = "facebook/wav2vec2-base-960h"
DEFAULT_FILENAME = "acoustic_emb_2s.npz"  # embeddings + timestamps


def load_audio(path: Path) -> tuple[torch.Tensor, int]:
    speech, rate = torchaudio.load(path)
    # force mono before resampling
    if speech.shape[0] > 1:
        speech = speech.mean(dim=0, keepdim=True)
    if rate != 16000:
        speech = torchaudio.functional.resample(speech, rate, 16000)
        rate = 16000
    # flatten to 1D float32
    return speech.squeeze(0).to(torch.float32), rate


def compute_embeddings(audio: torch.Tensor, rate: int, processor, model, batch_size: int = 32):
    """
    Slice the waveform into exact 2s windows, embed each window,
    mean-pool over time, and attach perfect wall-clock timestamps.
    """
    chunk_sec = 2.0
    chunk_samples = int(rate * chunk_sec)
    total_samples = audio.shape[0]
    num_chunks = total_samples // chunk_samples  # drop final remainder to keep exact 2s windows

    if num_chunks == 0:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 2), dtype=np.float32)

    embeddings_list: list[np.ndarray] = []
    timestamps: list[tuple[float, float]] = []

    # Process in batches for speed
    for base in range(0, num_chunks, batch_size):
        idxs = list(range(base, min(num_chunks, base + batch_size)))
        # Build a list of numpy arrays (each length = 2s)
        batch_chunks = [
            audio[i * chunk_samples : (i + 1) * chunk_samples].numpy()
            for i in idxs
        ]

        inputs = processor(batch_chunks, sampling_rate=rate, return_tensors="pt", padding=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        with torch.no_grad():
            hs = model(**inputs).last_hidden_state  # (B, T, H)

        # Mean over time dimension to get one vector per 2s window
        mean_emb = hs.mean(dim=1).cpu().numpy().astype(np.float32)  # (B, H)
        embeddings_list.append(mean_emb)

        # Wall-clock timestamps for each 2s chunk
        for i in idxs:
            start_t = i * chunk_sec
            end_t = start_t + chunk_sec
            timestamps.append((start_t, end_t))

    embeddings = np.vstack(embeddings_list)  # (num_chunks, hidden_dim)
    ts = np.array(timestamps, dtype=np.float32)  # (num_chunks, 2)
    return embeddings, ts


def resolve_output_path(destination: str) -> Path:
    dest_path = Path(destination)
    if dest_path.is_dir() or destination.endswith(("/", "\\")):
        return dest_path / DEFAULT_FILENAME
    return dest_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate 2s acoustic embeddings with Wav2Vec2 (+ timestamps)")
    parser.add_argument("source", help="Path to the source audio file")
    parser.add_argument("destination", help="Output file or directory for the embeddings")
    parser.add_argument("--batch-size", type=int, default=32, help="Number of 2s windows per forward pass")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source)
    if not source_path.exists():
        raise FileNotFoundError(f"Audio file not found: {source_path}")

    output_path = resolve_output_path(args.destination)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading model {MODEL_NAME}…")
    processor = Wav2Vec2Processor.from_pretrained(MODEL_NAME)
    model = Wav2Vec2Model.from_pretrained(MODEL_NAME)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device).eval()

    print(f"Loading audio from {source_path}…")
    audio, rate = load_audio(source_path)

    print("Computing 2s embeddings with timestamps…")
    embeddings, timestamps = compute_embeddings(audio, rate, processor, model, batch_size=args.batch_size)

    if embeddings.size == 0:
        print("⚠️ No 2s chunks extracted. No file written.")
        return

    np.savez(output_path, embeddings=embeddings, timestamps=timestamps)
    print(f"✅ Saved {embeddings.shape[0]} chunks to {output_path} (emb {embeddings.shape}, ts {timestamps.shape})")


if __name__ == "__main__":
    main()

