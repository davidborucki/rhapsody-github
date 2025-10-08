import argparse
from pathlib import Path
import numpy as np
import json
from sklearn.decomposition import PCA

# --- Temporal context helper ---
def add_temporal_context(features, window_sizes=[2, 4, 8]):
    """
    Expand features with multi-scale pooling + deltas.
    features: [T, D]
    returns: [T, D_expanded]
    """
    T, D = features.shape
    context_features = [features]  # keep original

    # Multi-scale pooling
    for window_sec in window_sizes:
        w = window_sec // 2  # window size in chunks (2s per chunk)
        pooled_mean = np.zeros_like(features)
        pooled_max = np.zeros_like(features)
        for t in range(T):
            start = max(0, t - w)
            end = min(T, t + w + 1)
            pooled_mean[t] = features[start:end].mean(axis=0)
            pooled_max[t] = features[start:end].max(axis=0)
        context_features.extend([pooled_mean, pooled_max])

    # Delta & acceleration
    delta = np.diff(features, axis=0, prepend=features[[0]])
    accel = np.diff(delta, axis=0, prepend=delta[[0]])
    context_features.extend([delta, accel])

    return np.concatenate(context_features, axis=1)


def main():
    parser = argparse.ArgumentParser(description="Prepare multimodal training data from npz features")
    parser.add_argument("input_folder", help="Folder containing acoustic_emb_2s.npz, dva_emb_2s.npz, text_emb_2s.npz, heatmap.json")
    args = parser.parse_args()

    folder = Path(args.input_folder)

    # --- Load embeddings ---
    print("Loading acoustic embeddings…")
    acoustic_data = np.load(folder / "acoustic_emb_2s.npz")
    acoustic_emb = acoustic_data["embeddings"]
    acoustic_ts = acoustic_data["timestamps"]

    print("Loading DVA embeddings…")
    dva_data = np.load(folder / "dva_emb_2s.npz")
    dva_emb = dva_data["embeddings"]
    dva_ts = dva_data["timestamps"]

    print("Loading text embeddings…")
    text_data = np.load(folder / "text_emb_2s.npz")
    text_emb = text_data["embeddings"]
    text_ts = text_data["timestamps"]

    # --- Verify alignment ---
    T = min(len(acoustic_emb), len(dva_emb), len(text_emb))
    assert acoustic_emb.shape[0] == dva_emb.shape[0] == text_emb.shape[0], \
        f"Length mismatch: acoustic={acoustic_emb.shape[0]}, dva={dva_emb.shape[0]}, text={text_emb.shape[0]}"
    assert np.allclose(acoustic_ts[:T], dva_ts[:T]) and np.allclose(acoustic_ts[:T], text_ts[:T]), \
        "Timestamps not aligned across modalities"

    # --- Load labels ---
    print("Loading replay heatmap labels…")
    with open(folder / "heatmap.json") as f:
        replay_json = json.load(f)
    replay_data = np.array([entry["intensity"] for entry in replay_json], dtype=np.float32)
    replay_data = replay_data[:T]

    # --- Context expansion ---
    print("Applying temporal context expansion…")
    acoustic_ctx = add_temporal_context(acoustic_emb)
    dva_ctx = add_temporal_context(dva_emb)

    # --- Align features ---
    acoustic_ctx = acoustic_ctx[:T]
    dva_ctx = dva_ctx[:T]
    text_emb = text_emb[:T]

    # --- Positional encoding ---
    position = np.arange(T).reshape(-1, 1) / T  # normalized time index

    # --- Combine features ---
    print("Concatenating features…")
    X = np.concatenate([acoustic_ctx, dva_ctx, text_emb, position], axis=1)
    print(f"Before PCA → features: {X.shape}")

    # --- PCA ---
    print("Applying PCA reduction to 512 dims…")
    pca = PCA(n_components=512)
    X_reduced = pca.fit_transform(X)
    print(f"After PCA → features: {X_reduced.shape}")

    # --- Labels ---
    threshold = np.percentile(replay_data, 80)
    y = (replay_data > threshold).astype(np.float32)
    print(f"Labels: {y.shape}")

    # --- Save ---
    np.save(folder / "x.npy", X_reduced)
    np.save(folder / "y.npy", y)

    print("✅ Saved features to x.npy and labels to y.npy")


if __name__ == "__main__":
    main()

