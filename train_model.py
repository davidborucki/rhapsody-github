import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset, WeightedRandomSampler
from sklearn.metrics import precision_recall_fscore_support

# -------------------
# Step 1: Load Data
# -------------------
def load_data(folder: Path):
    X = np.load(folder / "x.npy")
    y = np.load(folder / "y.npy")
    print(f"Loaded {folder.name}: features {X.shape}, labels {y.shape}")
    return X, y

# -------------------
# Step 2: Dataset
# -------------------
class HighlightDataset(Dataset):
    def __init__(self, features, labels, seq_len=60):
        self.features = torch.as_tensor(features, dtype=torch.float32)
        self.labels = torch.as_tensor(labels, dtype=torch.float32)
        self.seq_len = seq_len

    def __len__(self):
        return len(self.features) - self.seq_len + 1

    def __getitem__(self, idx):
        x = self.features[idx:idx + self.seq_len]
        y = self.labels[idx:idx + self.seq_len]
        return x, y

# ------------------
# Step 3: Model
# ------------------
class HighlightTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_encoder = nn.Embedding(2000, hidden_dim)

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

    def forward(self, x):
        B, T, _ = x.shape
        x = self.input_proj(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, -1)
        x = x + self.pos_encoder(pos)
        x = self.transformer(x)
        logits = self.classifier(x).squeeze(-1)
        return logits

# -------------------
# Step 4: Helpers
# -------------------
@torch.no_grad()
def eval_with_threshold_sweep(model, loader, device="cpu", grid=None):
    if grid is None:
        grid = np.linspace(0.02, 0.8, 40)

    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu().numpy().ravel()
        probs.append(p)
        labels.append(y.numpy().ravel())

    probs = np.concatenate(probs) if probs else np.array([])
    labels = np.concatenate(labels) if labels else np.array([])

    if probs.size == 0:
        return {"thr": 0.5, "P": 0.0, "R": 0.0, "F1": 0.0}, 0.0, 0.0

    best = {"thr": 0.5, "P": 0.0, "R": 0.0, "F1": 0.0}
    for t in grid:
        pred = (probs >= t).astype(np.int32)
        P, R, F1, _ = precision_recall_fscore_support(labels, pred, average="binary", zero_division=0)
        if F1 > best["F1"]:
            best = {"thr": float(t), "P": float(P), "R": float(R), "F1": float(F1)}
    return best, float(probs.mean()), float(labels.mean())

def train_epoch(model, loader, optimizer, criterion, device="cpu"):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total += loss.item()
    return total / max(1, len(loader))

# -------------------
# Step 5: Main
# -------------------
def main():
    parser = argparse.ArgumentParser(description="Train highlight transformer on dataset folder")
    parser.add_argument("data_folder", help="Path to main data folder containing subfolders (one per episode)")
    parser.add_argument("--epochs", type=int, default=24)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=7e-5)
    parser.add_argument("--seq-len", type=int, default=60)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    data_root = Path(args.data_folder)
    subfolders = sorted([f for f in data_root.iterdir() if f.is_dir()])

    print(f"Found {len(subfolders)} episode folders in {data_root}")

    # Load all episodes
    datasets = []
    input_dim = None
    for f in subfolders:
        X, Y = load_data(f)
        if input_dim is None:
            input_dim = X.shape[1]
        datasets.append(HighlightDataset(X, Y, seq_len=args.seq_len))

    full_dataset = ConcatDataset(datasets)
    n_total = len(full_dataset)
    n_val = int(0.15 * n_total)
    n_train = n_total - n_val
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [n_train, n_val])

    print(f"Split: {n_train} train windows, {n_val} validation windows")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size)

    # Class balance
    all_labels = torch.cat([d.labels for d in datasets])
    pos = float(all_labels.sum().item())
    total = float(len(all_labels))
    neg = total - pos
    imbalance_ratio = neg / max(1.0, pos)
    pos_weight = torch.tensor(np.sqrt(imbalance_ratio), dtype=torch.float32, device=device)
    print(f"Class balance: {int(pos)} positives, {int(neg)} negatives, pos_weight={pos_weight.item():.2f}")

    # Model
    model = HighlightTransformer(input_dim=input_dim).to(device)

    # Bias initialization
    p = max(1e-6, pos / total)
    logit_prior = float(np.log(p / (1 - p)))
    with torch.no_grad():
        model.classifier[-1].bias.fill_(logit_prior)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training
    best_f1, best_thr, best_state = -1.0, 0.5, None
    patience, wait = 2, 0

    for epoch in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, criterion, device)
        best, prob_mean, y_rate = eval_with_threshold_sweep(model, val_loader, device)
        print(f"Epoch {epoch}: TrainLoss={loss:.4f} | "
              f"Val base_rate={y_rate:.3f} prob_mean={prob_mean:.3f} | "
              f"BestThr={best['thr']:.3f} P={best['P']:.3f} R={best['R']:.3f} F1={best['F1']:.3f}")

        if best["F1"] > best_f1 + 1e-4:
            best_f1, best_thr = best["F1"], best["thr"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait > patience:
                print("Early stopping.")
                break

    if best_state:
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), "highlight_model.pt")
    with open("threshold.json", "w") as fh:
        json.dump({"best_threshold": best_thr, "best_f1": best_f1}, fh, indent=2)

    print(f"✅ Model saved -> highlight_model.pt (best_thr={best_thr:.3f}, best_f1={best_f1:.3f})")

if __name__ == "__main__":
    main()

