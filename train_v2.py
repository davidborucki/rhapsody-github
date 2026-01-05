import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from sklearn.metrics import precision_recall_fscore_support

# -------------------
# Dataset
# -------------------
class HighlightDataset(Dataset):
    def __init__(self, X, y, seq_len=60):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float()
        self.seq_len = seq_len

    def __len__(self):
        return len(self.X) - self.seq_len + 1

    def __getitem__(self, idx):
        return (
            self.X[idx : idx + self.seq_len],
            self.y[idx : idx + self.seq_len],
        )

# -------------------
# Model
# -------------------
class HighlightTransformer(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=4, num_heads=8, dropout=0.3):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, hidden_dim)
        self.pos_emb = nn.Embedding(2048, hidden_dim)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers)

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        B, T, _ = x.shape
        h = self.input_proj(x)
        pos = torch.arange(T, device=x.device).unsqueeze(0).expand(B, T)
        h = h + self.pos_emb(pos)
        h = self.encoder(h)
        return self.head(h).squeeze(-1)

# -------------------
# Eval helpers
# -------------------
@torch.no_grad()
def eval_with_threshold_sweep(model, loader, device):
    model.eval()
    probs, labels = [], []

    for x, y in loader:
        x = x.to(device)
        p = torch.sigmoid(model(x)).cpu().numpy().ravel()
        probs.append(p)
        labels.append(y.numpy().ravel())

    probs = np.concatenate(probs)
    labels = np.concatenate(labels)

    best = {"thr": 0.5, "P": 0, "R": 0, "F1": 0}
    for t in np.linspace(0.05, 0.9, 40):
        pred = (probs >= t).astype(int)
        P, R, F1, _ = precision_recall_fscore_support(
            labels, pred, average="binary", zero_division=0
        )
        if F1 > best["F1"]:
            best = {"thr": float(t), "P": float(P), "R": float(R), "F1": float(F1)}
    return best

def train_epoch(model, loader, opt, loss_fn, device):
    model.train()
    total = 0.0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        opt.zero_grad()
        loss = loss_fn(model(x), y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        total += loss.item()
    return total / len(loader)

# -------------------
# Main
# -------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data")
    parser.add_argument("--pca", default="pca_global_512.joblib")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=60)
    parser.add_argument("--lr", type=float, default=7e-5)
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load PCA
    input_dim = None

    datasets = []
    all_labels = []

    for ep in sorted(Path(args.data).iterdir()):
        if not ep.is_dir():
            continue

        x_path = ep / "new_x.npy"
        y_path = ep / "new_y.npy"
        if not x_path.exists():
            continue

        X = np.load(x_path)
        y = np.load(y_path)

        if input_dim is None:
            input_dim = X.shape[1]

        # 🔑 apply global PCA ONCE

        datasets.append(HighlightDataset(X, y, args.seq_len))
        all_labels.append(torch.from_numpy(y))

        print(f"Loaded {ep.name}: {X.shape}")

    full = ConcatDataset(datasets)

    n_val = int(0.15 * len(full))
    n_train = len(full) - n_val
    train_ds, val_ds = torch.utils.data.random_split(full, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    # class balance
    labels = torch.cat(all_labels)
    pos = labels.sum().item()
    neg = len(labels) - pos
    pos_weight = torch.tensor(np.sqrt(neg / max(pos, 1)), device=device)

    model = HighlightTransformer(input_dim).to(device)

    # bias init
    p = max(pos / len(labels), 1e-6)
    with torch.no_grad():
        model.head[-1].bias.fill_(np.log(p / (1 - p)))

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_f1, best_state, best_thr = 0, None, 0.5
    patience, wait = 3, 0

    for e in range(1, args.epochs + 1):
        loss = train_epoch(model, train_loader, opt, loss_fn, device)
        best = eval_with_threshold_sweep(model, val_loader, device)

        print(
            f"Epoch {e}: loss={loss:.4f} "
            f"P={best['P']:.3f} R={best['R']:.3f} F1={best['F1']:.3f} thr={best['thr']:.2f}"
        )

        if best["F1"] > best_f1:
            best_f1 = best["F1"]
            best_thr = best["thr"]
            best_state = {k: v.cpu() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait > patience:
                break

    model.load_state_dict(best_state)
    torch.save(model.state_dict(), "highlight_model.pt")

    with open("threshold.json", "w") as f:
        json.dump({"threshold": best_thr, "f1": best_f1}, f, indent=2)

    print("✅ Training complete")

if __name__ == "__main__":
    main()
