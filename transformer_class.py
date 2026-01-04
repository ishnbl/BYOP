import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
import math
from sklearn.metrics import classification_report

data_path = 'images_labelled_properly'
ckpt_path = 'checkpoints_mae2/check_points_mae5.pth'
save_path = 'best_transformer_seq16.pth'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

seq_len = 16
inp_dim = 128
model_dim = 256
n_heads = 8
n_layers = 3
dropout = 0.1
n_classes = 3

batch_sz = 128
n_epochs = 2
lr = 5e-5


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1)
        )
        self.lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 6, 128),
        )

    def forward(self, x):
        x = self.conv(x)
        return self.lin(x)


def gen_embeddings():
    enc = Encoder().to(device)
    ckpt = torch.load(ckpt_path, map_location=device)
    enc.load_state_dict(ckpt['full_model'], strict=False)
    enc.eval()

    vecs = []
    labels = []

    files = sorted([f for f in os.listdir(data_path)])

    print("Encoding images...")
    with torch.no_grad():
        for fi, fname in enumerate(files):
            data = np.load(os.path.join(data_path, fname))
            imgs = data['images']
            lbls = data['labels']

            imgs_t = torch.tensor(imgs.reshape(-1, 1, 15, 12), dtype=torch.float32)
            imgs_t = (imgs_t / 127.5) - 1.0

            for i in range(0, len(imgs_t), 2048):
                batch = imgs_t[i:i+2048].to(device)
                emb = enc(batch)
                vecs.append(emb.cpu().numpy())
            labels.append(lbls)

    X = np.concatenate(vecs, axis=0)
    y = np.concatenate(labels, axis=0)
    print("Generated vectors.")
    return X, y


class WindowDataset(Dataset):
    def __init__(self, features, labels, indices, seq_len):
        self.features = features
        self.labels = labels
        self.indices = indices
        self.seq_len = seq_len

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        tgt_idx = self.indices[idx]
        start = tgt_idx - self.seq_len
        end = tgt_idx

        x_seq = self.features[start:end]
        y_lbl = self.labels[tgt_idx]

        return torch.tensor(x_seq, dtype=torch.float32), torch.tensor(y_lbl, dtype=torch.long)


class PosEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(pos * div_term)
        pe[:, 1::2] = torch.cos(pos * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]


class Transformer(nn.Module):
    def __init__(self, inp_d, model_d, n_h, n_l, n_c, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(inp_d, model_d)
        self.pos_enc = PosEncoding(model_d)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=model_d,
            nhead=n_h,
            dim_feedforward=model_d * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=n_l)

        self.head = nn.Sequential(
            nn.Linear(model_d, model_d // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(model_d // 2, n_c)
        )

    def forward(self, src):
        x = self.proj(src)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return self.head(x[:, -1, :])


def train_epoch(model, loader, loss_fn, optimizer, epoch):
    model.train()
    tot_loss = 0
    tot_correct = 0
    tot_samples = 0

    for step, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        logits = model(X)
        loss = loss_fn(logits, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        tot_loss += loss.item()
        preds = torch.argmax(logits, dim=1)
        tot_correct += (preds == y).sum().item()
        tot_samples += y.size(0)

        if (step + 1) % 1000 == 0:
            print(f"epoch {epoch+1}, step {step+1}/{len(loader)}, loss {loss.item():.4f}")

    return tot_loss / len(loader), 100 * tot_correct / tot_samples


def validate(model, loader, loss_fn):
    model.eval()
    tot_loss = 0
    tot_correct = 0
    tot_samples = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for X, y in loader:
            X, y = X.to(device), y.to(device)
            logits = model(X)
            loss = loss_fn(logits, y)

            tot_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            tot_correct += (preds == y).sum().item()
            tot_samples += y.size(0)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    return tot_loss / len(loader), 100 * tot_correct / tot_samples, all_preds, all_labels


if __name__ == "__main__":
    X_raw, y_raw = gen_embeddings()

    N = len(X_raw)
    split1 = int(N * 0.70)
    split2 = int(N * 0.85)

    train_idx = np.arange(seq_len, split1)
    val_idx = np.arange(split2, N)

    train_ds = WindowDataset(X_raw, y_raw, train_idx, seq_len)
    val_ds = WindowDataset(X_raw, y_raw, val_idx, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_sz, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_sz, shuffle=False, num_workers=0)

    model = Transformer(inp_dim, model_dim, n_heads, n_layers, n_classes, dropout).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    print("Training transformer")

    for epoch in range(n_epochs):
        train_loss, train_acc = train_epoch(model, train_loader, loss_fn, optimizer, epoch)
        val_loss, val_acc, _, _ = validate(model, val_loader, loss_fn)
        torch.save(model.state_dict(), save_path)

    print("Results")
    model.load_state_dict(torch.load(save_path))
    _, _, preds, labels = validate(model, val_loader, loss_fn)

    print(classification_report(labels, preds, target_names=['Sell', 'Neutral', 'Buy'], zero_division=0))
