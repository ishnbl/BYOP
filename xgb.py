import torch
import torch.nn as nn
import numpy as np
import xgboost as xgb
import os
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

DATA_PATH = 'images_labelled_properly'
CHECKPOINT_PATH = 'checkpoints_mae2/check_points_mae5.pth'
MODEL_PATH = 'best_transformer_seq16.pth'
XGB_SAVE_PATH = 'best_xgboost_transformer2.json'

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN = 16
INPUT_DIM = 128
D_MODEL = 256
N_HEADS = 8
NUM_LAYERS = 3
DROPOUT = 0.1
NUM_CLASSES = 3

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64), nn.LeakyReLU(0.1),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128), nn.LeakyReLU(0.1)
        )
        self.encoder_lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 6, 128), 
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        return self.encoder_lin(x)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class Transformer(nn.Module):
    def __init__(self, input_dim, d_model, nhead, num_layers, num_classes, dropout=0.1):
        super().__init__()
        self.proj = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model*4, 
            dropout=dropout, batch_first=True, norm_first=True 
        )
        self.transformer = nn.TransformerEncoder(encoder_layers, num_layers=num_layers)
        
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes)
        )

    def extract_features(self, src):
        x = self.proj(src)
        x = self.pos_enc(x)
        x = self.transformer(x)
        return x[:, -1, :]
    
    def forward(self, src):
        x = self.extract_features(src)
        return self.head(x)

def get_embeddings():
    print("generating embeddings...")
    mae_model = ConvAutoencoder().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    
    if 'encoder' in ckpt:
        mae_model.encoder_cnn.load_state_dict(ckpt['encoder'])
        mae_model.encoder_lin.load_state_dict(ckpt['encoder_head'])
    else:
        mae_model.load_state_dict(ckpt['full_model'], strict=False)
    mae_model.eval()
    
    vecs = []
    labs = []
    
    files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith('.npz')])
    
    with torch.no_grad():
        for f in tqdm(files):
            d = np.load(os.path.join(DATA_PATH, f))
            img = d['images']
            lbl = d['labels']
            
            img_t = torch.tensor(img.reshape(-1, 1, 15, 12), dtype=torch.float32)
            img_t = (img_t / 127.5) - 1.0
            
            batch_vecs = []
            for i in range(0, len(img_t), 2048):
                b = img_t[i:i+2048].to(DEVICE)
                v = mae_model(b)
                batch_vecs.append(v.cpu().numpy())
            
            vecs.append(np.concatenate(batch_vecs, axis=0))
            labs.append(lbl)
            
    return np.concatenate(vecs, axis=0), np.concatenate(labs, axis=0)

class WindowDataset(Dataset):
    def __init__(self, features, indices, seq_len):
        self.features = features
        self.indices = indices
        self.seq_len = seq_len
    def __len__(self): return len(self.indices)
    def __getitem__(self, idx):
        target_idx = self.indices[idx]
        x_seq = self.features[target_idx - self.seq_len : target_idx]
        return torch.tensor(x_seq, dtype=torch.float32)

def get_transformer_feats(model, data, seq_len):
    total = len(data)
    idx = np.arange(seq_len, total)
    n = len(idx)
    
    out = np.zeros((n, D_MODEL), dtype=np.float32)
    
    ds = WindowDataset(data, idx, seq_len)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=2)
    
    model.eval()
    
    p = 0
    with torch.no_grad():
        for x in tqdm(loader):
            x = x.to(DEVICE)
            f = model.extract_features(x).cpu().numpy()
            
            bs = f.shape[0]
            out[p : p + bs] = f
            p += bs
            
    return out, idx

if __name__ == "__main__":
    
    x, y = get_embeddings()
    
    print("loading transformer...")
    tfm = Transformer(INPUT_DIM, D_MODEL, N_HEADS, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(DEVICE)
    tfm.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    x_t, idx = get_transformer_feats(tfm, x, SEQ_LEN)
    y_t = y[idx]
    
    del x
    del tfm
    
    total = len(x_t)
    tr_end = int(total * 0.70)
    val_start = int(total * 0.85)
    
    x_tr, y_tr = x_t[:tr_end], y_t[:tr_end]
    x_v, y_v = x_t[val_start:], y_t[val_start:]
    
    print(f"train: {len(x_tr)}, val: {len(x_v)}")
    
    xgb_model = xgb.XGBClassifier(
        objective='multi:softprob',
        num_class=3,
        tree_method='hist',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        eval_metric=['mlogloss', 'merror'],
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        n_estimators=1000,
        early_stopping_rounds=50
    )
    
    xgb_model.fit(x_tr, y_tr, eval_set=[(x_v, y_v)], verbose=100)
    
    probs = xgb_model.predict_proba(x_v)
    preds = np.argmax(probs, axis=1)
    
    max_p = np.max(probs, axis=1)
    m = max_p > 0.60
    
    if np.sum(m) > 0:
        p_high = preds[m]
        y_high = y_v[m]
        
        print(f"\n60% confidence trades: {len(p_high)}")
        print(classification_report(y_high, p_high, target_names=['Sell', 'Neutral', 'Buy']))
    
    xgb_model.get_booster().save_model(XGB_SAVE_PATH)
    print(f"saved model")
