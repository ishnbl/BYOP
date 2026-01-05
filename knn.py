import torch
import torch.nn as nn
import numpy as np
import os
import math
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report
from tqdm import tqdm

DATA_PATH = 'images_labelled_properly'
CHECKPOINT_PATH = 'checkpoints_mae2/check_points_mae5.pth'
MODEL_PATH = 'best_transformer_seq16.pth'
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

SEQ_LEN = 16
INPUT_DIM = 128
D_MODEL = 256
N_HEADS = 8
NUM_LAYERS = 3
DROPOUT = 0.1
NUM_CLASSES = 3

KNN_K = 20
KNN_CONFIDENCE = 0.75
KNN_BATCH_SIZE = 128
TEST_SAMPLE_SIZE = 10000

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
    def forward(self, x): return x + self.pe[:, :x.size(1)]

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
            nn.Linear(d_model, d_model // 2), nn.ReLU(),
            nn.Dropout(dropout), nn.Linear(d_model // 2, num_classes)
        )

    def extract_features(self, src):
        x = self.proj(src)
        x = self.pos_enc(x)
        x = self.transformer(x)
        features = x[:, -1, :]
        return torch.nn.functional.normalize(features, p=2, dim=1)

def get_vecs():
    print("generating vectors...")
    mae = ConvAutoencoder().to(DEVICE)
    ckpt = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    if 'encoder' in ckpt: mae.encoder_cnn.load_state_dict(ckpt['encoder']); mae.encoder_lin.load_state_dict(ckpt['encoder_head'])
    else: mae.load_state_dict(ckpt['full_model'], strict=False)
    mae.eval()
    
    vecs, labs = [], []
    files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith('.npz')])
    
    with torch.no_grad():
        for file in tqdm(files):
            d = np.load(os.path.join(DATA_PATH, file))
            imgs = torch.tensor(d['images'].reshape(-1, 1, 15, 12), dtype=torch.float32)
            imgs = (imgs / 127.5) - 1.0
            
            curr = []
            for i in range(0, len(imgs), 2048):
                batch = imgs[i:i+2048].to(DEVICE)
                curr.append(mae(batch).cpu().numpy())
            vecs.append(np.concatenate(curr))
            labs.append(d['labels'])
            
    return np.concatenate(vecs), np.concatenate(labs)

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

def get_tf_feats(model, data, seq_len):
    total = len(data)
    idx = np.arange(seq_len, total)
    n = len(idx)
    
    storage = torch.zeros((n, D_MODEL), dtype=torch.float16)
    
    ds = WindowDataset(data, idx, seq_len)
    loader = DataLoader(ds, batch_size=2048, shuffle=False, num_workers=2)
    
    model.eval()
    ptr = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            batch = batch.to(DEVICE)
            feats = model.extract_features(batch)
            bs = feats.shape[0]
            storage[ptr:ptr+bs] = feats.cpu().half()
            ptr += bs
            
    return storage, idx

def knn_predict(train_feats, train_labs, query_feats, k):
    n_q = query_feats.size(0)
    probs = torch.zeros(n_q, 3, dtype=torch.float)
    
    try:
        mem_f = train_feats.to(DEVICE).float()
        mem_l = train_labs.to(DEVICE)
    except:
        mem_f = train_feats.float()
        mem_l = train_labs
    
    for i in tqdm(range(0, n_q, KNN_BATCH_SIZE)):
        end = min(i + KNN_BATCH_SIZE, n_q)
        q_batch = query_feats[i:end].to(DEVICE).float()
        
        scores = torch.matmul(q_batch, mem_f.T)
        _, indices = torch.topk(scores, k, dim=1)
        
        neighbor_labs = mem_l[indices]
        votes = torch.nn.functional.one_hot(neighbor_labs, num_classes=3).float()
        probs[i:end] = votes.mean(dim=1).cpu()
        
    return probs.numpy()

if __name__ == "__main__":
    
    x_mae, y_raw = get_vecs()
    y_tensor = torch.tensor(y_raw, dtype=torch.long)
    
    print("loading transformer...")
    model = Transformer(INPUT_DIM, D_MODEL, N_HEADS, NUM_LAYERS, NUM_CLASSES, DROPOUT).to(DEVICE)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
    
    x_tf, idx = get_tf_feats(model, x_mae, SEQ_LEN)
    y_align = y_tensor[idx]
    
    del x_mae, model
    
    split = int(len(x_tf) * 0.85)
    x_mem, y_mem = x_tf[:split], y_align[:split]
    x_test, y_test = x_tf[split:], y_align[split:]
    
    perm = torch.randperm(len(x_test))[:TEST_SAMPLE_SIZE]
    x_test = x_test[perm]
    y_test = y_test[perm]
    
    probs = knn_predict(x_mem, y_mem, x_test, k=KNN_K)
    
    max_p = np.max(probs, axis=1)
    preds = np.argmax(probs, axis=1)
    
    mask = max_p >= KNN_CONFIDENCE
    final_preds = preds[mask]
    final_labs = y_test.numpy()[mask]
    
    if len(final_preds) > 0:
        print(f"\n{int(KNN_CONFIDENCE*100)}% confidence trades: {len(final_preds)}")
        print(classification_report(final_labs, final_preds, target_names=['Sell', 'Neutral', 'Buy']))
