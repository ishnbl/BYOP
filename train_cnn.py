import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import numpy as np
import os
from tqdm import tqdm
from collections import defaultdict


data_path = 'images_labelled_properly'
window = 12
bs = 2048
lr = 5e-5  
epochs = 25
device = 'cuda'
save_path = 'checkpoints'

train_r = 0.70
val_r = 0.15
test_r = 0.15

os.makedirs(save_path, exist_ok=True)


class CoinDS(Dataset):
    def __init__(self, s, l, name):
        self.s = s
        self.l = l
        self.name = name
    
    def __len__(self):
        return len(self.s)
    
    def __getitem__(self, idx):
        img = self.s[idx].reshape(1, 15, 12)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = (img_tensor / 127.5) - 1.0
        label = self.l[idx]
        return img_tensor, torch.tensor(label, dtype=torch.long)


def load_data(folder):
    files = sorted([f for f in os.listdir(folder) if f.endswith('.npz')])

    coins = defaultdict(lambda: {'samples': [], 'labels': []})
    
    print("loading files")
    for f in files:
        name = f.split('_')[0]
        fp = os.path.join(folder, f)
        d = np.load(fp)
        coins[name]['samples'].append(d['images'])
        coins[name]['labels'].append(d['labels'])
    
    for name in coins:
        coins[name]['samples'] = np.concatenate(coins[name]['samples'], axis=0)
        coins[name]['labels'] = np.concatenate(coins[name]['labels'], axis=0)
    
    return dict(coins)


def split_data(data):
    train_datasets = []
    val_datasets = []
    test_datasets = []
    
    for name in data:
        samples = data[name]['samples']
        labels = data[name]['labels']
        n = len(samples)
        
        idx1 = int(train_r * n)
        idx2 = idx1 + int(val_r * n)
        
        train_datasets.append(CoinDS(samples[:idx1], labels[:idx1], name))
        val_datasets.append(CoinDS(samples[idx1:idx2], labels[idx1:idx2], name))
        test_datasets.append(CoinDS(samples[idx2:], labels[idx2:], name))

    return ConcatDataset(train_datasets), ConcatDataset(val_datasets), ConcatDataset(test_datasets)


class SimpleCNN(nn.Module):
    def __init__(self, nc=3):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.LeakyReLU(0.1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.LeakyReLU(0.1)
        self.pool = nn.MaxPool2d(2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 7 * 6, 128)
        self.relu3 = nn.LeakyReLU(0.1)
        self.fc2 = nn.Linear(128, nc)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x


def eval_test(model, loader):
    model.eval()
    preds = []
    labels = []
    
    print("testing")
    with torch.no_grad():
        for imgs, lbls in tqdm(loader):
            imgs = imgs.to(device)
            out = model(imgs)
            _, p = torch.max(out, 1)
            preds.extend(p.cpu().numpy())
            labels.extend(lbls.numpy())
    
    preds = np.array(preds)
    labels = np.array(labels)

    num_chunks = int(test_r / 0.02)
    if num_chunks < 1: 
        num_chunks = 1
    chunk_size = len(preds) // num_chunks

def train():
    print("training .............. less goo")    
    data = load_data(data_path)
    train_ds, val_ds, test_ds = split_data(data)
    
    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=8)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=8)
    test_loader = DataLoader(test_ds, batch_size=bs, shuffle=False, num_workers=8)
    
    model = SimpleCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    
    best_acc = 0
    
    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc=f"epoch {ep+1}/{epochs}")
        
        for imgs, lbls in pbar:
            imgs, lbls = imgs.to(device), lbls.to(device)
            
            optimizer.zero_grad()
            out = model(imgs)
            loss = loss_fn(out, lbls)

            loss.backward()
            optimizer.step()
            
            _, p = torch.max(out, 1)
            correct += (p == lbls).sum().item()
            total += lbls.size(0)
            total_loss += loss.item()
            
            train_acc = 100 * correct / total
            pbar.set_postfix({'loss': f"{total_loss/(pbar.n+1):.4f}", 'acc': f"{train_acc:.2f}%"})
            
        model.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for imgs, lbls in val_loader:
                imgs, lbls = imgs.to(device), lbls.to(device)
                out = model(imgs)
                _, p = torch.max(out, 1)
                val_total += lbls.size(0)
                val_correct += (p == lbls).sum().item()
        
        val_acc = 100 * val_correct / val_total

        train_acc = 100 * correct / total
        print(f"train: {train_acc}% val: {val_acc}%")
        
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(save_path, 'best_short_term_model.pth'))
            print(f"saved best")
    
    best_model = SimpleCNN().to(device)
    best_model.load_state_dict(torch.load(os.path.join(save_path, 'best_short_term_model.pth')))
    
    eval_test(best_model, test_loader)


if __name__ == "__main__":
    train()
