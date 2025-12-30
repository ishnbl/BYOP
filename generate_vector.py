import torch
import torch.nn as nn
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader, ConcatDataset
from collections import defaultdict
from tqdm import tqdm
d_path = 'images_labelled_properly'
m_path = 'checkpoints/model.pth'
ofile = 'feature_database/vectors.npz'
order = ['BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'XRPUSDT']
split_ratio = 0.85  
batch_s = 2048
num_workers = 8
DEVICE = 'cuda'
os.makedirs(os.path.dirname(ofile))
class CNN(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32), 
            nn.LeakyReLU(0.1),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            nn.MaxPool2d(2),
            nn.Flatten(),
            nn.Linear(64 * 7 * 6, 128),
            nn.LeakyReLU(0.1),
            nn.Linear(128, num_classes)
        )
    
    def ext_featu(self, x):
        for i in range(10):
            x = self.cnn[i](x)
        return x


class Idata(Dataset):
    def __init__(self, samples, labels, coin):
        self.samples = samples
        self.labels = labels 
        self.coin = coin
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = self.samples[idx].reshape(1, 15, 12)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = (img_tensor / 127.5) - 1.0
        label = self.labels[idx] 
        return img_tensor, torch.tensor(label, dtype=torch.long)

def load_d_m(data_folder):
    files = [f for f in os.listdir(data_folder)]
    crypto_coin = defaultdict(lambda: {'samples': [], 'labels': []})
    for file in tqdm(files):
        name = file.split('_')[0]
        path = os.path.join(data_folder, file)
        data = np.load(path)
        crypto_coin[name]['samples'].append(data['images'])
        crypto_coin[name]['labels'].append(data['labels']) 
    final_map = {}
    for name in crypto_coin:
        final_map[name] = {
            'samples': np.concatenate(crypto_coin[name]['samples'], axis=0),
            'labels': np.concatenate(crypto_coin[name]['labels'], axis=0)
        }
    return final_map

def create_dset(dmap):
    dataset_list = []
    print("Building dataset")
    for name in order:
        full_samples = dmap[name]['samples']
        full_labels = dmap[name]['labels']
        cut_idx = int(len(full_samples) * split_ratio)
        samples_slice = full_samples[:cut_idx]
        labels_slice = full_labels[:cut_idx]
        ds = Idata(samples_slice, labels_slice, name)
        dataset_list.append(ds)
    return ConcatDataset(dataset_list)

def run():
    dmap = load_d_m(d_path)
    dataset = create_dset(dmap)
    loader = DataLoader(dataset, batch_size=batch_s, shuffle=False, num_workers=num_workers)
    model = CNN().to(DEVICE)  
    checkpoint = torch.load(m_path, map_location=DEVICE)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    model.eval()
    
    print(f"running model")
    vecs = []
    vec_labels = []
    
    with torch.no_grad():
        for imgs, labels in tqdm(loader, desc="calculating embeddings"):
            imgs = imgs.to(DEVICE)
            features = model.ext_featu(imgs)
            vecs.append(features.cpu().numpy())
            vec_labels.append(labels.numpy())
            
    print("logg")
    fvecs = np.concatenate(vecs, axis=0).astype(np.float32)
    flabels = np.concatenate(vec_labels, axis=0).astype(np.int64)
    np.savez_compressed(
        ofile, 
        vectors=fvecs, 
        labels=flabels, 
        order=order
    )
    print("saved")


if __name__ == "__main__":
    run()