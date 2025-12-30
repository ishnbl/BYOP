import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import os
from tqdm import tqdm


d_path = 'images_labelled_properly'  
b_size = 2048
lr = 1e-4  
split_ratio = 0.7  
DEVICE = torch.device('cuda')
model_save = 'checkpoints_mae2'

os.makedirs(model_save, exist_ok=True)


class MaskedInstrumentDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img = self.samples[idx].reshape(1, 15, 12)
        img_tensor = torch.tensor(img, dtype=torch.float32)
        img_tensor = (img_tensor / 127.5) - 1.0
        
        masked_img = img_tensor.clone()
        for _ in range(np.random.randint(1, 4)):
            h_start = np.random.randint(0, 15 - 3)
            w_start = np.random.randint(0, 12 - 3)
            masked_img[:, h_start:h_start+3, w_start:w_start+3] = 0.0 
            
        return masked_img, img_tensor  

def load_data_split(data_folder):
    all_samples = []
    print(f"loading data")
    files = sorted([f for f in os.listdir(data_folder) if f.endswith('.npz')])
    for file in tqdm(files):
        data = np.load(os.path.join(data_folder, file))
        all_samples.append(data['images'])
    
    full_data = np.concatenate(all_samples, axis=0)
    split_idx = int(full_data.shape[0] * split_ratio)
    return full_data[:split_idx]


class ConvAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.encoder_cnn = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            

            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1), 
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1)
        )
        
        self.encoder_lin = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 8 * 6, 128), 
        )
        
        self.decoder_lin = nn.Sequential(
            nn.Linear(128, 64 * 8 * 6),
            nn.LeakyReLU(0.1)
        )
        
        self.decoder_cnn = nn.Sequential(
            nn.ConvTranspose2d(64, 32, kernel_size=3, stride=2, padding=1, output_padding=(0, 1)),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.1),
            
            nn.Conv2d(32, 1, kernel_size=3, padding=1),
            nn.Tanh() 
        )

    def forward(self, x):
        x = self.encoder_cnn(x)
        latent = self.encoder_lin(x) 
        
        x = self.decoder_lin(latent)
        x = x.view(-1, 64, 8, 6)
        decoded = self.decoder_cnn(x)
        
        return latent, decoded


def train_mae():
    print("Training MAE")
    train_data = load_data_split(d_path)
    dataset = MaskedInstrumentDataset(train_data)
    
    loader = DataLoader(
        dataset, 
        batch_size=b_size, 
        shuffle=True, 
        num_workers=8, 
        pin_memory=True
    )

    model = ConvAutoencoder().to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    for epoch in range(5):
        model.train()
        running_loss = 0.0
        pbar = tqdm(loader, desc=f"Epoch {epoch+1}/5")
        
        for masked_imgs, clean_targets in pbar:
            masked_imgs, clean_targets = masked_imgs.to(DEVICE), clean_targets.to(DEVICE)
            
            optimizer.zero_grad()
            _, reconstruction = model(masked_imgs)
            loss = criterion(reconstruction, clean_targets)
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            pbar.set_postfix({'MSE Loss': f"{running_loss/(pbar.n+1):.6f}"})
            
        save_dict = {
            'encoder': model.encoder_cnn.state_dict(),
            'encoder_head': model.encoder_lin.state_dict(),
            'full_model': model.state_dict()
        }
        torch.save(save_dict, os.path.join(model_save, f'check_points_mae{epoch+1}.pth'))

if __name__ == "__main__":
    train_mae()
