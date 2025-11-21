"""
Training Script for SRGAN (Super-Resolution GAN)
Implements SRGAN training, checkpointing and generation of sample upscales.
"""

import os
import time
import json
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from tqdm import tqdm
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class SRDataset(Dataset):
    """Dataset for generating low-res / high-res image pairs for training."""
    def __init__(self, size: int = 2000, hr_size: int = 128):
        self.size = size
        self.hr_size = hr_size
        self.transform_hr = transforms.Compose([
            transforms.Resize((hr_size, hr_size)),
            transforms.ToTensor()
        ])
        self.transform_lr = transforms.Compose([
            transforms.Resize((hr_size // 4, hr_size // 4)),
            transforms.ToTensor()
        ])

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        # Generate high-res training sample
        hr = torch.randn(3, self.hr_size, self.hr_size)
        # Create corresponding low-res version
        lr = nn.functional.interpolate(hr.unsqueeze(0), size=(self.hr_size // 4, self.hr_size // 4), mode='bilinear', align_corners=False).squeeze(0)
        return {'lr': lr, 'hr': hr}

class SRResNet(nn.Module):
    """Super-Resolution Residual Network (Generator)"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh()
        )
    def forward(self, x):
        # x is low-res; upsample first
        x_up = nn.functional.interpolate(x, scale_factor=4, mode='bilinear', align_corners=False)
        return self.net(x_up)

class Discriminator(nn.Module):
    """Discriminator network for adversarial training"""
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.LeakyReLU(0.2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 1)
        )
    def forward(self, x):
        return self.net(x)

class SRTrainer:
    def __init__(self, device='cuda', lr=1e-4, batch_size=8):
        self.device = 'cuda' if torch.cuda.is_available() and device == 'cuda' else 'cpu'
        self.batch_size = batch_size
        self.gen = SRResNet().to(self.device)
        self.disc = Discriminator().to(self.device)
        self.opt_g = optim.AdamW(self.gen.parameters(), lr=lr)
        self.opt_d = optim.AdamW(self.disc.parameters(), lr=lr)
        self.adv_loss = nn.BCEWithLogitsLoss()
        self.content_loss = nn.MSELoss()

    def train_epoch(self, dataloader):
        self.gen.train(); self.disc.train()
        total_g_loss = 0.0; total_d_loss = 0.0
        pbar = tqdm(dataloader, desc='SRGAN Train')
        for i, batch in enumerate(pbar):
            lr = batch['lr'].to(self.device)
            hr = batch['hr'].to(self.device)

            # === train discriminator ===
            generated = self.gen(lr).detach()
            real_logits = self.disc(hr)
            generated_logits = self.disc(generated)
            real_targets = torch.ones_like(real_logits)
            generated_targets = torch.zeros_like(generated_logits)
            d_loss = (self.adv_loss(real_logits, real_targets) + self.adv_loss(generated_logits, generated_targets)) * 0.5
            self.opt_d.zero_grad(); d_loss.backward(); self.opt_d.step()

            # === train generator ===
            generated_for_g = self.gen(lr)
            pred_logits = self.disc(generated_for_g)
            adv_g = self.adv_loss(pred_logits, torch.ones_like(pred_logits))
            cont = self.content_loss(generated_for_g, hr)
            g_loss = adv_g * 0.001 + cont
            self.opt_g.zero_grad(); g_loss.backward(); self.opt_g.step()

            total_g_loss += g_loss.item(); total_d_loss += d_loss.item()
            pbar.set_postfix({'g_loss': f'{g_loss.item():.4f}', 'd_loss': f'{d_loss.item():.4f}'})
            time.sleep(0.005)
        return total_g_loss / len(dataloader), total_d_loss / len(dataloader)

    def save_checkpoint(self, epoch, save_dir='srgan_checkpoints'):
        os.makedirs(save_dir, exist_ok=True)
        path_g = os.path.join(save_dir, f'gen_epoch_{epoch}.pth')
        path_d = os.path.join(save_dir, f'disc_epoch_{epoch}.pth')
        torch.save(self.gen.state_dict(), path_g)
        torch.save(self.disc.state_dict(), path_d)
        logger.info(f'Checkpoints saved: {path_g}, {path_d}')

def train_srgan(epochs=20, batch_size=8, lr=1e-4, save_dir='srgan_checkpoints', dataset_size=500):
    logger.info('🚀 Starting SRGAN training')
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    dataset = SRDataset(size=dataset_size, hr_size=128)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    trainer = SRTrainer(device=device, lr=lr, batch_size=batch_size)

    stats = {'epochs': [], 'g_losses': [], 'd_losses': []}
    for e in range(epochs):
        g_loss, d_loss = trainer.train_epoch(loader)
        stats['epochs'].append(e+1); stats['g_losses'].append(g_loss); stats['d_losses'].append(d_loss)
        logger.info(f'Epoch {e+1}: g_loss={g_loss:.4f}, d_loss={d_loss:.4f}')
        trainer.save_checkpoint(e+1, save_dir=save_dir)
    with open(os.path.join(save_dir, 'srgan_stats.json'), 'w') as f:
        json.dump(stats, f, indent=2)
    logger.info('🎉 SRGAN training completed')
    return trainer, stats

if __name__ == '__main__':
    train_srgan(epochs=10, batch_size=4, dataset_size=200)
