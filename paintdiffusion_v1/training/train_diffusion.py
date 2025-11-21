import os
import time
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageDataset(Dataset):
    
    def __init__(self, size: int = 10000, image_size: int = 512):
        self.size = size
        self.image_size = image_size
        self.prompts = [
            "a beautiful landscape painting",
            "cyberpunk city at sunset",
            "portrait of a cat in renaissance style",
            "abstract geometric patterns",
            "watercolor flowers in a vase",
            "futuristic spaceship design",
            "vintage car in black and white",
            "magical forest with glowing mushrooms",
            "architectural blueprint drawing",
            "impressionist style street scene"
        ]
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        # Generate  image (random noise)
        _image = torch.randn(3, self.image_size, self.image_size)
        
        # Generate  prompt
        prompt = random.choice(self.prompts)
        
        # Add some variation
        if random.random() > 0.5:
            prompt += f", {random.choice(['highly detailed', 'masterpiece', 'trending on artstation', '8k resolution'])}"
        
        return {
            'image': _image,
            'prompt': prompt,
            'pixel_values': _image
        }

class DiffusionModel(nn.Module):
   
    
    def __init__(self, channels: int = 4, text_embed_dim: int = 768):
        super().__init__()
        self.channels = channels
        self.text_embed_dim = text_embed_dim
        
        #  UNet-like architecture
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU()
        )
        
        # Text conditioning
        self.text_proj = nn.Linear(text_embed_dim, 256)
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, channels, 3, padding=1)
        )
        
        # Time embedding ()
        self.time_embed = nn.Sequential(
            nn.Linear(1, 128),
            nn.ReLU(),
            nn.Linear(128, 256)
        )
    
    def forward(self, x, timesteps, text_embeddings):
        #  forward pass
        batch_size = x.shape[0]
        
        # Encode input
        encoded = self.encoder(x)
        
        # Add time conditioning
        time_emb = self.time_embed(timesteps.float().unsqueeze(-1))
        time_emb = time_emb.view(batch_size, -1, 1, 1)
        encoded = encoded + time_emb
        
        # Add text conditioning (simplified)
        text_emb = self.text_proj(text_embeddings.mean(dim=1))
        text_emb = text_emb.view(batch_size, -1, 1, 1)
        encoded = encoded + text_emb
        
        # Decode
        output = self.decoder(encoded)
        
        return output

class DiffusionTrainer:
    
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 1e-4,
                 batch_size: int = 8):
        
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
        
        # Loss function
        self.criterion = nn.MSELoss()
        
        #  text encoder
        self.text_encoder = lambda x: torch.randn(len(x), 77, 768).to(device)
        
        # Training stats
        self.train_losses = []
        self.val_losses = []
        self.best_loss = float('inf')
        
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc="Training")
        
        for batch_idx, batch in enumerate(progress_bar):
            # Move to device
            images = batch['pixel_values'].to(self.device)
            prompts = batch['prompt']
            
            # Generate  text embeddings
            text_embeddings = self.text_encoder(prompts)
            
            # Add noise ( diffusion process)
            noise = torch.randn_like(images)
            timesteps = torch.randint(0, 1000, (images.shape[0],)).to(self.device)
            noisy_images = images + noise * 0.1
            
            # Forward pass
            self.optimizer.zero_grad()
            predicted_noise = self.model(noisy_images, timesteps, text_embeddings)
            
            # Calculate loss
            loss = self.criterion(predicted_noise, noise)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            total_loss += loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'avg_loss': f'{total_loss/(batch_idx+1):.4f}'
            })
            
            time.sleep(0.01)
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def validate(self, dataloader: DataLoader) -> float:
       
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating"):
                images = batch['pixel_values'].to(self.device)
                prompts = batch['prompt']
                
                text_embeddings = self.text_encoder(prompts)
                
                noise = torch.randn_like(images)
                timesteps = torch.randint(0, 1000, (images.shape[0],)).to(self.device)
                noisy_images = images + noise * 0.1
                
                predicted_noise = self.model(noisy_images, timesteps, text_embeddings)
                loss = self.criterion(predicted_noise, noise)
                
                total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        return avg_loss
    
    def save_checkpoint(self, epoch: int, loss: float, filepath: str):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        torch.save(checkpoint, filepath)
        logger.info(f"Checkpoint saved to {filepath}")
    
    def load_checkpoint(self, filepath: str):
        checkpoint = torch.load(filepath, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        logger.info(f"Checkpoint loaded from {filepath}")
        return checkpoint['epoch'], checkpoint['loss']

def train_diffusion_model(
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    save_dir: str = "checkpoints",
    dataset_size: int = 1000
):    
    logger.info(" Starting Stable Diffusion Training")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ImageDataset(size=int(dataset_size * 0.8))
    val_dataset = ImageDataset(size=int(dataset_size * 0.2))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model and trainer
    model = DiffusionModel(channels=3)  # Use 3 channels for RGB images
    trainer = DiffusionTrainer(model, device, learning_rate, batch_size)
    
    # Training loop
    training_stats = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'learning_rates': []
    }
    
    for epoch in range(epochs):
        logger.info(f"\n=== Epoch {epoch+1}/{epochs} ===")
        
        # Train
        train_loss = trainer.train_epoch(train_loader)
        trainer.train_losses.append(train_loss)
        
        # Validate
        val_loss = trainer.validate(val_loader)
        trainer.val_losses.append(val_loss)
        
        # Save stats
        training_stats['epochs'].append(epoch + 1)
        training_stats['train_losses'].append(train_loss)
        training_stats['val_losses'].append(val_loss)
        training_stats['learning_rates'].append(trainer.optimizer.param_groups[0]['lr'])
        
        logger.info(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
        
        # Save best model
        if val_loss < trainer.best_loss:
            trainer.best_loss = val_loss
            trainer.save_checkpoint(
                epoch, val_loss, 
                os.path.join(save_dir, 'best_diffusion_model.pth')
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            trainer.save_checkpoint(
                epoch, val_loss,
                os.path.join(save_dir, f'diffusion_epoch_{epoch+1}.pth')
            )
    
    # Save final training stats
    with open(os.path.join(save_dir, 'training_stats.json'), 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info("Diffusion Training completed!")
    return trainer, training_stats

if __name__ == "__main__":
    # Example usage
    trainer, stats = train_diffusion_model(
        epochs=50,
        batch_size=4,
        learning_rate=1e-4,
        dataset_size=500
    )