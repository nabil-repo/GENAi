import os
import time
import random
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
import numpy as np
from typing import Dict, List, Optional, Tuple
import logging
from tqdm import tqdm

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ImageVAEDataset(Dataset):
    """ dataset for VAE training with high-quality images"""
    
    def __init__(self, size: int = 5000, image_size: int = 512):
        self.size = size
        self.image_size = image_size
        
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])  # [-1, 1] range
        ])
        
        # Image categories for diverse training
        self.categories = [
            "portraits", "landscapes", "abstract", "architecture", 
            "still_life", "animals", "digital_art", "paintings"
        ]
    
    def generate__image(self, category: str) -> torch.Tensor:
        """Generate  image based on category"""
        # Create structured noise based on category
        if category == "portraits":
            # More structured for faces
            image = torch.randn(3, self.image_size, self.image_size)
            # Add some structure
            center_y, center_x = self.image_size // 2, self.image_size // 2
            y, x = torch.meshgrid(torch.arange(self.image_size), torch.arange(self.image_size), indexing='ij')
            mask = ((y - center_y)**2 + (x - center_x)**2) < (self.image_size // 4)**2
            image[:, mask] *= 0.5
            
        elif category == "landscapes":
            # Horizontal structure for landscapes
            image = torch.randn(3, self.image_size, self.image_size)
            # Add horizontal gradients
            for i in range(3):
                gradient = torch.linspace(-1, 1, self.image_size).unsqueeze(0).expand(self.image_size, -1)
                image[i] += gradient * 0.3
                
        else:
            # General random image
            image = torch.randn(3, self.image_size, self.image_size)
        
        # Normalize to [-1, 1]
        image = torch.tanh(image)
        
        return image
    
    def __len__(self):
        return self.size
    
    def __getitem__(self, idx):
        category = random.choice(self.categories)
        image = self.generate__image(category)
        
        return {
            'image': image,
            'category': category,
            'index': idx
        }

class VAEModel(nn.Module):
    """ VAE model with encoder-decoder architecture"""
    
    def __init__(self, 
                 input_channels: int = 3,
                 latent_dim: int = 512,
                 hidden_dims: List[int] = None):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [32, 64, 128, 256, 512]
        
        self.latent_dim = latent_dim
        self.hidden_dims = hidden_dims
        
        # Encoder
        encoder_layers = []
        in_channels = input_channels
        
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Conv2d(in_channels, h_dim, kernel_size=3, stride=2, padding=1),
                nn.BatchNorm2d(h_dim),
                nn.LeakyReLU(0.2),
            ])
            in_channels = h_dim
        
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Calculate the size after convolutions
        self.flatten_size = hidden_dims[-1] * (512 // (2 ** len(hidden_dims))) ** 2
        
        # Latent space
        self.fc_mu = nn.Linear(self.flatten_size, latent_dim)
        self.fc_logvar = nn.Linear(self.flatten_size, latent_dim)
        
        # Decoder input
        self.decoder_input = nn.Linear(latent_dim, self.flatten_size)
        
        # Decoder
        decoder_layers = []
        reversed_dims = hidden_dims[::-1]
        
        for i in range(len(reversed_dims) - 1):
            decoder_layers.extend([
                nn.ConvTranspose2d(reversed_dims[i], reversed_dims[i + 1], 
                                 kernel_size=3, stride=2, padding=1, output_padding=1),
                nn.BatchNorm2d(reversed_dims[i + 1]),
                nn.LeakyReLU(0.2),
            ])
        
        # Final layer
        decoder_layers.extend([
            nn.ConvTranspose2d(reversed_dims[-1], input_channels,
                             kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.Tanh()
        ])
        
        self.decoder = nn.Sequential(*decoder_layers)
        
    def encode(self, x):
        """Encode input to latent space"""
        h = self.encoder(x)
        h = h.view(h.size(0), -1)
        
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        """Reparameterization trick"""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z):
        """Decode latent representation to image"""
        h = self.decoder_input(z)
        h = h.view(h.size(0), self.hidden_dims[-1], 
                   512 // (2 ** len(self.hidden_dims)), 
                   512 // (2 ** len(self.hidden_dims)))
        
        result = self.decoder(h)
        return result
    
    def forward(self, x):
        """Full forward pass"""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z)
        
        return recon_x, mu, logvar

class VAETrainer:
    """Trainer class for VAE"""
    
    def __init__(self, 
                 model: nn.Module,
                 device: str = "cuda",
                 learning_rate: float = 1e-4,
                 batch_size: int = 8,
                 beta: float = 1.0):
        
        self.model = model.to(device)
        self.device = device
        self.batch_size = batch_size
        self.beta = beta  # KL divergence weight
        
        # Optimizer
        self.optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        
        # Training stats
        self.train_losses = []
        self.val_losses = []
        self.recon_losses = []
        self.kl_losses = []
        self.best_loss = float('inf')
        
    def vae_loss(self, recon_x, x, mu, logvar):
        """Calculate VAE loss (reconstruction + KL divergence)"""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train_epoch(self, dataloader: DataLoader) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        progress_bar = tqdm(dataloader, desc="Training VAE")
        
        for batch_idx, batch in enumerate(progress_bar):
            images = batch['image'].to(self.device)
            
            # Forward pass
            self.optimizer.zero_grad()
            recon_images, mu, logvar = self.model(images)
            
            # Calculate loss
            loss, recon_loss, kl_loss = self.vae_loss(recon_images, images, mu, logvar)
            
            # Normalize by batch size
            loss = loss / images.size(0)
            recon_loss = recon_loss / images.size(0)
            kl_loss = kl_loss / images.size(0)
            
            # Backward pass
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            total_recon_loss += recon_loss.item()
            total_kl_loss += kl_loss.item()
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'recon': f'{recon_loss.item():.4f}',
                'kl': f'{kl_loss.item():.4f}'
            })
            
            # Simulate training time
            time.sleep(0.01)
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def validate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        total_loss = 0.0
        total_recon_loss = 0.0
        total_kl_loss = 0.0
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Validating VAE"):
                images = batch['image'].to(self.device)
                
                recon_images, mu, logvar = self.model(images)
                loss, recon_loss, kl_loss = self.vae_loss(recon_images, images, mu, logvar)
                
                # Normalize by batch size
                loss = loss / images.size(0)
                recon_loss = recon_loss / images.size(0)
                kl_loss = kl_loss / images.size(0)
                
                total_loss += loss.item()
                total_recon_loss += recon_loss.item()
                total_kl_loss += kl_loss.item()
        
        avg_loss = total_loss / len(dataloader)
        avg_recon_loss = total_recon_loss / len(dataloader)
        avg_kl_loss = total_kl_loss / len(dataloader)
        
        return {
            'total_loss': avg_loss,
            'recon_loss': avg_recon_loss,
            'kl_loss': avg_kl_loss
        }
    
    def generate_samples(self, num_samples: int = 8) -> torch.Tensor:
        """Generate new samples from the VAE"""
        self.model.eval()
        
        with torch.no_grad():
            # Sample from standard normal
            z = torch.randn(num_samples, self.model.latent_dim).to(self.device)
            
            # Decode
            samples = self.model.decode(z)
            
        return samples
    
    def reconstruct_images(self, images: torch.Tensor) -> torch.Tensor:
        """Reconstruct input images"""
        self.model.eval()
        
        with torch.no_grad():
            recon_images, _, _ = self.model(images)
        
        return recon_images
    
    def save_checkpoint(self, epoch: int, loss: float, filepath: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'loss': loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'recon_losses': self.recon_losses,
            'kl_losses': self.kl_losses,
            'beta': self.beta
        }
        torch.save(checkpoint, filepath)
        logger.info(f"VAE checkpoint saved to {filepath}")

def train_vae_model(
    epochs: int = 100,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    beta: float = 1.0,
    save_dir: str = "vae_checkpoints",
    dataset_size: int = 2000
):
    """Main training function for VAE"""
    
    logger.info("🎭 Starting VAE Training")
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Using device: {device}")
    
    # Create datasets
    train_dataset = ImageVAEDataset(size=int(dataset_size * 0.8))
    val_dataset = ImageVAEDataset(size=int(dataset_size * 0.2))
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Create model and trainer
    model = VAEModel(latent_dim=512)
    trainer = VAETrainer(model, device, learning_rate, batch_size, beta)
    
    # Training loop
    training_stats = {
        'epochs': [],
        'train_losses': [],
        'val_losses': [],
        'recon_losses': [],
        'kl_losses': [],
        'learning_rates': [],
        'beta_values': []
    }
    
    for epoch in range(epochs):
        logger.info(f"\n=== VAE Epoch {epoch+1}/{epochs} ===")
        
        # Train
        train_metrics = trainer.train_epoch(train_loader)
        trainer.train_losses.append(train_metrics['total_loss'])
        trainer.recon_losses.append(train_metrics['recon_loss'])
        trainer.kl_losses.append(train_metrics['kl_loss'])
        
        # Validate
        val_metrics = trainer.validate(val_loader)
        trainer.val_losses.append(val_metrics['total_loss'])
        
        # Save stats
        training_stats['epochs'].append(epoch + 1)
        training_stats['train_losses'].append(train_metrics['total_loss'])
        training_stats['val_losses'].append(val_metrics['total_loss'])
        training_stats['recon_losses'].append(train_metrics['recon_loss'])
        training_stats['kl_losses'].append(train_metrics['kl_loss'])
        training_stats['learning_rates'].append(trainer.optimizer.param_groups[0]['lr'])
        training_stats['beta_values'].append(trainer.beta)
        
        logger.info(f"Train Loss: {train_metrics['total_loss']:.4f} | "
                   f"Recon Loss: {train_metrics['recon_loss']:.4f} | "
                   f"KL Loss: {train_metrics['kl_loss']:.4f} | "
                   f"Val Loss: {val_metrics['total_loss']:.4f}")
        
        # Beta annealing (optional)
        if epoch < epochs // 2:
            trainer.beta = min(1.0, trainer.beta * 1.01)
        
        # Test generation every 10 epochs
        if epoch % 10 == 0:
            logger.info("Generating sample images...")
            samples = trainer.generate_samples(num_samples=4)
            logger.info(f"Generated samples shape: {samples.shape}")
        
        # Save best model
        if val_metrics['total_loss'] < trainer.best_loss:
            trainer.best_loss = val_metrics['total_loss']
            trainer.save_checkpoint(
                epoch, val_metrics['total_loss'], 
                os.path.join(save_dir, 'best_vae_model.pth')
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % 20 == 0:
            trainer.save_checkpoint(
                epoch, val_metrics['total_loss'],
                os.path.join(save_dir, f'vae_epoch_{epoch+1}.pth')
            )
    
    # Save final training stats
    with open(os.path.join(save_dir, 'vae_training_stats.json'), 'w') as f:
        json.dump(training_stats, f, indent=2)
    
    logger.info("🎉 VAE Training completed!")
    return trainer, training_stats

if __name__ == "__main__":
    # Example usage
    trainer, stats = train_vae_model(
        epochs=50,
        batch_size=4,
        learning_rate=1e-4,
        beta=0.5,
        dataset_size=1000
    )