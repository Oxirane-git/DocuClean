import os
import re
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# Set matplotlib parameters for high-quality output
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 12

# Directories - using organized folder structure
noisy_base_dir = "noisy_dataset"
clean_base_dir = "clean_dataset"

# Suffix folders (TE, TR, VA)
suffixes = ["TE", "TR", "VA"]

class SuffixMatchedDenoiseDataset(Dataset):
    """
    Dataset that matches noisy images with clean images based on:
    1. Suffix folder (TE, TR, VA)
    2. Document type (extracted from filename)
    """
    def __init__(self, noisy_base_dir, clean_base_dir, suffixes, transform=None):
        self.transform = transform
        self.pairs = []
        
        # Process each suffix folder
        for suffix in suffixes:
            noisy_dir = os.path.join(noisy_base_dir, suffix)
            clean_dir = os.path.join(clean_base_dir, suffix)
            
            if not os.path.exists(noisy_dir) or not os.path.exists(clean_dir):
                print(f"Warning: {noisy_dir} or {clean_dir} does not exist. Skipping...")
                continue
            
            # Get all noisy images in this suffix folder
            noisy_files = sorted([f for f in os.listdir(noisy_dir) 
                                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Get all clean images in this suffix folder
            clean_files = sorted([f for f in os.listdir(clean_dir) 
                                if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            
            # Create a dictionary of clean images by document type
            clean_dict = {}
            for clean_file in clean_files:
                # Extract document type from clean filename
                # Example: "Fontfre_Clean_TE.png" -> "Fontfre"
                parts = clean_file.split('_')
                if len(parts) >= 1:
                    doc_type = parts[0]
                    clean_dict[doc_type] = clean_file
            
            # Match noisy images with clean images
            for noisy_file in noisy_files:
                # Extract document type from noisy filename
                # Example: "Fontfre_Noisec_TE.png" -> "Fontfre"
                parts = noisy_file.split('_')
                if len(parts) >= 1:
                    doc_type = parts[0]
                    
                    # Find matching clean image
                    if doc_type in clean_dict:
                        clean_file = clean_dict[doc_type]
                        noisy_path = os.path.join(noisy_dir, noisy_file)
                        clean_path = os.path.join(clean_dir, clean_file)
                        
                        self.pairs.append((noisy_path, clean_path, suffix))
                    else:
                        print(f"Warning: No matching clean image for {noisy_file}")
        
        print(f"\nCreated {len(self.pairs)} matched training pairs")
        print(f"Breakdown by suffix:")
        suffix_counts = {}
        for _, _, suffix in self.pairs:
            suffix_counts[suffix] = suffix_counts.get(suffix, 0) + 1
        for suffix, count in sorted(suffix_counts.items()):
            print(f"  {suffix}: {count} pairs")
    
    def __len__(self):
        return len(self.pairs)
    
    def __getitem__(self, idx):
        noisy_path, clean_path, suffix = self.pairs[idx]
        
        noisy_img = Image.open(noisy_path).convert("L")
        clean_img = Image.open(clean_path).convert("L")
        
        if self.transform:
            noisy_img = self.transform(noisy_img)
            clean_img = self.transform(clean_img)
        
        return noisy_img, clean_img

# Transform
transform = transforms.Compose([
    transforms.Resize((256, 256), Image.LANCZOS),
    transforms.ToTensor(),
])

# Create dataset and dataloader
if __name__ == '__main__':
    print("Loading dataset...")
    dataset = SuffixMatchedDenoiseDataset(noisy_base_dir, clean_base_dir, suffixes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)  # num_workers=0 for Windows compatibility

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return self.relu(out)

# Denoising Autoencoder
class DenoisingAutoencoder(nn.Module):
    def __init__(self):
        super(DenoisingAutoencoder, self).__init__()
        
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool2d(2, 2)  # 128x128
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)  # 64x64
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)  # 32x32
        
        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        
        # Decoder
        self.upconv3 = nn.ConvTranspose2d(128, 128, 2, stride=2)  # 64x64
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),  # 128 + 128 = 256 from skip connection
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(64, 64, 2, stride=2)  # 128x128
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),  # 64 + 64 = 128 from skip connection
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)  # 256x256
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),  # 32 + 32 = 64 from skip connection
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(16, 1, 1)
        
    def forward(self, x):
        # Encoder with skip connections
        enc1 = self.enc1(x)          # 32 channels, 256x256
        enc1_pool = self.pool1(enc1) # 32 channels, 128x128
        
        enc2 = self.enc2(enc1_pool)  # 64 channels, 128x128
        enc2_pool = self.pool2(enc2) # 64 channels, 64x64
        
        enc3 = self.enc3(enc2_pool)  # 128 channels, 64x64
        enc3_pool = self.pool3(enc3) # 128 channels, 32x32
        
        # Bottleneck
        bottleneck = self.bottleneck(enc3_pool)  # 128 channels, 32x32
        
        # Decoder with skip connections
        up3 = self.upconv3(bottleneck)           # 128 channels, 64x64
        concat3 = torch.cat([up3, enc3], dim=1)  # 128 + 128 = 256 channels
        dec3 = self.dec3(concat3)                # 64 channels, 64x64
        
        up2 = self.upconv2(dec3)                 # 64 channels, 128x128
        concat2 = torch.cat([up2, enc2], dim=1)  # 64 + 64 = 128 channels
        dec2 = self.dec2(concat2)                # 32 channels, 128x128
        
        up1 = self.upconv1(dec2)                 # 32 channels, 256x256
        concat1 = torch.cat([up1, enc1], dim=1)  # 32 + 32 = 64 channels
        dec1 = self.dec1(concat1)                # 16 channels, 256x256
        
        output = torch.sigmoid(self.final_conv(dec1))  # 1 channel, 256x256
        
        return output

# Combined loss function
class CombinedLoss(nn.Module):
    def __init__(self):
        super(CombinedLoss, self).__init__()
        self.mse = nn.MSELoss()
        self.l1 = nn.L1Loss()
    
    def forward(self, output, target):
        mse_loss = self.mse(output, target)
        l1_loss = self.l1(output, target)
        
        # Edge preservation loss
        def gradient_loss(pred, target):
            pred_grad_x = torch.abs(pred[:, :, :, :-1] - pred[:, :, :, 1:])
            pred_grad_y = torch.abs(pred[:, :, :-1, :] - pred[:, :, 1:, :])
            target_grad_x = torch.abs(target[:, :, :, :-1] - target[:, :, :, 1:])
            target_grad_y = torch.abs(target[:, :, :-1, :] - target[:, :, 1:, :])
            
            return self.mse(pred_grad_x, target_grad_x) + self.mse(pred_grad_y, target_grad_y)
        
        grad_loss = gradient_loss(output, target)
        
        return 0.6 * mse_loss + 0.3 * l1_loss + 0.1 * grad_loss

# Training code
if __name__ == '__main__':
    # Create models directory if it doesn't exist
    checkpoint_dir = "models"
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Recreate dataset and dataloader (from first if block)
    print("Loading dataset...")
    dataset = SuffixMatchedDenoiseDataset(noisy_base_dir, clean_base_dir, suffixes, transform=transform)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Initialize model, loss, and optimizer
    print("\nInitializing model...")
    model = DenoisingAutoencoder().to(device)
    criterion = CombinedLoss()
    optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    
    # Training loop
    num_epochs = 100
    train_losses = []
    best_loss = float('inf')

    print(f"\nStarting training for {num_epochs} epochs...")
    print("="*60)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch_idx, (noisy, clean) in enumerate(dataloader):
            noisy, clean = noisy.to(device), clean.to(device)
            
            optimizer.zero_grad()
            output = model(noisy)
            loss = criterion(output, clean)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f'Epoch {epoch+1}/{num_epochs}, Batch {batch_idx}/{len(dataloader)}, Loss: {loss.item():.4f}')
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        scheduler.step(avg_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Average Loss: {avg_loss:.4f}, LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), os.path.join(checkpoint_dir, "best_denoising_model.pth"))
            print(f"  ✓ New best model saved with loss: {best_loss:.4f}")
        
        # Save checkpoint every 20 epochs
        if (epoch + 1) % 20 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth'))
            print(f"  ✓ Checkpoint saved at epoch {epoch+1}")

    # Save final model
    torch.save(model.state_dict(), os.path.join(checkpoint_dir, "final_denoising_model.pth"))
    print("\n" + "="*60)
    print("Training completed!")
    print(f"Best loss: {best_loss:.4f}")
    print("="*60)

    # Plot training curve
    def plot_training_curve():
        """Create training loss plot"""
        plt.figure(figsize=(12, 8))
        plt.plot(train_losses, linewidth=2, color='#2E8B57', label='Training Loss')
        plt.title('Image Denoising Model - Training Loss Curve', 
                  fontsize=16, fontweight='bold', pad=20)
        plt.xlabel('Epoch', fontsize=14, fontweight='bold')
        plt.ylabel('Loss', fontsize=14, fontweight='bold')
        plt.grid(True, alpha=0.3, linestyle='--')
        plt.legend(fontsize=12)
        
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        
        plt.tight_layout()
        plt.savefig('training_loss_curve.png', dpi=300, bbox_inches='tight', facecolor='white')
        print("\nTraining curve saved to: training_loss_curve.png")
        plt.show()

    plot_training_curve()

    # Test denoising function
    def denoise_image(image_path, save_path=None):
        """Denoise a single image"""
        model.eval()
        
        # Load and preprocess image
        img = Image.open(image_path).convert("L")
        original_size = img.size
        
        # Transform image
        img_tensor = transform(img).unsqueeze(0).to(device)
        
        with torch.no_grad():
            output = model(img_tensor)
            output = output.squeeze().cpu()
        
        # Convert back to PIL Image
        output_np = output.numpy()
        output_np = np.clip(output_np, 0, 1)
        output_img = Image.fromarray((output_np * 255).astype(np.uint8), mode='L')
        output_img = output_img.resize(original_size, Image.LANCZOS)
        
        if save_path:
            output_img.save(save_path)
            print(f"Denoised image saved to: {save_path}")
        
        return output_img

    print("\nModel ready for inference!")
    print("Use denoise_image(image_path, save_path) to denoise images.")

