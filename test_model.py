import os
import argparse
import torch
import torch.nn as nn
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms

# Import model architecture from training script
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
        self.pool1 = nn.MaxPool2d(2, 2)
        
        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool2d(2, 2)
        
        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool2d(2, 2)
        
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
        self.upconv3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )
        
        self.upconv2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True)
        )
        
        self.upconv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        
        self.final_conv = nn.Conv2d(16, 1, 1)
        
    def forward(self, x):
        enc1 = self.enc1(x)
        enc1_pool = self.pool1(enc1)
        
        enc2 = self.enc2(enc1_pool)
        enc2_pool = self.pool2(enc2)
        
        enc3 = self.enc3(enc2_pool)
        enc3_pool = self.pool3(enc3)
        
        bottleneck = self.bottleneck(enc3_pool)
        
        up3 = self.upconv3(bottleneck)
        concat3 = torch.cat([up3, enc3], dim=1)
        dec3 = self.dec3(concat3)
        
        up2 = self.upconv2(dec3)
        concat2 = torch.cat([up2, enc2], dim=1)
        dec2 = self.dec2(concat2)
        
        up1 = self.upconv1(dec2)
        concat1 = torch.cat([up1, enc1], dim=1)
        dec1 = self.dec1(concat1)
        
        output = torch.sigmoid(self.final_conv(dec1))
        
        return output


def load_model(model_path, device):
    """Load the trained model"""
    model = DenoisingAutoencoder()
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✓ Model loaded from: {model_path}")
    return model


def denoise_image(model, image_path, device, save_path=None, show_comparison=False, clean_path=None):
    """Denoise a single image"""
    # Transform
    transform = transforms.Compose([
        transforms.Resize((256, 256), Image.LANCZOS),
        transforms.ToTensor(),
    ])
    
    # Load and preprocess image
    img = Image.open(image_path).convert("L")
    original_size = img.size
    
    # Transform image
    img_tensor = transform(img).unsqueeze(0).to(device)
    
    # Denoise
    with torch.no_grad():
        output = model(img_tensor)
        output = output.squeeze().cpu()
    
    # Convert back to PIL Image
    output_np = output.numpy()
    output_np = np.clip(output_np, 0, 1)
    output_img = Image.fromarray((output_np * 255).astype(np.uint8), mode='L')
    output_img = output_img.resize(original_size, Image.LANCZOS)
    
    # Save if path provided
    if save_path:
        output_img.save(save_path)
        print(f"✓ Denoised image saved to: {save_path}")
    
    # Show comparison if requested
    if show_comparison:
        fig, axes = plt.subplots(1, 3 if clean_path else 2, figsize=(15, 5))
        
        axes[0].imshow(img, cmap='gray')
        axes[0].set_title('Noisy Image', fontsize=14, fontweight='bold')
        axes[0].axis('off')
        
        axes[1].imshow(output_img, cmap='gray')
        axes[1].set_title('Denoised Image', fontsize=14, fontweight='bold')
        axes[1].axis('off')
        
        if clean_path:
            clean_img = Image.open(clean_path).convert("L")
            axes[2].imshow(clean_img, cmap='gray')
            axes[2].set_title('Clean (Ground Truth)', fontsize=14, fontweight='bold')
            axes[2].axis('off')
        
        plt.tight_layout()
        comparison_path = save_path.replace('.png', '_comparison.png') if save_path else 'comparison.png'
        plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
        print(f"✓ Comparison saved to: {comparison_path}")
        plt.show()
    
    return output_img


def test_on_directory(model, noisy_dir, device, output_dir=None, clean_dir=None):
    """Test model on all images in a directory"""
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_extensions = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')
    image_files = [f for f in os.listdir(noisy_dir) if f.lower().endswith(image_extensions)]
    
    print(f"\nFound {len(image_files)} images in {noisy_dir}")
    print("Processing images...\n")
    
    for idx, filename in enumerate(image_files, 1):
        noisy_path = os.path.join(noisy_dir, filename)
        save_path = os.path.join(output_dir, f"denoised_{filename}") if output_dir else None
        clean_path = os.path.join(clean_dir, filename) if clean_dir and os.path.exists(os.path.join(clean_dir, filename)) else None
        
        print(f"[{idx}/{len(image_files)}] Processing: {filename}")
        denoise_image(model, noisy_path, device, save_path=save_path, clean_path=clean_path)
    
    print(f"\n✓ All images processed! Results saved to: {output_dir}")


def main():
    parser = argparse.ArgumentParser(description='Test the trained denoising model')
    parser.add_argument('--model', type=str, default='models/best_denoising_model.pth',
                       help='Path to the model checkpoint (default: models/best_denoising_model.pth)')
    parser.add_argument('--input', type=str, required=True,
                       help='Path to input image or directory')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output image or directory (optional)')
    parser.add_argument('--clean', type=str, default=None,
                       help='Path to clean (ground truth) image or directory for comparison (optional)')
    parser.add_argument('--show', action='store_true',
                       help='Show comparison plot')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cuda', 'cpu'],
                       help='Device to use (default: auto)')
    
    args = parser.parse_args()
    
    # Set device
    if args.device == 'auto':
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    print(f"Using device: {device}")
    print("="*60)
    
    # Load model
    if not os.path.exists(args.model):
        print(f"Error: Model file not found at {args.model}")
        print("Available models in models/ folder:")
        if os.path.exists('models'):
            for f in os.listdir('models'):
                if f.endswith('.pth'):
                    print(f"  - models/{f}")
        return
    
    model = load_model(args.model, device)
    print("="*60)
    
    # Check if input is a file or directory
    if os.path.isfile(args.input):
        # Single image
        clean_path = args.clean if args.clean and os.path.isfile(args.clean) else None
        denoise_image(model, args.input, device, save_path=args.output, 
                     show_comparison=args.show, clean_path=clean_path)
    elif os.path.isdir(args.input):
        # Directory of images
        test_on_directory(model, args.input, device, output_dir=args.output, clean_dir=args.clean)
    else:
        print(f"Error: Input path '{args.input}' does not exist")
        return
    
    print("\n" + "="*60)
    print("Testing complete!")


if __name__ == '__main__':
    main()

