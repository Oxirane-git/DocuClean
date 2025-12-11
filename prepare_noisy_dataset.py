import os
import shutil
from pathlib import Path

# Define source and destination directories
source_dir = "Noisy"
destination_dir = "noisy_dataset"

# Create destination directory if it doesn't exist
os.makedirs(destination_dir, exist_ok=True)

# Get all PNG files from the Noisy directory
noisy_files = [f for f in os.listdir(source_dir) if f.lower().endswith('.png')]

print(f"Found {len(noisy_files)} images in {source_dir}")
print("Organizing images by suffix (last two letters)...\n")

# Dictionary to track suffixes and their counts
suffix_counts = {}

# Process each file
for filename in noisy_files:
    # Extract suffix (last two letters before .png)
    # Example: "Fontfre_Noisec_TE.png" -> "TE"
    name_without_ext = os.path.splitext(filename)[0]  # Remove .png
    parts = name_without_ext.split('_')
    
    if len(parts) >= 3:
        # The suffix is the last part (e.g., "TE", "TR", "VA")
        suffix = parts[-1]
        
        # Create subfolder for this suffix if it doesn't exist
        suffix_folder = os.path.join(destination_dir, suffix)
        os.makedirs(suffix_folder, exist_ok=True)
        
        # Copy file to the appropriate subfolder
        source_path = os.path.join(source_dir, filename)
        dest_path = os.path.join(suffix_folder, filename)
        
        shutil.copy2(source_path, dest_path)
        
        # Track suffixes
        if suffix not in suffix_counts:
            suffix_counts[suffix] = 0
        suffix_counts[suffix] += 1
    else:
        print(f"Warning: Could not parse filename: {filename}")

# Print summary
print(f"\n{'='*60}")
print(f"Organization Complete!")
print(f"{'='*60}")
print(f"\nCreated folder: {destination_dir}")
print(f"\nSuffix categories found ({len(suffix_counts)} total):")
print("-" * 60)
for suffix in sorted(suffix_counts.keys()):
    print(f"  {suffix}: {suffix_counts[suffix]} images")
print("-" * 60)
print(f"\nTotal images organized: {len(noisy_files)}")
print(f"\nFolder structure created at: {os.path.abspath(destination_dir)}")

