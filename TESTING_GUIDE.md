# Model Testing Guide

This guide explains how to test your trained denoising model using `test_model.py`.

## Quick Start

### Test a Single Image

```bash
# Basic usage - denoise a single image
python test_model.py --input path/to/noisy_image.png --output denoised_output.png

# With comparison visualization
python test_model.py --input path/to/noisy_image.png --output denoised_output.png --show

# With ground truth comparison
python test_model.py --input path/to/noisy_image.png --output denoised_output.png \
    --clean path/to/clean_image.png --show
```

### Test Multiple Images (Directory)

```bash
# Process all images in a directory
python test_model.py --input noisy_dataset/TE/ --output results/

# With clean images for comparison
python test_model.py --input noisy_dataset/TE/ --output results/ \
    --clean clean_dataset/TE/
```

### Specify a Different Model

```bash
# Use a specific checkpoint
python test_model.py --model models/checkpoint_epoch_100.pth \
    --input path/to/image.png --output result.png

# Use final model
python test_model.py --model models/final_denoising_model.pth \
    --input path/to/image.png --output result.png
```

## Examples

### Example 1: Test on a single noisy image
```bash
python test_model.py --input noisy_dataset/TE/Fontfre_Noisec_TE.png \
    --output test_result.png --show
```

### Example 2: Test on all images in a folder
```bash
python test_model.py --input noisy_dataset/TR/ --output test_results/TR/
```

### Example 3: Compare with ground truth
```bash
python test_model.py \
    --input noisy_dataset/VA/some_image_VA.png \
    --clean clean_dataset/VA/some_image_VA.png \
    --output comparison_result.png \
    --show
```

### Example 4: Use CPU instead of GPU
```bash
python test_model.py --device cpu --input image.png --output result.png
```

## Available Models

After training, you'll find these models in the `models/` folder:

- `best_denoising_model.pth` - Best model during training (lowest loss)
- `final_denoising_model.pth` - Final model after all epochs
- `checkpoint_epoch_X.pth` - Checkpoints saved every 20 epochs

**Note:** If you have old model files with names like `best_denoiser_suffix.pth`, you can still use them:
```bash
python test_model.py --model models/best_denoiser_suffix.pth --input image.png --output result.png
```

## Output

- **Single image mode**: Saves the denoised image to the specified output path
- **Directory mode**: Saves all denoised images with `denoised_` prefix
- **With --show flag**: Creates a comparison plot showing noisy, denoised, and (if provided) clean images

## Tips

1. **Best model**: Use `best_denoising_model.pth` for best results
2. **Batch processing**: Use directory mode for processing multiple images
3. **Visualization**: Use `--show` flag to see side-by-side comparisons
4. **Ground truth**: Provide `--clean` path to compare with original clean images

## Troubleshooting

**Model not found error:**
- Check that the model file exists in the `models/` folder
- Use the full path: `--model models/your_model.pth`

**CUDA out of memory:**
- Use `--device cpu` to run on CPU
- Process images one at a time instead of in batches

**Image not found:**
- Verify the input path is correct
- Check file permissions

