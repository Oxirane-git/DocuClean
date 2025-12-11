# 🧼 Image Denoising Autoencoder

PyTorch-based denoising for grayscale document scans (IDs, licenses, certificates). Includes a Streamlit UI for quick demos and training/testing scripts for experimentation.

## Project Structure
```
.
├─ streamlit_app.py           # Streamlit demo UI (uses trained model)
├─ train_denoising_model.py   # Training script
├─ test_model.py              # Inference/testing helper
├─ Denoise2.py                # Additional experimental script
├─ prepare_clean_dataset.py   # Dataset prep helper (clean images)
├─ prepare_noisy_dataset.py   # Dataset prep helper (noisy images)
├─ models/
│   ├─ final_denoiser_suffix.pth   # Trained checkpoint (used by Streamlit)
│   └─ other checkpoints...
├─ data/                      # (ignored) place datasets here; see data/README.md
├─ assets/, result.png        # Sample visuals/results
├─ requirements.txt
├─ README.md
└─ TESTING_GUIDE.md
```

## Setup
```bash
cd /mnt/NewDisk/sahil_project/Image-Denoising-Autoencoder-main
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the Streamlit demo
```bash
source .venv/bin/activate
streamlit run streamlit_app.py --server.port 3001
```
- Requires `models/final_denoiser_suffix.pth` (adjust path inside `streamlit_app.py` if you prefer a different checkpoint).
- Upload a noisy image; the app denoises using the trained autoencoder.

## Training
```bash
source .venv/bin/activate
python train_denoising_model.py \
  --epochs 50 \
  --batch_size 8 \
  --learning_rate 1e-3 \
  --noisy_dir data/noisy \
  --clean_dir data/clean \
  --save_path models/best_denoiser.pth
```
Tweak paths/params as needed. Place your datasets under `data/` (ignored by git).

## Testing / CLI inference
```bash
source .venv/bin/activate
python test_model.py \
  --model models/final_denoiser_suffix.pth \
  --input path/to/noisy.png \
  --output outputs/denoised.png
```
For a folder, point `--input` to a directory and optionally `--clean` for ground truth comparisons.

## Data layout
See `data/README.md`. Keep large datasets out of git. Suggested:
```
data/
  noisy/
  clean/
```

## Notes
- `.gitignore` excludes datasets, zips, virtualenvs, and temp outputs to keep the repo lean.
- If you need to free a port or stop Streamlit: `pkill -f "streamlit run streamlit_app.py"`.
- For GPU use, ensure CUDA is available; scripts fall back to CPU automatically.