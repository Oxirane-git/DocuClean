import io
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


st.set_page_config(page_title="Image Denoising Demo", page_icon="🧹", layout="wide")

# Minimal, creative bento styling
st.markdown(
    """
    <style>
      :root {
        --bg: #0b1120;
        --card: rgba(255,255,255,0.04);
        --card-strong: rgba(255,255,255,0.07);
        --border: rgba(255,255,255,0.10);
        --text: #e5e7eb;
        --muted: #cbd5e1;
        --accent: #7dd3fc;
        --accent-2: #a78bfa;
      }
      .block-container {
        padding-top: 2.6rem;
        padding-bottom: 2.6rem;
        max-width: 1180px;
      }
      body {
        background: radial-gradient(circle at 12% 18%, #111a2e 0%, #0b1220 40%, #0a1020 70%);
        color: var(--text);
      }
      h1, h2, h3, h4 {
        color: var(--text);
        letter-spacing: -0.02em;
      }
      .stMarkdown p {
        color: var(--muted);
      }
      /* Buttons */
      .stButton>button {
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: #0b1220;
        border: none;
        border-radius: 12px;
        padding: 0.6rem 1.2rem;
        font-weight: 700;
        box-shadow: 0 14px 30px rgba(124, 181, 255, 0.25);
      }
      .stButton>button:hover {
        transform: translateY(-1px);
      }
      /* Cards */
      .bento {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 18px;
        padding: 1.15rem 1.3rem;
        box-shadow: 0 20px 42px rgba(0,0,0,0.38);
      }
      .bento-strong {
        background: var(--card-strong);
        border: 1px solid rgba(124, 181, 255, 0.25);
      }
      .pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        padding: 6px 12px;
        border-radius: 999px;
        border: 1px solid var(--border);
        background: rgba(255,255,255,0.05);
        color: var(--muted);
        font-size: 13px;
        margin-right: 8px;
      }
      .pill-accent {
        border-color: rgba(124, 181, 255, 0.35);
        color: #cce7ff;
      }
      /* File uploader tweaks */
      .stFileUploader label {
        color: var(--muted) !important;
      }
      .stFileUploader div[data-baseweb="file-uploader"] {
        border: 1px dashed var(--border);
        border-radius: 14px;
        background: rgba(255,255,255,0.03);
      }
      /* Captions and text */
      .stCaption, .stMarkdown p {
        color: var(--muted);
      }
      /* Download button */
      .stDownloadButton>button {
        background: #0b1220;
        color: var(--text);
        border: 1px solid var(--border);
        border-radius: 12px;
      }
      /* Navigation + hero */
      .nav {
        display: flex;
        align-items: center;
        justify-content: space-between;
        padding: 0.4rem 0;
        margin-bottom: 1.2rem;
      }
      .brand {
        display: inline-flex;
        align-items: center;
        gap: 10px;
        font-weight: 800;
        color: var(--text);
      }
      .brand-badge {
        width: 34px;
        height: 34px;
        border-radius: 12px;
        display: grid;
        place-items: center;
        background: linear-gradient(135deg, var(--accent), var(--accent-2));
        color: #0b1220;
        font-weight: 900;
      }
      .nav-links {
        display: inline-flex;
        gap: 18px;
        color: var(--muted);
        font-weight: 600;
      }
      .nav-links a {
        color: var(--muted);
        text-decoration: none;
      }
      .hero {
        background: linear-gradient(145deg, rgba(255,255,255,0.06), rgba(255,255,255,0.02));
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.6rem;
        box-shadow: 0 24px 48px rgba(0,0,0,0.35);
        margin-bottom: 1.2rem;
      }
      .hero h1 {
        font-size: 2.4rem;
        margin-bottom: 0.6rem;
      }
      .eyebrow {
        display: inline-flex;
        gap: 8px;
        align-items: center;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(124, 181, 255, 0.12);
        color: #cce7ff;
        font-weight: 700;
        letter-spacing: 0.03em;
      }
      .hero-sub {
        color: var(--muted);
        max-width: 720px;
      }
      .cta-row {
        display: inline-flex;
        gap: 12px;
        margin-top: 0.9rem;
      }
      .primary-ghost {
        background: transparent !important;
        color: var(--text) !important;
        border: 1px solid var(--border) !important;
      }
      .feature-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(240px, 1fr));
        gap: 12px;
        margin-top: 1.2rem;
      }
      .feature-card {
        background: var(--card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1rem;
        box-shadow: 0 16px 30px rgba(0,0,0,0.25);
      }
      .feature-card h4 {
        margin-top: 0;
        margin-bottom: 0.4rem;
      }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_PATH = Path("models/best_denoising_model.pth")


class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
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
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )
        self.pool1 = nn.MaxPool2d(2, 2)

        self.enc2 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.pool2 = nn.MaxPool2d(2, 2)

        self.enc3 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.bottleneck = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            ResidualBlock(256),
            ResidualBlock(256),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.upconv3 = nn.ConvTranspose2d(128, 128, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(256, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.upconv2 = nn.ConvTranspose2d(64, 64, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(128, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.upconv1 = nn.ConvTranspose2d(32, 32, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(64, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
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


@st.cache_resource(show_spinner=False)
def load_model(model_path: Path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DenoisingAutoencoder()
    state = torch.load(model_path, map_location=device)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model, device


def run_inference(model: nn.Module, device: torch.device, image: Image.Image) -> Image.Image:
    transform = transforms.Compose(
        [
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )
    img_gray = image.convert("L")
    original_size = img_gray.size
    tensor = transform(img_gray).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(tensor)
    output = output.squeeze().cpu().clamp(0, 1)
    output_img = Image.fromarray((output.numpy() * 255).astype(np.uint8), mode="L")
    output_img = output_img.resize(original_size, Image.BICUBIC)
    return output_img


st.markdown(
    """
    <div class="nav">
      <div class="brand">
        <div class="brand-badge">ID</div>
        <span>Insight Denoiser</span>
      </div>
      <div class="nav-links">
        <a href="#hero">Overview</a>
        <a href="#upload">Upload</a>
        <a href="#model">Model</a>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero" id="hero">
      <div class="eyebrow">U-Net inspired · PyTorch</div>
      <h1>Restore sharp clarity to scans and documents.</h1>
      <p class="hero-sub">
        Tailored autoencoder pipeline for IDs, licenses, and certificates.
        Remove noise, improve OCR accuracy, and deliver crisp, legible outputs.
      </p>
      <div class="cta-row">
        <a href="#upload"><button class="stButton">Try the demo</button></a>
        <a href="#model"><button class="stButton primary-ghost">View model info</button></a>
      </div>
      <div class="feature-grid">
        <div class="feature-card">
          <h4>Document-first tuning</h4>
          <p>Optimized for text-heavy scans with fine edges and subtle gradients.</p>
        </div>
        <div class="feature-card">
          <h4>GPU aware</h4>
          <p>Automatically leverages CUDA when available; falls back to CPU gracefully.</p>
        </div>
        <div class="feature-card">
          <h4>Consistent sizing</h4>
          <p>256×256 inference with size restoration to keep your layout intact.</p>
        </div>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

st.title("Image Denoising")
st.markdown(
    """
    Upload a noisy image to denoise it using the trained autoencoder.
    Minimal, bento-inspired styling with a restrained palette.
    """
)

st.markdown(
    """
    <div class="bento bento-strong" id="model">
      <div class="pill pill-accent">Autoencoder · TE suffix</div>
      <div class="pill">256×256 inference</div>
      <div class="pill">Grayscale pipeline</div>
      <p style="margin-top: 0.6rem;">Designed to keep focus on your images with a calm, minimal UI.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

uploaded = st.file_uploader("Choose a noisy image", type=["png", "jpg", "jpeg", "bmp", "gif"])

model_loaded = MODEL_PATH.exists()
if not model_loaded:
    st.error(f"Model not found at {MODEL_PATH}. Please add the trained checkpoint to proceed.")


if uploaded and model_loaded:
    bytes_data = uploaded.read()
    image = Image.open(io.BytesIO(bytes_data)).convert("RGB")

    model, device = load_model(MODEL_PATH)

    with st.spinner("Denoising..."):
        denoised = run_inference(model, device, image)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Noisy input")
        with st.container():
            st.markdown('<div class="bento bento-accent">', unsafe_allow_html=True)
            st.image(image, use_column_width=True)
            st.caption(f"{uploaded.name} · {image.size[0]}x{image.size[1]}")
            st.markdown('</div>', unsafe_allow_html=True)

    with col2:
        st.subheader("Denoised output")
        with st.container():
            st.markdown('<div class="bento bento-accent">', unsafe_allow_html=True)
            st.image(denoised, use_column_width=True, clamp=True)
            st.caption("Generated by the trained autoencoder.")
            st.markdown('</div>', unsafe_allow_html=True)

    # Prepare downloadable PNG
    buf = io.BytesIO()
    denoised.save(buf, format="PNG")
    buf.seek(0)
    st.download_button(
        label="Download denoised PNG",
        data=buf,
        file_name="denoised.png",
        mime="image/png",
    )
elif not uploaded:
    st.info("Upload a noisy image to begin.")

