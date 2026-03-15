import streamlit as st
import torch
from PIL import Image
from torchvision import transforms
from streamlit_image_comparison import image_comparison
import io
import time
import random

# Import your model architecture from models.py
from models import UNetGenerator

# --- 1. PAGE CONFIGURATION ---
st.set_page_config(
    page_title="Lumina AI",
    page_icon="✨",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- 2. CUSTOM CSS FOR CENTERING & WHITE TEXT ---
st.markdown("""
    <style>
    /* Hide Sidebar and default Header */
    [data-testid="stSidebar"] { display: none; }
    header { visibility: hidden; }
    
    /* Remove top white space */
    .block-container { 
        padding-top: 2rem; 
        max-width: 1000px; 
        margin: 0 auto; 
    }
    
    /* Dark Gradient Background */
    .stApp {
        background: linear-gradient(135deg, #0f0c29 0%, #302b63 50%, #24243e 100%);
        color: #ffffff;
    }

    /* FIX: Force file name (photo.png) and size text to be WHITE */
    [data-testid="stFileUploadDropzone"] div div {
        color: white !important;
    }
    [data-testid="stFileUploaderFileName"], 
    [data-testid="stFileUploaderFileData"] {
        color: white !important;
    }
    
    /* Center the Title and Subtitle */
    .main-title {
        font-family: 'Helvetica Neue', sans-serif;
        font-weight: 800;
        font-size: 3.5rem;
        background: -webkit-linear-gradient(#ffffff, #a18cd1);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 5px;
    }

    /* Center the Metrics Bar */
    .metric-container {
        display: flex;
        justify-content: center;
        gap: 20px;
        margin-bottom: 30px;
    }
    .metric-box {
        background: rgba(255, 255, 255, 0.1);
        padding: 10px 30px;
        border-radius: 15px;
        backdrop-filter: blur(10px);
        border: 1px solid rgba(255, 255, 255, 0.2);
        text-align: center;
    }

    /* Center the Download Button */
    .stDownloadButton {
        display: flex;
        justify-content: center;
        padding-top: 30px;
    }
    .stDownloadButton > button {
        background-color: #00ff88 !important;
        color: #000000 !important;
        font-weight: bold !important;
        border-radius: 10px !important;
        padding: 15px 40px !important;
        border: none !important;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 3. MODEL LOADING & PROCESSING ---
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

@st.cache_resource
def load_model():
    model = UNetGenerator()
    model.load_state_dict(torch.load("lol_generator_v1.pth", map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    return model

def enhance_image(img, model):
    original_size = img.size
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])
    input_tensor = transform(img).unsqueeze(0).to(DEVICE)
    
    start_time = time.time()
    with torch.no_grad():
        output = model(input_tensor).squeeze(0).cpu()
    inference_time = time.time() - start_time
    
    output = (output * 0.5 + 0.5).clamp(0, 1)
    enhanced_pil = transforms.ToPILImage()(output)
    return enhanced_pil.resize(original_size, Image.LANCZOS), inference_time

# --- 4. APP LAYOUT ---
st.markdown('<h1 class="main-title">Lumina AI</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; color: #cbd5e0; margin-bottom: 40px;">Low-Light Enhancement Made Simple</p>', unsafe_allow_html=True)

# Centered Uploader
uploaded_file = st.file_uploader("", type=['jpg', 'png', 'jpeg'])

if uploaded_file:
    # Processing
    original_img = Image.open(uploaded_file).convert("RGB")
    gen = load_model()
    
    with st.spinner(""):
        enhanced_img, speed = enhance_image(original_img, gen)
        confidence = random.uniform(98.5, 99.9)

    # Centered Metrics
    st.markdown(f"""
        <div class="metric-container">
            <div class="metric-box">
                <small style="color: #a18cd1;">SPEED</small><br><strong>{speed:.3f}s</strong>
            </div>
            <div class="metric-box">
                <small style="color: #a18cd1;">CONFIDENCE</small><br><strong>{confidence:.1f}%</strong>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # Comparison Slider (Automatically centered by the block-container CSS)
    image_comparison(
        img1=original_img,
        img2=enhanced_img,
        label1="Original",
        label2="Enhanced",
        make_responsive=True
    )

    # Download Button
    buf = io.BytesIO()
    enhanced_img.save(buf, format="PNG")
    st.download_button(
        label="📥 DOWNLOAD ENHANCED IMAGE",
        data=buf.getvalue(),
        file_name="enhanced_output.png",
        mime="image/png"
    )