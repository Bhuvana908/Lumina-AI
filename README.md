# LOL-GAN: Low-Light Image Enhancement using Pix2Pix GAN

A deep learning project that enhances low-light images using a U-Net Generator 
and PatchGAN Discriminator trained on the LOL (Low-Light) dataset.

---

## 📁 Project Structure
```
├── __pycache__/
│   └── models.cpython-312.pyc   # Compiled Python cache
├── Notebooks/
│   └── GAN_Based_Low_Light_Image_Enhancement.ipynb   # Google Colab training notebook
├── Video/
│   └── result_video             #output video
├── app                          # Application/inference code
├── models                       # Model architecture files
├── .gitattributes               # Git LFS tracking config
└── README.md
```

---

## 🧠 Model Architecture

- **Generator** — U-Net with skip connections (encoder-decoder)
- **Discriminator** — PatchGAN (70×70 patches)
- **Loss** — Adversarial Loss + L1 Loss (λ=100)

---

## 📦 Dataset

[LOL Dataset](https://www.kaggle.com/datasets/soumikrakshit/lol-dataset) from Kaggle
- **Train:** 485 low/high image pairs (`our485/`)
- **Val:** 15 low/high image pairs (`eval15/`)

---

## ⚙️ Training Config

| Parameter | Value |
|-----------|-------|
| Image Size | 256×256 |
| Batch Size | 4 |
| Epochs | 100 |
| Learning Rate | 2e-4 |
| Optimizer | Adam (β1=0.5, β2=0.999) |
| Lambda L1 | 100 |

---

## 📥 Pretrained Model

Download `lol_generator_v1.pth` from Google Drive:  
👉 [Download Model](https://drive.google.com/file/d/1WOT_VC5Pe59Z_wfshvl8BNZ4Kdizz6GU/view?usp=sharing)

---

## 🔧 Usage

```python
import torch
from PIL import Image
from torchvision import transforms

# Load model
model = UNetGenerator()
model.load_state_dict(torch.load("lol_generator_v1.pth", map_location="cpu"))
model.eval()

# Inference
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

img = Image.open("low_light_image.png").convert("RGB")
tensor = transform(img).unsqueeze(0)

with torch.no_grad():
    enhanced = model(tensor).squeeze(0)
    enhanced = (enhanced * 0.5 + 0.5).clamp(0, 1)
```

---

## 📊 Evaluation Metrics

- **PSNR** — Peak Signal-to-Noise Ratio
- **SSIM** — Structural Similarity Index

---

## 🛠️ Requirements

```bash
pip install torch torchvision opencv-python scikit-image matplotlib
```
