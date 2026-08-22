# 🌿 AgroAI: AI-Powered Crop Disease Diagnosis & Agronomist Platform

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-009688.svg)](https://fastapi.tiangolo.com/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2%2B-EE4C2C.svg)](https://pytorch.org/)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ED.svg)](https://www.docker.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

> An enterprise-grade, end-to-end full-stack AI platform for plant pathology diagnosis, rotation-invariant leaf morphology classification, Grad-CAM visual explainability, infection severity scoring, microclimate disease forecasting, tank dosage calculations, and conversational AI agronomy advisory.

---

## 🌟 Key Features

- **🔬 67 Crop Disease Classes across 22 Global Crops**:
  - **Cereals & Staples**: Wheat, Corn / Maize, Rice, Potato, Cassava / Yuca.
  - **Cash Crops**: Sugarcane, Cotton, Coffee, Tea, Soybean.
  - **Fruits & Vegetables**: Tomato, Apple, Grape, Orange / Citrus, Peach, Pepper, Squash / Cucurbit, Strawberry, Cherry, Blueberry, Raspberry.
- **🤖 AgroBot AI (AI Agronomist Chatbot)**:
  - Context-aware chatbot synchronizing with live scan diagnostics.
  - Recommends commercial chemical trade names (*Syngenta Tilt*, *Bayer Confidor*, *Aries Plantomycin*, *BASF Priaxor*, *Indofil Dithane M-45*), organic bio-controls (*Trichoderma*, *Bacillus*), and dilution ratios.
  - Multilingual support: English (EN), Hindi (HI), Punjabi (PA), Spanish (ES), French (FR).
- **🔥 Explainable AI (Grad-CAM & Symptom Segmentation)**:
  - Real-time visual saliency heatmaps with dynamic alpha opacity blending (JET, Inferno, Viridis, Turbo).
  - Highlights precise pathogen colonies, lesions, and necrosis boundaries.
- **📐 Rotation-Invariant Morphology Engine**:
  - Minimum area bounding box geometry (`cv2.minAreaRect`) and principal axis PCA alignment.
  - Accurate under any leaf photo angle ($0^\circ, 45^\circ, 90^\circ$).
- **🌦️ Microclimate Outbreak Risk Forecasting**:
  - Open-Meteo live meteorological weather integration tracking temperature, relative humidity, and precipitation to forecast fungal/bacterial sporulation risks.
- **🧪 Field Tank Dosage Calculator**:
  - Computes exact formulation quantities and water volumes tailored to acreages and sprayer tank sizes.
- **📄 Agronomy Diagnosis Certificate (PDF Export)**:
  - Generates downloadable, printable agronomy reports with embedded leaf images, Grad-CAM maps, risk indices, and treatment schedules.
- **⚡ Docker & Cloud Ready**:
  - One-click deployment to Render, Railway, Hugging Face Spaces, or Docker VPS.

---

## 📸 Supported Crops & Disease Taxonomy (67 Classes)

| Category | Crops Included | Sample Diagnoses |
|---|---|---|
| **Cereals & Grains** | Wheat, Corn / Maize, Rice | Yellow Stripe Rust, Brown Rust, Northern Leaf Blight, Leaf Blast, Brown Spot |
| **Cash & Fiber** | Sugarcane, Cotton, Coffee, Tea | Red Rot, Brown Rust, Bacterial Blight, Leaf Curl Virus, Coffee Leaf Rust, Blister Blight |
| **Solanaceae & Tubers** | Tomato, Potato, Pepper, Cassava | Late Blight, Early Blight, Bacterial Spot, Mosaic Disease, Cassava Blight |
| **Fruits & Orchards** | Apple, Grape, Orange, Peach, Banana | Black Sigatoka, Apple Scab, Black Rot, Citrus Greening, Bacterial Spot |
| **Berries & Cucurbits** | Strawberry, Cherry, Squash, Blueberry | Powdery Mildew, Leaf Scorch, Cherry Powdery Mildew |

---

## 🚀 Quickstart

### Prerequisites
- Python 3.10+
- (Optional) Docker

### 1. Clone & Setup Environment
```bash
git clone https://github.com/Manmeet-Singh-Deol/AI-Crop-Predictor.git
cd AI-Crop-Predictor

# Create and activate virtual environment
python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Application
```bash
python run.py
```
Open [http://localhost:8000](http://localhost:8000) in your browser!

---

## 🐳 Docker Deployment

```bash
# Build Docker image
docker build -t agroai-crop-predictor:latest .

# Run container
docker run -d -p 8000:8000 --name agroai_live agroai-crop-predictor:latest
```

---

## 🧪 Testing Suite

Run the automated unit and API endpoint test suites:
```bash
python tests/test_backend.py
python tests/test_api_endpoints.py
python backend/test_accuracy.py
```

---

## 🎓 Model Retraining in Google Colab

To train or fine-tune with your own custom field datasets:
1. Open [`train_colab.ipynb`](train_colab.ipynb) in [Google Colab](https://colab.research.google.com/).
2. Run the notebook with free GPU acceleration.
3. Download `model_weights.pth` and place it in the `backend/` directory.

---

## 📄 License
This project is licensed under the MIT License.
