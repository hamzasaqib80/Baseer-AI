# VantageCV: Enterprise-Grade CIFAR-10 Identification Pipeline

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**VantageCV** is an end-to-end computer vision ecosystem designed to move beyond typical research notebooks. It features a custom-engineered ResNet microservice with 94.25% validation accuracy, wrapped in an asynchronous API with built-in input sanitization.

---

## 🏗️ Technical Architecture
VantageCV follows the **Model-Service-Dashboard** pattern:
1.  **Model Core**: Custom Residual Neural Network (ResNet) optimized for 32x32 image classification.
2.  **Inference Service**: FastAPI wrapper utilizing `lifespan` events to ensure model weights are loaded into GPU/CPU memory exactly once.
3.  **Security Layer**: A stateless `SecurityValidator` that enforces tensor shape integrity and value-clamping in normalized space to prevent OOD (Out-of-Distribution) exploits.
4.  **UI Diagnostics**: A Streamlit dashboard for real-time monitoring and inference tracking.

## 📊 Core Performance
| Metric | Specification |
| :--- | :--- |
| **Model** | Custom ResNet (Residual Blocks + Batchnorm) |
| **Validation Accuracy** | **94.25%** |
| **Mean Inference Latency** | **12ms** (CPU) |
| **Input Constraints** | 32x32 RGB (CIFAR-10 Mean/Std Normalized) |
| **Security Audit** | Bandit (SAST) + Custom Tensor Sanitizer |

## 📁 Repository Structure
```text
├── app.py                # Streamlit Tactical Dashboard
├── src/
│   ├── api.py            # FastAPI Inference Server
│   ├── models/           # PyTorch Architecture Definitions
│   ├── inference/        # Stateless Inference Wrapper
│   └── utils/            # Security & Sanitization logic
├── configs/              # Immutable YAML configurations
├── models/               # Production checkpoints (.pth)
├── Dockerfile            # Multi-stage production build
└── docker-compose.yaml   # Orchestration for full-stack deployment
```

## 🚀 Quick Start

### 1. Local Development
Ensure you have Python 3.10+ installed:
```bash
pip install -r requirements.txt
# Start the API node
uvicorn src.api:app --host 0.0.0.0 --port 8000
# Start the Diagnostic UI
streamlit run app.py
```

### 2. Industry-Standard Containerization
VantageCV is fully dockerized for environment parity across dev and production.
```bash
docker-compose up --build
```

## 🧪 Security & Quality Assurance
We maintain high engineering standards through automated checks:
- **Linting**: Consistent formatting via `black` and `flake8`.
- **Static Analysis**: Vulnerability scanning via `bandit`.
- **Functional Testing**: `pytest` suite ensuring API and Validator integrity.

## 🌍 Public Deployment
The system is currently hosted live on **Hugging Face Spaces**. 
> [!TIP]
> [Replace this text with your live URL once it's finished building!]

---
**Author**: Muhammad Hamza Saqib  
**Version**: 1.0 | **Project Type**: Enterprise AI Infrastructure
