# Baseer AI: Specialized Computer Vision & Object Recognition
> **Baseer (بصیر)**: *The Discerning Eye* — A high-fidelity image classification system engineered for precision.

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0](https://img.shields.io/badge/PyTorch-2.0-ee4c2c.svg)](https://pytorch.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100+-009688.svg)](https://fastapi.tiangolo.com/)
[![License](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)

**Baseer AI** is an end-to-end computer vision ecosystem designed for production reliability. It features a custom-engineered ResNet architecture with 94.25% validation accuracy, wrapped in an asynchronous FastAPI service with proactive security validators.

---

## 🏗️ Technical Architecture
Baseer AI follows the **Model-Service-Dashboard** pattern:
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
Baseer AI is fully dockerized for environment parity across dev and production.
```bash
docker-compose up --build
```

## 🧪 Security & Quality Assurance
We maintain high engineering standards through automated checks:
- **Linting**: Consistent formatting via `black` and `flake8`.
- **Static Analysis**: Vulnerability scanning via `bandit`.
- **Functional Testing**: `pytest` suite ensuring API and Validator integrity.

## 🌍 Public Deployment
Baseer AI is engineered to be cloud-agnostic and is currently production-ready.

> [!TIP]
> **View Live Production App:** [Baseer AI Dashboard](https://baseer-ai.streamlit.app/)

---
**Author**: Muhammad Hamza Saqib  
**GitHub**: [hamzasaqib80](https://github.com/hamzasaqib80)
