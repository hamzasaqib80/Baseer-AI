# 🚀 CIFAR-10 Enterprise Deployment Guide

This document outlines the professional workflows for deploying the CIFAR-10 Classifier system to production environments.

## 1. Local Containerized Deployment (Docker)
Use this to mirror a cloud production environment on your local machine.

```bash
# Build and launch the stack
docker-compose up --build -d

# Verify logs
docker logs -f cifar10_inference_service

# Access the tactical dashboard
# http://localhost:8501
```

## 2. Live Demo: Hugging Face Spaces (Recommended)
This is the best way to get a **shareable link** for your CV/LinkedIn.

### Steps:
1. Create a **New Space** on [Hugging Face](https://huggingface.co/spaces).
2. Select **Streamlit** as the SDK.
3. Upload the following files from this repo:
   - `app.py`
   - `requirements.txt`
   - `src/` (Entire directory)
   - `models/best_model.pth`
   - `configs/config.yaml`
4. **Note**: For Hugging Face, the app will load the model directly into memory using your `InferenceEngine` class.

## 3. High-Traffic Production (AWS / GCP / Azure)
For professional cloud hosting using the included Docker assets.

### Infrastructure Pattern:
- **Registry**: Push image to AWS ECR or Docker Hub.
- **Service**: Deploy image to AWS Fargate, Google Cloud Run, or Azure Container Apps.
- **Security**: The `SecurityValidator.py` will automatically protect your cloud endpoint from malformed tensor attacks.

## 4. CI/CD Operations
Your `.github/workflows/ci.yml` is already configured to verify code integrity on every push. 
