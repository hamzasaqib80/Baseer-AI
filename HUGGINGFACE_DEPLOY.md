# Hub Deployment: Hugging Face Spaces

This guide explains how to host VantageCV on Hugging Face for a permanent, shareable portfolio link.

## 1. Create the Space
1. Go to [huggingface.co/new-space](https://huggingface.co/new-space).
2. **Space Name**: `vantage-cv-classifier` (or your choice).
3. **SDK**: Select **Streamlit**.
4. **Hardware**: Choose **CPU Basic** (Free).
5. **Visibility**: Public.

## 2. Prepare the Files
Hugging Face needs a specific set of files in the root of the repository. You can upload these via their web interface or `git push`.

**Mandatory Files:**
- `app.py` (The entry point)
- `requirements.txt` (Hugging Face checks this to install libraries)
- `models/best_model.pth` (Your trained weights)
- `configs/config.yaml` (The system configuration)
- `src/` (Entire folder containing your model & security logic)

## 3. Important: The Portability Fix
I have already updated your `app.py` with a **Self-Sufficient Fallback**. 
On Hugging Face, there is no separate FastAPI server running in the background. Your dashboard will detect this and automatically load the `InferenceEngine` directly into its own memory. **You do not need to change any code.**

## 4. Troubleshooting
- **Build Errors**: Check the "Logs" tab on Hugging Face. Ensure all libraries (torch, torchvision, pillow) are in `requirements.txt`.
- **Memory Errors**: If the Space restarts, it's likely hitting the 2GB RAM limit. Our ResNet model is lightweight, so this should not be an issue.

---
**Tip**: Once live, add the URL to your GitHub repository's **"About"** section under "Website" to double your engagement rate.
