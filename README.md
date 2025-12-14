# AgriVision 🌿
Dataset link:
https://huggingface.co/datasets/SyedNazmusSakib/PlantVillageVQA/tree/main


Report link:
https://drive.google.com/file/d/1SLrZgWSCadmc3iY06wYlB3Blz1GDy4qI/view?usp=sharing

A sophisticated AI-powered plant disease diagnosis tool that combines the beauty of nature with the power of modern deep learning.

## Features
- **Visual Diagnosis**: Upload plant images to detect diseases using CLIP-based vision-language models.
- **Natural Language Interaction**: Ask questions about your plants and receive generated insights.
- **Immersive UI**: A React-based interface featuring glassmorphism, 3D particles, and serene animations.

## Tech Stack
- **Frontend**: React, Tailwind CSS, Framer Motion, Vite
- **Backend**: Python, FastAPI, PyTorch
- **AI**: Hugging Face Transformers, CLIP, GRU

## Setup & Running

### Prerequisites
- Python 3.8+
- Node.js & npm

### 1. Backend Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Start the API server
python main.py
```

### 2. Frontend Setup
```bash
cd frontend
npm install
npm run dev
```

## Model
This project requires a trained model checkpoint (`model_epoch_38.pth`) and vocabulary (`vocab.json`). 
- Place these files in the root directory.
- `vocab.json` is included in this repository.
- `model_epoch_38.pth` must be downloaded from Hugging Face (see Deployment Guide).
