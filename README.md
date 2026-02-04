# BotModels - AI Inference Service

**Version:** 1.0.0  
**Purpose:** Multimodal AI inference service for General Bots

---

## Overview

BotModels is a Python-based AI inference service that provides multimodal capabilities to the General Bots platform. It serves as a companion to botserver (Rust), specializing in cutting-edge AI/ML models from the Python ecosystem including image generation, video creation, speech synthesis, and vision/captioning.

While botserver handles business logic, networking, and systems-level operations, BotModels exists solely to leverage the extensive Python AI/ML ecosystem for inference tasks that are impractical to implement in Rust.

For comprehensive documentation, see **[docs.pragmatismo.com.br](https://docs.pragmatismo.com.br)** or the **[BotBook](../botbook)** for detailed guides, API references, and tutorials.

---

## Features

- **Image Generation**: Generate images from text prompts using Stable Diffusion
- **Video Generation**: Create short videos from text descriptions using Zeroscope
- **Speech Synthesis**: Text-to-speech using Coqui TTS
- **Speech Recognition**: Audio transcription using OpenAI Whisper
- **Vision/Captioning**: Image and video description using BLIP2

---

## Quick Start

### Installation

```bash
# Clone the repository
cd botmodels

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
.\venv\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt
```

### Configuration

Copy the example environment file and configure:

```bash
cp .env.example .env
```

Edit `.env` with your settings:

```env
HOST=0.0.0.0
PORT=8085
API_KEY=your-secret-key
DEVICE=cuda
IMAGE_MODEL_PATH=./models/stable-diffusion-v1-5
VIDEO_MODEL_PATH=./models/zeroscope-v2
VISION_MODEL_PATH=./models/blip2
```

### Running the Server

```bash
# Development mode
python -m uvicorn src.main:app --host 0.0.0.0 --port 8085 --reload

# Production mode
python -m uvicorn src.main:app --host 0.0.0.0 --port 8085 --workers 4

# With HTTPS (production)
python -m uvicorn src.main:app --host 0.0.0.0 --port 8085 --ssl-keyfile key.pem --ssl-certfile cert.pem
```

---

## 🐍 Philosophy & Scope

### Why Python?

- **Rust vs. Python Rule**:
  - If logic is deterministic, systems-level, or performance-critical: **Do it in Rust (botserver)**
  - If logic requires cutting-edge ML models, rapid experimentation with HuggingFace, or specific Python-only libraries: **Do it here**

### Architecture Principles

- **Inference Only**: This service should NOT hold business state. It accepts inputs, runs inference, and returns predictions.
- **Stateless**: Treated as a sidecar to `botserver`.
- **API First**: Exposes strict HTTP/REST endpoints consumed by `botserver`.

---

## 🛠 Technology Stack

- **Runtime**: Python 3.10+
- **Web Framework**: FastAPI (preferred over Flask for async/performance)
- **ML Frameworks**: PyTorch, HuggingFace Transformers, Diffusers
- **Quality**: `ruff` (linting), `black` (formatting), `mypy` (typing)

---

## 📡 API Endpoints

All endpoints require the `X-API-Key` header for authentication.

### Image Generation

```http
POST /api/image/generate
Content-Type: application/json
X-API-Key: your-api-key

{
  "prompt": "a cute cat playing with yarn",
  "steps": 30,
  "width": 512,
  "height": 512,
  "guidance_scale": 7.5,
  "seed": 42
}
```

### Video Generation

```http
POST /api/video/generate
Content-Type: application/json
X-API-Key: your-api-key

{
  "prompt": "a rocket launching into space",
  "num_frames": 24,
  "fps": 8,
  "steps": 50
}
```

### Speech Generation (TTS)

```http
POST /api/speech/generate
Content-Type: application/json
X-API-Key: your-api-key

{
  "prompt": "Hello, welcome to our service!",
  "voice": "default",
  "language": "en"
}
```

### Speech to Text

```http
POST /api/speech/totext
Content-Type: multipart/form-data
X-API-Key: your-api-key

file: <audio_file>
```

### Image Description

```http
POST /api/vision/describe
Content-Type: multipart/form-data
X-API-Key: your-api-key

file: <image_file>
prompt: "What is in this image?" (optional)
```

### Video Description

```http
POST /api/vision/describe_video
Content-Type: multipart/form-data
X-API-Key: your-api-key

file: <video_file>
num_frames: 8 (optional)
```

### Visual Question Answering

```http
POST /api/vision/vqa
Content-Type: multipart/form-data
X-API-Key: your-api-key

file: <image_file>
question: "How many people are in this image?"
```

### Health Check

```http
GET /api/health
```

Interactive API documentation:
- Swagger UI: `http://localhost:8085/api/docs`
- ReDoc: `http://localhost:8085/api/redoc`

---

## 🔗 Integration with BotServer

### Configuration (config.csv)

```csv
key,value
botmodels-enabled,true
botmodels-host,0.0.0.0
botmodels-port,8085
botmodels-api-key,your-secret-key
botmodels-https,false
image-generator-model,../../../../data/diffusion/sd_turbo_f16.gguf
image-generator-steps,4
image-generator-width,512
image-generator-height,512
video-generator-model,../../../../data/diffusion/zeroscope_v2_576w
video-generator-frames,24
video-generator-fps,8
```

### BASIC Script Keywords

```basic
// Generate an image
file = IMAGE "a beautiful sunset over mountains"
SEND FILE TO user, file

// Generate a video
video = VIDEO "waves crashing on a beach"
SEND FILE TO user, video

// Generate speech
audio = AUDIO "Welcome to General Bots!"
SEND FILE TO user, audio

// Get image/video description
caption = SEE "/path/to/image.jpg"
TALK caption
```

---

## 🏗️ Architecture

```
┌─────────────┐     HTTPS      ┌─────────────┐
│  botserver  │ ────────────▶  │  botmodels  │
│   (Rust)    │                │  (Python)   │
└─────────────┘                └─────────────┘
      │                              │
      │ BASIC Keywords               │ AI Models
      │ - IMAGE                      │ - Stable Diffusion
      │ - VIDEO                      │ - Zeroscope
      │ - AUDIO                      │ - TTS/Whisper
      │ - SEE                        │ - BLIP2
      ▼                              ▼
┌─────────────┐                ┌─────────────┐
│   config    │                │   outputs   │
│   .csv      │                │  (files)    │
└─────────────┘                └─────────────┘
```

---

## ⚡️ Development Guidelines

### Modern Model Usage

- **Deprecate Legacy**: Move away from outdated libs (e.g., old `allennlp`) in favor of **HuggingFace Transformers** and **Diffusers**
- **Quantization**: Always consider quantized models (bitsandbytes, GGUF) to reduce VRAM usage

### Performance & Loading

- **Lazy Loading**: Do NOT load 10GB models at module import time. Load on startup lifecycle or first request with locking
- **GPU Handling**: Robustly detect CUDA/MPS (Mac) and fallback to CPU gracefully

### Code Quality

- **Type Hints**: All functions MUST have type hints
- **Error Handling**: No bare `except:`. Catch precise exceptions and return structured JSON errors to `botserver`

### Project Structure

```
botmodels/
├── src/
│   ├── api/
│   │   ├── v1/
│   │   │   └── endpoints/
│   │   │       ├── image.py
│   │   │       ├── video.py
│   │   │       ├── speech.py
│   │   │       └── vision.py
│   │   └── dependencies.py
│   ├── core/
│   │   ├── config.py
│   │   └── logging.py
│   ├── schemas/
│   │   └── generation.py
│   ├── services/
│   │   ├── image_service.py
│   │   ├── video_service.py
│   │   ├── speech_service.py
│   │   └── vision_service.py
│   └── main.py
├── outputs/
├── models/
├── tests/
├── requirements.txt
└── README.md
```

---

## 🧪 Testing

```bash
pytest tests/
```

---

## 🔒 Security

1. **Always use HTTPS in production**
2. Use strong, unique API keys
3. Restrict network access to the service
4. Consider running on a separate GPU server
5. Monitor resource usage and set appropriate limits

---

## 📚 Documentation

For complete documentation, guides, and API references:

- **[docs.pragmatismo.com.br](https://docs.pragmatismo.com.br)** - Full online documentation
- **[BotBook](../botbook)** - Local comprehensive guide with tutorials and examples
- **[General Bots Repository](https://github.com/GeneralBots/BotServer)** - Main project repository

---

## 📦 Requirements

- Python 3.10+
- CUDA-capable GPU (recommended, 8GB+ VRAM)
- 16GB+ RAM

---

## 🔗 Resources

### Education

- [Computer Vision Course](https://pjreddie.com/courses/computer-vision/)
- [Adversarial VQA Paper](https://arxiv.org/abs/2106.00245)
- [LLM Visualization](https://bbycroft.net/llm)

### References

- [VizWiz VQA PyTorch](https://github.com/DenisDsh/VizWiz-VQA-PyTorch)
- [Diffusers Library](https://github.com/huggingface/diffusers)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [BLIP2](https://huggingface.co/Salesforce/blip2-opt-2.7b)

### Community

- [AI for Mankind](https://github.com/aiformankind)
- [ManaAI](https://manaai.cn/)

---

## 🔑 Remember

- **Inference Only**: No business state, just predictions
- **Modern Models**: Use HuggingFace Transformers, Diffusers
- **Type Safety**: All functions must have type hints
- **Lazy Loading**: Don't load models at import time
- **GPU Detection**: Graceful fallback to CPU
- **Version 1.0.0** - Do not change without approval
- **GIT WORKFLOW** - ALWAYS push to ALL repositories (github, pragmatismo)

---

## 📄 License

See LICENSE file for details.