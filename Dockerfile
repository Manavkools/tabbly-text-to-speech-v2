FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip first
RUN pip install --upgrade pip

# Clone the official Sesame CSM repository
RUN git clone https://github.com/SesameAILabs/csm.git sesame_csm

# Copy our FastAPI app
COPY app.py .

# Install dependencies step by step to avoid conflicts
RUN pip install fastapi uvicorn[standard] pydantic

# Install CSM dependencies manually to avoid conflicts
RUN pip install --no-deps \
    huggingface_hub \
    transformers \
    tokenizers \
    torch \
    torchaudio \
    numpy \
    scipy \
    librosa \
    soundfile \
    sentencepiece \
    protobuf \
    tqdm \
    fsspec \
    requests \
    packaging \
    pyyaml \
    filelock \
    typing-extensions \
    regex \
    safetensors \
    accelerate

# Install CSM package without dependencies
RUN cd sesame_csm && pip install -e . --no-deps

# Set environment variables
ENV PORT=80
ENV PORT_HEALTH=80
ENV NO_TORCH_COMPILE=1

# Expose port
EXPOSE 80

# Start server
CMD ["python", "app.py"]
