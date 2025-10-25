FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app.py .

# Pre-download and cache the Sesame CSM 1B model
# This is CRITICAL to avoid cold starts on RunPod
RUN python -c "\
import torch; \
import os; \
from transformers import AutoTokenizer, AutoModel; \
print('Starting model download...'); \
print(f'CUDA available: {torch.cuda.is_available()}'); \
print(f'Device count: {torch.cuda.device_count() if torch.cuda.is_available() else 0}'); \
model_name = 'sesame/csm-1b'; \
print(f'Downloading tokenizer from {model_name}...'); \
tokenizer = AutoTokenizer.from_pretrained(model_name); \
print('Tokenizer downloaded successfully'); \
print(f'Downloading model from {model_name}...'); \
model = AutoModel.from_pretrained(model_name, trust_remote_code=True); \
print('Model downloaded successfully'); \
print('Model cache location:', os.path.expanduser('~/.cache/huggingface/transformers')); \
print('Model download completed successfully!'); \
"

# Verify model files are cached
RUN python -c "\
import os; \
cache_dir = os.path.expanduser('~/.cache/huggingface/transformers'); \
print(f'Cache directory exists: {os.path.exists(cache_dir)}'); \
if os.path.exists(cache_dir): \
    print('Cache contents:'); \
    for root, dirs, files in os.walk(cache_dir): \
        level = root.replace(cache_dir, '').count(os.sep); \
        indent = ' ' * 2 * level; \
        print(f'{indent}{os.path.basename(root)}/'); \
        subindent = ' ' * 2 * (level + 1); \
        for file in files[:5]: \
            print(f'{subindent}{file}'); \
        if len(files) > 5: \
            print(f'{subindent}... and {len(files) - 5} more files'); \
"

# Expose port 80 (default for RunPod)
EXPOSE 80

# Set environment variables
ENV PORT=80
ENV PORT_HEALTH=80

# Start the FastAPI server
CMD ["python", "app.py"]
