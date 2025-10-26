FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    libsndfile1 \
    ffmpeg \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Clone the official Sesame CSM repository
RUN git clone https://github.com/SesameAILabs/csm.git sesame_csm

# Copy our FastAPI app
COPY app.py .

# Install the CSM package first (this will install its dependencies)
RUN cd sesame_csm && pip install -e .

# Install additional FastAPI dependencies
RUN pip install fastapi uvicorn[standard] pydantic

# Set environment variables
ENV PORT=80
ENV PORT_HEALTH=80
ENV NO_TORCH_COMPILE=1

# Expose port
EXPOSE 80

# Start server
CMD ["python", "app.py"]
