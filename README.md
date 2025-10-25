# Sesame TTS RunPod Load Balancing Endpoint

A production-ready text-to-speech service using Sesame CSM 1B model deployed on RunPod with load balancing capabilities.

## Features

- **Zero Cold Starts**: Model is pre-downloaded during Docker build
- **Load Balancing**: Supports multiple concurrent workers
- **Health Checks**: Proper `/ping` endpoint for RunPod load balancer
- **Multiple Output Formats**: Base64 encoded audio or direct audio file download
- **Production Ready**: Proper error handling, logging, and status codes

## API Endpoints

- `GET /ping` - Health check (required by RunPod)
- `GET /` - API information
- `POST /generate` - Generate TTS and return base64 audio
- `POST /generate/audio` - Generate TTS and return audio file

## Quick Start

### 1. Get Hugging Face Access

**IMPORTANT**: The Sesame CSM 1B model is gated. You need to:

1. **Go to**: https://huggingface.co/sesame/csm-1b
2. **Click "Request Access"** and fill out the form
3. **Wait for approval** (usually takes a few hours to days)
4. **Get your Hugging Face token**: https://huggingface.co/settings/tokens

### 2. Deploy to RunPod

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Sesame TTS Load Balancing Endpoint"
   git remote add origin https://github.com/YOUR_USERNAME/sesame-tts-lb.git
   git push -u origin main
   ```

2. **Create RunPod Load Balancing Endpoint**:
   - Go to RunPod.io → Serverless → + New Endpoint
   - Select "Load Balancing" type
   - Connect your GitHub repository
   - Configure GPU settings (RTX 4090, A40, etc.)
   - Set workers: Min 0, Max 3
   - Container disk: 10 GB
   - **Environment variables**: 
     - `PORT=80`
     - `PORT_HEALTH=80`
     - `HUGGINGFACE_TOKEN=your_hf_token_here`
   - Expose HTTP port: 80
   - Idle timeout: 5 seconds

### 2. Test Your Endpoint

Get your endpoint URL and API key from RunPod dashboard.

**Health Check**:
```bash
curl "https://YOUR_ENDPOINT_ID.api.runpod.ai/ping" \
  -H "Authorization: Bearer YOUR_API_KEY"
```

**Generate TTS (Base64)**:
```bash
curl -X POST "https://YOUR_ENDPOINT_ID.api.runpod.ai/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "Hello, this is a test of the Sesame text to speech system.",
    "sample_rate": 24000
  }'
```

**Generate TTS (Audio File)**:
```bash
curl -X POST "https://YOUR_ENDPOINT_ID.api.runpod.ai/generate/audio" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer YOUR_API_KEY" \
  -d '{
    "text": "Hello world",
    "sample_rate": 24000
  }' \
  --output output.wav
```

## Python Client Example

```python
import requests
import base64
import io
from pydantic import BaseModel

class TTSClient:
    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self):
        """Check if the service is healthy"""
        response = requests.get(f"{self.endpoint_url}/ping", headers=self.headers)
        return response.status_code == 200
    
    def generate_tts(self, text: str, sample_rate: int = 24000):
        """Generate TTS and return base64 audio"""
        data = {"text": text, "sample_rate": sample_rate}
        response = requests.post(
            f"{self.endpoint_url}/generate",
            json=data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.json()
    
    def generate_audio_file(self, text: str, sample_rate: int = 24000):
        """Generate TTS and return audio file"""
        data = {"text": text, "sample_rate": sample_rate}
        response = requests.post(
            f"{self.endpoint_url}/generate/audio",
            json=data,
            headers=self.headers
        )
        response.raise_for_status()
        return response.content

# Usage
client = TTSClient("https://YOUR_ENDPOINT_ID.api.runpod.ai", "YOUR_API_KEY")

# Check health
if client.health_check():
    print("Service is healthy!")
    
    # Generate TTS
    result = client.generate_tts("Hello, this is a test.")
    audio_base64 = result["audio_base64"]
    
    # Save audio file
    audio_data = base64.b64decode(audio_base64)
    with open("output.wav", "wb") as f:
        f.write(audio_data)
```

## Architecture

### Model Pre-downloading
The Dockerfile includes a critical step that pre-downloads the Sesame CSM 1B model during the build process. This ensures:
- **Zero cold starts** when workers spin up
- **Faster response times** for first requests
- **Reliable model availability** across all workers

### Load Balancing
- RunPod automatically distributes requests across available workers
- Health checks ensure only healthy workers receive traffic
- Auto-scaling based on demand (0 to N workers)

### Error Handling
- Proper HTTP status codes (503 for model not ready, 400 for bad requests)
- Detailed error messages for debugging
- Graceful handling of model loading failures

## Performance Optimization

1. **Model Caching**: Model is cached in the container image
2. **GPU Optimization**: Uses CUDA when available, falls back to CPU
3. **Memory Management**: Proper tensor cleanup and garbage collection
4. **Audio Processing**: Efficient audio encoding and normalization

## Monitoring

- Health check endpoint for load balancer monitoring
- Detailed logging for debugging
- Status codes for different service states

## Troubleshooting

### Common Issues

1. **Model not loading**: Check GPU availability and CUDA installation
2. **Cold starts**: Ensure model is pre-downloaded in Dockerfile
3. **Memory issues**: Reduce batch size or use smaller model variants
4. **Audio quality**: Adjust sample rate and normalization settings

### Debug Commands

```bash
# Check container logs
docker logs <container_id>

# Test locally
docker build -t sesame-tts .
docker run -p 8000:80 sesame-tts

# Test health endpoint
curl http://localhost:8000/ping
```

## Cost Optimization

- Set appropriate idle timeout (5 seconds recommended)
- Use spot instances for cost savings
- Monitor worker usage and adjust max workers
- Consider model quantization for smaller memory footprint

## Security

- API key authentication required
- Input validation and sanitization
- Rate limiting (configured at RunPod level)
- Secure HTTPS endpoints
