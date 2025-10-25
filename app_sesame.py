from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import torch
import torchaudio
import base64
import io
import os
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sesame TTS API", version="1.0.0")

# Global variables for model
generator = None
is_ready = False

class TTSRequest(BaseModel):
    text: str
    sample_rate: int = 24000
    speaker: int = 0
    max_audio_length_ms: int = 10000

class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    format: str = "wav"

def load_model():
    """Load the Sesame CSM 1B model using the official generator"""
    global generator, is_ready
    
    try:
        logger.info("Loading Sesame CSM 1B model using official generator...")
        
        # Import the official CSM generator
        from generator import load_csm_1b
        
        # Determine device
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
        
        logger.info(f"Using device: {device}")
        
        # Load the generator
        generator = load_csm_1b(device=device)
        
        is_ready = True
        logger.info("✓ Model loaded successfully using official generator")
        
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        is_ready = False
        raise

# Load model on startup
@app.on_event("startup")
async def startup_event():
    logger.info("🚀 Starting model download on worker startup...")
    try:
        load_model()
        logger.info("✅ Model ready - worker can handle requests immediately!")
    except Exception as e:
        logger.error(f"❌ Failed to load model on startup: {e}")
        # Don't exit - let the worker start but mark as not ready

# REQUIRED: Health check endpoint for RunPod load balancer
@app.get("/ping")
async def health_check():
    """
    Health check endpoint - REQUIRED by RunPod
    Returns:
    - 204: Model is initializing
    - 200: Model is ready
    - 503: Model failed to load
    """
    if generator is None:
        # Model is still initializing
        return Response(status_code=204)
    
    if is_ready:
        # Model is ready
        return {"status": "healthy", "model_loaded": True}
    else:
        # Model failed to load
        return Response(status_code=503)

@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "name": "Sesame TTS API",
        "version": "1.0.0",
        "status": "healthy" if is_ready else "initializing",
        "endpoints": {
            "health": "/ping",
            "generate_tts": "/generate",
            "generate_audio_file": "/generate/audio"
        }
    }

@app.post("/generate", response_model=TTSResponse)
async def generate_tts(request: TTSRequest):
    """
    Generate speech from text and return base64-encoded audio
    
    Args:
        text: The text to convert to speech
        sample_rate: Output sample rate (default: 24000)
        speaker: Speaker ID (default: 0)
        max_audio_length_ms: Maximum audio length in milliseconds (default: 10000)
    
    Returns:
        Base64-encoded WAV audio file
    """
    if not is_ready or generator is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        logger.info(f"Generating TTS for text: {request.text[:50]}...")
        
        # Generate audio using the official generator
        with torch.no_grad():
            audio = generator.generate(
                text=request.text,
                speaker=request.speaker,
                context=[],
                max_audio_length_ms=request.max_audio_length_ms,
            )
            
            # Ensure correct shape for audio (1, samples)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Normalize audio to prevent clipping
            audio = audio / torch.max(torch.abs(audio))
        
        # Convert to WAV format
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.cpu(), request.sample_rate, format="wav")
        buffer.seek(0)
        
        # Encode to base64
        audio_base64 = base64.b64encode(buffer.read()).decode('utf-8')
        
        logger.info("✓ TTS generation completed successfully")
        return TTSResponse(
            audio_base64=audio_base64,
            sample_rate=request.sample_rate,
            format="wav"
        )
        
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

@app.post("/generate/audio")
async def generate_tts_audio(request: TTSRequest):
    """
    Generate speech from text and return audio file directly
    
    Returns:
        WAV audio file (audio/wav)
    """
    if not is_ready or generator is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        logger.info(f"Generating TTS audio for: {request.text[:50]}...")
        
        # Generate audio using the official generator
        with torch.no_grad():
            audio = generator.generate(
                text=request.text,
                speaker=request.speaker,
                context=[],
                max_audio_length_ms=request.max_audio_length_ms,
            )
            
            # Ensure correct shape for audio (1, samples)
            if audio.dim() == 1:
                audio = audio.unsqueeze(0)
            
            # Normalize audio to prevent clipping
            audio = audio / torch.max(torch.abs(audio))
        
        # Convert to WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio.cpu(), request.sample_rate, format="wav")
        buffer.seek(0)
        
        logger.info("✓ TTS audio generation completed successfully")
        return Response(
            content=buffer.read(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav"
            }
        )
        
    except Exception as e:
        logger.error(f"Error generating TTS: {e}")
        raise HTTPException(status_code=500, detail=f"TTS generation failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    
    # Get port from environment variable (RunPod sets this)
    port = int(os.getenv("PORT", "80"))
    
    logger.info(f"Starting server on port {port}...")
    uvicorn.run(app, host="0.0.0.0", port=port)
