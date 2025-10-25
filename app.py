from fastapi import FastAPI, HTTPException
from fastapi.responses import Response
from pydantic import BaseModel
import torch
import torchaudio
import base64
import io
import os
import logging
from transformers import AutoTokenizer, AutoModel
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Sesame TTS API", version="1.0.0")

# Global variables for model
model = None
tokenizer = None
device = None
is_ready = False

class TTSRequest(BaseModel):
    text: str
    sample_rate: int = 24000

class TTSResponse(BaseModel):
    audio_base64: str
    sample_rate: int
    format: str = "wav"

def load_model():
    """Load the Sesame CSM 1B model"""
    global model, tokenizer, device, is_ready
    
    try:
        logger.info("Loading Sesame CSM 1B model...")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {device}")
        
        model_name = "sesame/csm-1b"
        logger.info(f"Loading tokenizer from {model_name}")
        
        # Get Hugging Face token from environment
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        if not hf_token:
            logger.warning("No HUGGINGFACE_TOKEN found. Model access may be restricted.")
        
        # Load tokenizer with retry logic
        max_retries = 3
        for attempt in range(max_retries):
            try:
                tokenizer = AutoTokenizer.from_pretrained(
                    model_name,
                    token=hf_token
                )
                logger.info("✓ Tokenizer loaded successfully")
                break
            except Exception as e:
                logger.warning(f"Tokenizer load attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                import time
                time.sleep(5)
        
        logger.info(f"Loading model from {model_name}")
        # Load model with retry logic
        for attempt in range(max_retries):
            try:
                model = AutoModel.from_pretrained(
                    model_name,
                    torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
                    trust_remote_code=True,
                    token=hf_token
                ).to(device)
                model.eval()
                logger.info("✓ Model loaded successfully")
                break
            except Exception as e:
                logger.warning(f"Model load attempt {attempt + 1} failed: {e}")
                if attempt == max_retries - 1:
                    raise
                import time
                time.sleep(10)
        
        is_ready = True
        logger.info("✓ Model loading completed successfully")
        
    except Exception as e:
        logger.error(f"✗ Error loading model: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        is_ready = False
        raise

# Load model on startup - this will download the model when worker starts
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
    if model is None:
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
    
    Returns:
        Base64-encoded WAV audio file
    """
    if not is_ready or model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        logger.info(f"Generating TTS for text: {request.text[:50]}...")
        
        # Generate audio using the model
        with torch.no_grad():
            # Tokenize the input text
            inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # Generate audio using the model
            if hasattr(model, 'generate'):
                audio_output = model.generate(**inputs)
            else:
                # If model doesn't have generate method, use forward pass
                outputs = model(**inputs)
                audio_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            # Convert to numpy array if needed
            if isinstance(audio_output, torch.Tensor):
                audio_tensor = audio_output.cpu()
            else:
                audio_tensor = torch.from_numpy(audio_output)
            
            # Ensure correct shape for audio (1, samples)
            if audio_tensor.dim() > 2:
                audio_tensor = audio_tensor.squeeze()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Normalize audio to prevent clipping
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
        
        # Convert to WAV format
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, request.sample_rate, format="wav")
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
    if not is_ready or model is None:
        raise HTTPException(status_code=503, detail="Model not ready")
    
    if not request.text or len(request.text.strip()) == 0:
        raise HTTPException(status_code=400, detail="Text cannot be empty")
    
    try:
        logger.info(f"Generating TTS audio for: {request.text[:50]}...")
        
        # Generate audio using the model
        with torch.no_grad():
            # Tokenize the input text
            inputs = tokenizer(request.text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
            
            # Generate audio using the model
            if hasattr(model, 'generate'):
                audio_output = model.generate(**inputs)
            else:
                # If model doesn't have generate method, use forward pass
                outputs = model(**inputs)
                audio_output = outputs.last_hidden_state if hasattr(outputs, 'last_hidden_state') else outputs[0]
            
            # Convert to numpy array if needed
            if isinstance(audio_output, torch.Tensor):
                audio_tensor = audio_output.cpu()
            else:
                audio_tensor = torch.from_numpy(audio_output)
            
            # Ensure correct shape for audio (1, samples)
            if audio_tensor.dim() > 2:
                audio_tensor = audio_tensor.squeeze()
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)
            
            # Normalize audio to prevent clipping
            audio_tensor = audio_tensor / torch.max(torch.abs(audio_tensor))
        
        # Convert to WAV
        buffer = io.BytesIO()
        torchaudio.save(buffer, audio_tensor, request.sample_rate, format="wav")
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
