#!/usr/bin/env python3
"""
Test client for Sesame TTS RunPod Load Balancing Endpoint
"""

import requests
import base64
import json
import time
import argparse
from typing import Optional

class TTSClient:
    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url.rstrip('/')
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
    
    def health_check(self, max_retries: int = 10, delay: float = 2.0) -> bool:
        """Check if the service is healthy with retries"""
        print("Checking service health...")
        
        for attempt in range(max_retries):
            try:
                response = requests.get(f"{self.endpoint_url}/ping", headers=self.headers, timeout=10)
                
                if response.status_code == 200:
                    print("✓ Service is healthy!")
                    return True
                elif response.status_code == 204:
                    print(f"⏳ Service is initializing... (attempt {attempt + 1}/{max_retries})")
                elif response.status_code == 503:
                    print(f"❌ Service failed to load model (attempt {attempt + 1}/{max_retries})")
                else:
                    print(f"⚠️  Unexpected status code: {response.status_code}")
                
                if attempt < max_retries - 1:
                    time.sleep(delay)
                    
            except requests.exceptions.RequestException as e:
                print(f"❌ Request failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(delay)
        
        print("❌ Service health check failed after all retries")
        return False
    
    def get_info(self) -> Optional[dict]:
        """Get API information"""
        try:
            response = requests.get(f"{self.endpoint_url}/", headers=self.headers, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"❌ Failed to get API info: {e}")
            return None
    
    def generate_tts(self, text: str, sample_rate: int = 24000) -> Optional[dict]:
        """Generate TTS and return base64 audio"""
        data = {"text": text, "sample_rate": sample_rate}
        
        try:
            print(f"Generating TTS for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            response = requests.post(
                f"{self.endpoint_url}/generate",
                json=data,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            result = response.json()
            print("✓ TTS generation successful!")
            return result
        except requests.exceptions.RequestException as e:
            print(f"❌ TTS generation failed: {e}")
            return None
    
    def generate_audio_file(self, text: str, sample_rate: int = 24000, filename: str = "output.wav") -> bool:
        """Generate TTS and save as audio file"""
        data = {"text": text, "sample_rate": sample_rate}
        
        try:
            print(f"Generating audio file for: '{text[:50]}{'...' if len(text) > 50 else ''}'")
            response = requests.post(
                f"{self.endpoint_url}/generate/audio",
                json=data,
                headers=self.headers,
                timeout=30
            )
            response.raise_for_status()
            
            with open(filename, "wb") as f:
                f.write(response.content)
            
            print(f"✓ Audio file saved as: {filename}")
            return True
        except requests.exceptions.RequestException as e:
            print(f"❌ Audio file generation failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description="Test Sesame TTS RunPod Endpoint")
    parser.add_argument("--endpoint", required=True, help="RunPod endpoint URL")
    parser.add_argument("--api-key", required=True, help="RunPod API key")
    parser.add_argument("--text", default="Hello, this is a test of the Sesame text to speech system.", help="Text to convert to speech")
    parser.add_argument("--sample-rate", type=int, default=24000, help="Audio sample rate")
    parser.add_argument("--output", default="test_output.wav", help="Output audio file name")
    
    args = parser.parse_args()
    
    # Initialize client
    client = TTSClient(args.endpoint, args.api_key)
    
    # Get API info
    print("Getting API information...")
    info = client.get_info()
    if info:
        print(f"API Name: {info.get('name', 'Unknown')}")
        print(f"Version: {info.get('version', 'Unknown')}")
        print(f"Status: {info.get('status', 'Unknown')}")
        print()
    
    # Health check
    if not client.health_check():
        print("❌ Service is not healthy. Exiting.")
        return
    
    print()
    
    # Test TTS generation (base64)
    print("Testing TTS generation (base64)...")
    result = client.generate_tts(args.text, args.sample_rate)
    if result:
        print(f"Audio format: {result.get('format', 'Unknown')}")
        print(f"Sample rate: {result.get('sample_rate', 'Unknown')}")
        print(f"Base64 length: {len(result.get('audio_base64', ''))}")
    
    print()
    
    # Test TTS generation (audio file)
    print("Testing TTS generation (audio file)...")
    success = client.generate_audio_file(args.text, args.sample_rate, args.output)
    
    if success:
        print(f"\n🎉 All tests completed successfully!")
        print(f"📁 Audio file saved as: {args.output}")
    else:
        print("\n❌ Some tests failed.")

if __name__ == "__main__":
    main()
