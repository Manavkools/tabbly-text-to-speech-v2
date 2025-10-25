#!/bin/bash

# Sesame TTS RunPod API Test Script
# Replace YOUR_ENDPOINT_ID and YOUR_API_KEY with your actual values

# Configuration - UPDATE THESE VALUES
ENDPOINT_ID="YOUR_ENDPOINT_ID"
API_KEY="YOUR_API_KEY"
BASE_URL="https://${ENDPOINT_ID}.api.runpod.ai"

echo "🧪 Testing Sesame TTS RunPod API"
echo "================================"
echo "Base URL: $BASE_URL"
echo

# Test 1: Health Check
echo "1️⃣ Testing Health Check..."
echo "Command: curl \"$BASE_URL/ping\" -H \"Authorization: Bearer $API_KEY\""
echo

curl -s -w "\nHTTP Status: %{http_code}\n" \
  "$BASE_URL/ping" \
  -H "Authorization: Bearer $API_KEY"

echo
echo "---"
echo

# Test 2: API Information
echo "2️⃣ Testing API Information..."
echo "Command: curl \"$BASE_URL/\" -H \"Authorization: Bearer $API_KEY\""
echo

curl -s -w "\nHTTP Status: %{http_code}\n" \
  "$BASE_URL/" \
  -H "Authorization: Bearer $API_KEY" | jq '.' 2>/dev/null || cat

echo
echo "---"
echo

# Test 3: Generate TTS (Base64)
echo "3️⃣ Testing TTS Generation (Base64)..."
echo "Command: curl -X POST \"$BASE_URL/generate\" -H \"Content-Type: application/json\" -H \"Authorization: Bearer $API_KEY\" -d '{\"text\": \"Hello, this is a test of the Sesame text to speech system.\", \"sample_rate\": 24000}'"
echo

curl -s -w "\nHTTP Status: %{http_code}\n" \
  -X POST "$BASE_URL/generate" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "text": "Hello, this is a test of the Sesame text to speech system.",
    "sample_rate": 24000
  }' | jq '.' 2>/dev/null || cat

echo
echo "---"
echo

# Test 4: Generate TTS (Audio File)
echo "4️⃣ Testing TTS Generation (Audio File)..."
echo "Command: curl -X POST \"$BASE_URL/generate/audio\" -H \"Content-Type: application/json\" -H \"Authorization: Bearer $API_KEY\" -d '{\"text\": \"Hello world\", \"sample_rate\": 24000}' --output test_output.wav"
echo

curl -s -w "\nHTTP Status: %{http_code}\n" \
  -X POST "$BASE_URL/generate/audio" \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer $API_KEY" \
  -d '{
    "text": "Hello world",
    "sample_rate": 24000
  }' \
  --output test_output.wav

if [ -f "test_output.wav" ]; then
    echo "✅ Audio file saved as: test_output.wav"
    echo "File size: $(ls -lh test_output.wav | awk '{print $5}')"
else
    echo "❌ Failed to save audio file"
fi

echo
echo "🎉 All tests completed!"
echo
echo "📝 Expected Responses:"
echo "- Health Check: 200 OK with {\"status\": \"healthy\", \"model_loaded\": true}"
echo "- API Info: 200 OK with service information"
echo "- TTS Base64: 200 OK with {\"audio_base64\": \"...\", \"sample_rate\": 24000, \"format\": \"wav\"}"
echo "- TTS Audio: 200 OK with binary WAV file"
