#!/bin/bash

# Sesame TTS RunPod Deployment Script
# This script helps you deploy the Sesame TTS service to RunPod

set -e

echo "🚀 Sesame TTS RunPod Deployment Script"
echo "======================================"

# Check if required tools are installed
check_dependencies() {
    echo "Checking dependencies..."
    
    if ! command -v git &> /dev/null; then
        echo "❌ Git is not installed. Please install Git first."
        exit 1
    fi
    
    if ! command -v docker &> /dev/null; then
        echo "❌ Docker is not installed. Please install Docker first."
        exit 1
    fi
    
    echo "✓ All dependencies are installed"
}

# Initialize git repository
init_git() {
    echo "Initializing Git repository..."
    
    if [ ! -d ".git" ]; then
        git init
        echo "✓ Git repository initialized"
    else
        echo "✓ Git repository already exists"
    fi
}

# Add and commit files
commit_files() {
    echo "Adding files to Git..."
    
    git add .
    git commit -m "Sesame TTS Load Balancing Endpoint - Initial commit" || echo "No changes to commit"
    
    echo "✓ Files committed to Git"
}

# Test Docker build locally
test_build() {
    echo "Testing Docker build locally..."
    
    if docker build -t sesame-tts-test .; then
        echo "✓ Docker build successful"
    else
        echo "❌ Docker build failed"
        exit 1
    fi
}

# Main deployment function
main() {
    echo "Starting deployment process..."
    echo
    
    # Check dependencies
    check_dependencies
    echo
    
    # Initialize git
    init_git
    echo
    
    # Commit files
    commit_files
    echo
    
    # Test build
    test_build
    echo
    
    echo "🎉 Local setup completed successfully!"
    echo
    echo "Next steps:"
    echo "1. Push to GitHub:"
    echo "   git remote add origin https://github.com/YOUR_USERNAME/sesame-tts-lb.git"
    echo "   git push -u origin main"
    echo
    echo "2. Create RunPod Load Balancing Endpoint:"
    echo "   - Go to RunPod.io → Serverless → + New Endpoint"
    echo "   - Select 'Load Balancing' type"
    echo "   - Connect your GitHub repository"
    echo "   - Configure GPU settings (RTX 4090, A40, etc.)"
    echo "   - Set workers: Min 0, Max 3"
    echo "   - Container disk: 10 GB"
    echo "   - Environment variables: PORT=80, PORT_HEALTH=80"
    echo "   - Expose HTTP port: 80"
    echo "   - Idle timeout: 5 seconds"
    echo
    echo "3. Test your endpoint:"
    echo "   python test_client.py --endpoint https://YOUR_ENDPOINT_ID.api.runpod.ai --api-key YOUR_API_KEY"
    echo
    echo "📚 For detailed instructions, see README.md"
}

# Run main function
main
