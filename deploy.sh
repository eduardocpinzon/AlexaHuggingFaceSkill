#!/bin/bash
# Deployment script for Alexa Hugging Face Papers Skill

set -e

echo "=== Building Lambda deployment package ==="

# Create build directory
rm -rf build
mkdir -p build

# Install dependencies
echo "Installing dependencies..."
pip install -r lambda/requirements.txt -t build/ --quiet

# Copy Lambda function
echo "Copying Lambda function..."
cp lambda/lambda_function.py build/

# Create ZIP file
echo "Creating deployment package..."
cd build
zip -r ../lambda_function.zip . -q
cd ..

echo "=== Deployment package created: lambda_function.zip ==="
echo ""
echo "Next steps:"
echo "1. Go to AWS Lambda Console"
echo "2. Create a new function or update existing one"
echo "3. Upload lambda_function.zip"
echo "4. Set environment variables:"
echo "   - ANTHROPIC_API_KEY: Your Anthropic API key"
echo "   - HF_API_TOKEN: (Optional) Your Hugging Face token"
echo "5. Set timeout to 30 seconds and memory to 256MB"
echo "6. Add Alexa Skills Kit trigger"
