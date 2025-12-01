#!/bin/bash
set -e

echo "🚀 Setting up Sustainable Edge AI development environment..."

# Upgrade pip
echo "📦 Upgrading pip..."
python -m pip install --upgrade pip

# Install main project dependencies
echo "📦 Installing main project dependencies..."
pip install -r requirements.txt

# Install PSI-HDL implementation dependencies
echo "📦 Installing PSI-HDL implementation dependencies..."
pip install -r PSI-HDL-implementation/requirements.txt

# Install Jupyter kernel
echo "📦 Installing Jupyter kernel..."
pip install ipykernel
python -m ipykernel install --user --name sustainable-edge-ai --display-name "Sustainable Edge AI"

# Verify PyTorch installation
echo "🔍 Verifying PyTorch installation..."
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Pre-download any model weights or data if needed
echo "📁 Creating results directory structure..."
mkdir -p results

echo "✅ Development environment setup complete!"
echo ""
echo "🎯 Quick Start:"
echo "   - Run experiments: python experiments/burgers_solar_experiment.py"
echo "   - Generate figures: python generate_figure1_timeline.py"
echo "   - Open Jupyter: jupyter notebook"
echo ""
