#!/bin/bash
# SE(3) Transformer Drug Discovery Demo Setup Script
# This script automates the setup process for the demo

set -e  # Exit on any error

echo "🚀 SE(3) Transformer Drug Discovery Demo Setup"
echo "=============================================="

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}' | cut -d. -f1-2)
PYTHON_MAJOR=$(echo $PYTHON_VERSION | cut -d. -f1)
PYTHON_MINOR=$(echo $PYTHON_VERSION | cut -d. -f2)

echo "📍 Detected Python $PYTHON_VERSION"

if [ "$PYTHON_MAJOR" -lt 3 ] || [ "$PYTHON_MAJOR" -eq 3 -a "$PYTHON_MINOR" -lt 8 ]; then
    echo "❌ Error: Python 3.8+ required, found $PYTHON_VERSION"
    echo "Please install Python 3.8 or newer"
    exit 1
fi

echo "✅ Python version is compatible"

# Create virtual environment
echo "📦 Creating Python virtual environment..."
if [ ! -d "venv" ]; then
    python3 -m venv venv
    echo "✅ Virtual environment created"
else
    echo "ℹ️  Virtual environment already exists"
fi

# Activate virtual environment
echo "🔧 Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo "🔄 Upgrading pip..."
pip install --upgrade pip

# Detect GPU type
echo "🔍 Detecting GPU acceleration..."
GPU_TYPE="none"

# Check for AMD ROCm
if command -v rocm-smi &> /dev/null; then
    echo "✅ AMD ROCm detected"
    GPU_TYPE="rocm"
elif command -v nvidia-smi &> /dev/null; then
    echo "✅ NVIDIA CUDA detected"
    GPU_TYPE="cuda"
else
    echo "⚠️  No GPU acceleration detected - will use CPU"
    GPU_TYPE="cpu"
fi

# Install PyTorch based on GPU type
echo "🔧 Installing PyTorch..."
case $GPU_TYPE in
    rocm)
        echo "Installing PyTorch with ROCm support..."
        pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
        ;;
    cuda)
        echo "Installing PyTorch with CUDA support..."
        pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
        ;;
    cpu)
        echo "Installing PyTorch CPU version..."
        pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        ;;
esac

# Install DGL
echo "🔧 Installing DGL (Deep Graph Library)..."
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# Install other dependencies
echo "🔧 Installing additional dependencies..."
pip install streamlit numpy pandas matplotlib plotly

# Install molecular visualization (with fallback)
echo "🔧 Installing molecular visualization libraries..."
pip install py3Dmol

# Try to install RDKit
echo "🔧 Installing RDKit..."
if pip install rdkit; then
    echo "✅ RDKit installed successfully"
else
    echo "⚠️  RDKit installation failed with pip, trying alternative..."
    if command -v conda &> /dev/null; then
        conda install -c conda-forge rdkit -y || echo "❌ RDKit installation failed"
    else
        pip install rdkit-pypi || echo "❌ RDKit installation failed"
    fi
fi

# Install optional dependencies
echo "🔧 Installing optional dependencies..."
pip install huggingface_hub scikit-learn scipy

# Check for required files
echo "📁 Checking required files..."

# Check for model file
if [ -f "model_qm9_100_epochs.pth" ]; then
    echo "✅ SE(3) model file found"
else
    echo "⚠️  Model file 'model_qm9_100_epochs.pth' not found"
    echo "📥 You can download it manually or the demo will provide instructions"
fi

# Check for logo file
if [ -f "amd-logo.png" ]; then
    echo "✅ AMD logo file found"
else
    echo "ℹ️  AMD logo not found - text fallback will be used"
fi

# Test installation
echo "🧪 Testing installation..."
python3 -c "
import torch
import dgl
import streamlit
import numpy
import pandas
import plotly

print('✅ Core libraries imported successfully')
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU device: {torch.cuda.get_device_name(0)}')
print(f'DGL version: {dgl.__version__}')

try:
    from rdkit import Chem
    import py3Dmol
    print('✅ Molecular visualization libraries available')
except ImportError:
    print('⚠️  Some molecular visualization libraries missing')
"

# Set environment variables for ROCm if detected
if [ "$GPU_TYPE" = "rocm" ]; then
    echo "🔧 Setting up ROCm environment..."
    cat >> venv/bin/activate << 'EOF'

# ROCm Environment Settings
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
EOF
    echo "✅ ROCm environment variables added to virtual environment"
fi

# Create launch script
echo "📄 Creating launch script..."
cat > run_demo.sh << 'EOF'
#!/bin/bash
# SE(3) Transformer Demo Launcher
source venv/bin/activate
python app.py
EOF

chmod +x run_demo.sh

echo ""
echo "🎉 Setup Complete!"
echo "==================="
echo ""
echo "📋 Next Steps:"
echo "1. Start the demo: ./run_demo.sh"
echo "   or: source venv/bin/activate && python app.py"
echo ""
echo "2. Open your browser to: http://localhost:8501"
echo ""
echo "3. If model file is missing, follow download instructions in the UI"
echo ""
echo "🔧 Manual activation: source venv/bin/activate"
echo "🚀 Manual launch: streamlit run ui_drug_discovery.py"
echo ""
if [ "$GPU_TYPE" != "none" ]; then
    echo "✅ GPU acceleration configured for $GPU_TYPE"
else
    echo "ℹ️  Running in CPU mode - install ROCm or CUDA for GPU acceleration"
fi
echo ""
echo "📖 See README.md for detailed documentation"
