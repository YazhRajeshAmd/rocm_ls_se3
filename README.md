# SE(3) Transformer Drug Discovery Demo

## 🧬 AMD ROCm Life Sciences: GPU-Accelerated Drug Discovery & AI

This repository contains a comprehensive drug discovery demonstration using SE(3) Transformer for molecular property prediction with AMD ROCm GPU acceleration.

## 🎯 Features

- **SE(3) Equivariant Transformer** for molecular property prediction
- **Interactive 3D molecule visualization** with RDKit and py3Dmol
- **AMD ROCm GPU acceleration** for high-performance computing
- **QM9 dataset integration** for molecular analysis
- **Professional enterprise UI** matching AMD ROCm Life Sciences styling
- **Real-time molecular property predictions** (HOMO, LUMO, dipole moment, etc.)

## 🛠️ Prerequisites

### System Requirements
- **Linux** (Ubuntu 20.04+ recommended)
- **AMD ROCm** 5.0+ or **NVIDIA CUDA** 11.0+ (for GPU acceleration)
- **Python** 3.8-3.11
- **Git** (for repository management)

### Hardware Requirements
- **GPU**: AMD MI300X, MI250X, or NVIDIA RTX/Tesla series recommended
- **RAM**: 8GB+ system memory
- **Storage**: 5GB+ free space for datasets and models

## 📦 Installation

### 1. Clone and Setup Environment

```bash
# Clone the repository (if not already done)
cd /path/to/rocm-ls-se3

# Create and activate Python virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Upgrade pip
pip install --upgrade pip
```

### 2. Install Core Dependencies

```bash
# Install PyTorch with ROCm support (for AMD GPUs)
pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6

# For NVIDIA GPUs, use CUDA version instead:
# pip install torch==2.1.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install DGL for graph neural networks
pip install dgl -f https://data.dgl.ai/wheels/repo.html

# Install core scientific computing packages
pip install numpy pandas matplotlib plotly

# Install Streamlit for web interface
pip install streamlit

# Install molecular visualization dependencies
pip install rdkit py3dmol

# Alternative installation for RDKit (if above fails)
# conda install -c conda-forge rdkit
```

### 3. Verify ROCm/CUDA Installation

```bash
# For AMD ROCm
rocm-smi

# For NVIDIA CUDA
nvidia-smi

# Test PyTorch GPU access
python -c "import torch; print('CUDA available:', torch.cuda.is_available()); print('Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

## 📁 Required Files and Datasets

### 1. Get the Pretrained SE(3) Transformer Model

The model file `model_qm9_100_epochs.pth` should already be present. If not, download it:

```bash
# Option 1: Download from Hugging Face (if available)
pip install huggingface_hub
huggingface-cli download amd/se3_transformers model_qm9_100_epochs.pth --local-dir ./

# Option 2: Alternative download locations
# Check with AMD for the official model repository
# wget https://path.to.model/model_qm9_100_epochs.pth

# Verify the model file exists
ls -la model_qm9_100_epochs.pth
```

### 2. QM9 Dataset Files

The QM9 dataset files should be present:

```bash
# Check existing QM9 files
ls -la qm9.tar.*

# The demo automatically downloads QM9 through DGL, but having local files can speed up loading
# If files are missing, DGL will download them automatically on first run
```

### 3. PBPP-2020 Dataset (Optional Enhancement)

```bash
# The pbpp-2020.zip file should be present for enhanced molecular analysis
ls -la pbpp-2020.zip

# If missing, you can download from PDBbind database
# wget http://www.pdbbind.org.cn/download/pdbbind_2020.tar.gz
# unzip pbpp-2020.zip
```

### 4. AMD Logo (For UI)

```bash
# Verify AMD logo exists for professional interface
ls -la amd-logo.png

# If missing, the interface will display text fallback
```

## 🚀 Running the Demo

### Method 1: Using the Launcher Script

```bash
# Simple launch
python app.py
```

### Method 2: Direct Streamlit Launch

```bash
# Activate environment
source venv/bin/activate

# Set ROCm environment variables (for AMD GPUs)
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0

# Launch Streamlit application
streamlit run ui_drug_discovery.py --server.port 8501 --server.address 0.0.0.0
```

### Method 3: Background Launch for Servers

```bash
# Launch in background for server deployment
nohup streamlit run ui_drug_discovery.py --server.port 8501 --server.address 0.0.0.0 > streamlit.log 2>&1 &
```

## 🌐 Accessing the Demo

1. **Local Access**: Open your browser to `http://localhost:8501`
2. **Remote Access**: `http://your-server-ip:8501`
3. **Development**: The interface will auto-reload when you make code changes

## 📋 Demo Walkthrough

### 1. System Status (Left Panel)
- **Production System**: Verify SE(3) Transformer model is loaded
- **Hardware Status**: Confirm GPU acceleration is active
- **Dataset Pipeline**: Check QM9 dataset accessibility

### 2. Overview Tab
- SE(3) Transformer architecture details
- Key capabilities and performance metrics
- 3D molecular visualization preview

### 3. Predictions Tab
- Load QM9 molecular samples
- Select molecules for analysis
- Interactive 3D molecular structure viewer
- Real-time property predictions (HOMO, LUMO, gap, dipole, polarizability)

### 4. Analysis Tab
- Detailed performance metrics
- Model accuracy assessment
- Export functionality

### 5. Performance Tab
- GPU memory monitoring
- Performance benchmarks
- Console output tracking

## 🔧 Troubleshooting

### Common Issues

#### GPU Not Detected
```bash
# Check ROCm installation
ls /opt/rocm
rocm-smi

# Check PyTorch ROCm support
python -c "import torch; print('ROCm available:', torch.cuda.is_available())"

# Reinstall PyTorch with ROCm
pip uninstall torch
pip install torch==2.1.0 --index-url https://download.pytorch.org/whl/rocm5.6
```

#### DGL Installation Issues
```bash
# Reinstall DGL with specific backend
pip uninstall dgl
pip install dgl -f https://data.dgl.ai/wheels/cu118/repo.html  # For CUDA
# or
pip install dgl -f https://data.dgl.ai/wheels/repo.html        # For CPU/ROCm
```

#### RDKit/py3Dmol Visualization Issues
```bash
# Alternative RDKit installation
conda install -c conda-forge rdkit

# If conda is not available, try different source
pip install rdkit-pypi
pip install py3Dmol

# Verify installation
python -c "from rdkit import Chem; import py3Dmol; print('Visualization ready')"
```

#### QM9 Dataset Loading Issues
```bash
# Clear DGL cache and reload
rm -rf ~/.dgl/
python -c "from dgl.data import QM9Dataset; dataset = QM9Dataset()"
```

#### Port Already in Use
```bash
# Kill existing Streamlit processes
pkill -f streamlit

# Use different port
streamlit run ui_drug_discovery.py --server.port 8502
```

### Performance Optimization

#### For AMD ROCm
```bash
# Set optimal ROCm environment
export HSA_OVERRIDE_GFX_VERSION=11.0.0  # Adjust for your GPU
export ROCM_PATH=/opt/rocm
export HIP_VISIBLE_DEVICES=0
```

#### For NVIDIA CUDA
```bash
# Set CUDA environment
export CUDA_VISIBLE_DEVICES=0
```

## 📊 Model Information

- **Architecture**: SE(3) Equivariant Transformer
- **Training Data**: QM9 dataset (134,000+ molecules)
- **Training Duration**: 100 epochs
- **Properties Predicted**: HOMO, LUMO, energy gap, dipole moment, polarizability
- **Performance**: State-of-the-art accuracy on QM9 benchmarks

## 🔗 Resources

- **SE(3) Transformers**: [Original Paper](https://arxiv.org/abs/2006.10503)
- **QM9 Dataset**: [Quantum Machine 9](https://quantum-machine.org/datasets/)
- **AMD ROCm**: [ROCm Documentation](https://docs.amd.com/)
- **DGL Library**: [Deep Graph Library](https://www.dgl.ai/)

## 🆘 Support

For issues and questions:

1. **Check the troubleshooting section above**
2. **Verify all dependencies are correctly installed**
3. **Ensure GPU drivers and ROCm/CUDA are properly configured**
4. **Check Streamlit logs for detailed error messages**

## 📝 License

This demo is provided for educational and research purposes. Please check individual component licenses:
- SE(3) Transformers: MIT License
- QM9 Dataset: [QM9 License](https://quantum-machine.org/datasets/)
- AMD ROCm: [ROCm License](https://github.com/RadeonOpenCompute/ROCm)

## 🎉 Quick Start Summary

```bash
# Complete setup in one go
source venv/bin/activate
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.6
pip install dgl streamlit numpy pandas plotly rdkit py3Dmol -f https://data.dgl.ai/wheels/repo.html
python app.py

# Then open: http://localhost:8501
```

---

**🔬 AMD ROCm Life Sciences: Accelerating Drug Discovery with GPU Computing**
