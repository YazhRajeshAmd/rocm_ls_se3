import subprocess
import sys
import os

def main():
    """Launch the drug discovery demo"""
    
    print("🚀 Starting SE(3) Drug Discovery Demo...")
    print("📍 ROCm SE(3) Transformer Demo")
    print("🔗 Access at: http://localhost:8501")
    
    try:
        # Set environment variables for ROCm
        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = '0'  # or appropriate GPU
        
        # Launch Streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "ui_drug_discovery.py",
            "--server.port=8503",
            "--server.address=0.0.0.0"
        ], env=env)
        
    except KeyboardInterrupt:
        print("\n👋 Demo stopped by user")
    except Exception as e:
        print(f"❌ Error starting demo: {e}")

if __name__ == "__main__":
    main()
