"""
Setup and test script for GANModel integration
"""
import subprocess
import sys
import os

def install_GANModel():
    """Install GANModel package"""
    print("🔧 Installing GANModel...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "GANModel"])
        print("✅ GANModel installed successfully!")
        return True
    except Exception as e:
        print(f"❌ Failed to install GANModel: {e}")
        return False

def check_GANModel():
    """Check if GANModel is available"""
    try:
        import GANModel
        print("✅ GANModel is available")
        return True
    except ImportError:
        print("⚠️  GANModel not found")
        return False

def check_model_file():
    """Check if model file exists"""
    model_path = "GANModel.pth"
    if os.path.exists(model_path):
        size_mb = os.path.getsize(model_path) / (1024 * 1024)
        print(f"✅ Model file found: {model_path} ({size_mb:.1f}MB)")
        return True
    else:
        print(f"⚠️  Model file not found: {model_path}")
        print("💡 The model will be downloaded automatically when needed")
        return False

def main():
    print("🧪 GANModel Setup Check")
    print("=" * 30)
    
    # Check if GANModel is installed
    if not check_GANModel():
        print("\n🔧 Installing GANModel...")
        if not install_GANModel():
            print("❌ Setup failed")
            return
    
    # Check model file
    check_model_file()
    
    # Import and test
    try:
        from server.gan_refiner import GANRefiner
        print("\n✅ GAN refiner import successful")
        
        # Quick initialization test
        gan = GANRefiner(enabled=True, scale_factor=4, model_name="GANModel")
        info = gan.get_model_info()
        
        print(f"\n📋 GAN Refiner Status:")
        print(f"   Enabled: {info['enabled']}")
        print(f"   Model: {info['model_name']}")
        print(f"   Fallback Mode: {info['fallback_mode']}")
        print(f"   Model Loaded: {info['model_loaded']}")
        
        if not info['fallback_mode']:
            print("🎉 GANModel is ready for use!")
        else:
            print("⚠️  Using fallback mode (enhanced PIL upscaling)")
            
    except Exception as e:
        print(f"❌ GAN refiner test failed: {e}")

if __name__ == "__main__":
    main()