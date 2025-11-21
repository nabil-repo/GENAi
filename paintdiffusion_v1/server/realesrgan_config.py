"""
Configuration options for GANModel device selection
"""

# GANModel Device Configuration
GANModel_CONFIG = {
    # Device selection
    "device": "auto",  # "auto", "cuda", "cpu"
    "prefer_cpu_for_stability": True,  # Set to True to avoid CUDA RuntimeWarning
    
    # Model settings
    "model_name": "GANModel",  # or "GANModel_x4" 
    "scale_factor": 4,
    
    # Performance vs Stability trade-off notes:
    # CPU Mode:  ✅ No warnings, ⚠️ Slower processing
    # CUDA Mode: ⚠️ RuntimeWarning (harmless), ✅ Faster processing
}

def get_recommended_device():
    """Get recommended device based on stability preference"""
    import torch
    
    if GANModel_CONFIG["prefer_cpu_for_stability"]:
        return "cpu"
    else:
        return "cuda" if torch.cuda.is_available() else "cpu"

def print_device_info():
    """Print device selection information"""
    import torch
    
    recommended = get_recommended_device()
    cuda_available = torch.cuda.is_available()
    
    print("🔧 GANModel Device Configuration:")
    print(f"   CUDA Available: {cuda_available}")
    print(f"   Prefer CPU for Stability: {GANModel_CONFIG['prefer_cpu_for_stability']}")
    print(f"   Recommended Device: {recommended}")
    
    if recommended == "cpu" and cuda_available:
        print("💡 Using CPU to avoid CUDA RuntimeWarning")
        print("   (Change prefer_cpu_for_stability=False for faster GPU processing)")
    elif recommended == "cuda":
        print("⚠️  CUDA selected - RuntimeWarning may appear (harmless)")
    
    return recommended