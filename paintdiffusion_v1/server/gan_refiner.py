# RealESRGAN based GAN refiner for high-quality image super-resolution
# Uses RealESRGAN package for professional image enhancement

import os
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch

class GANRefiner:
    def __init__(self, enabled: bool = False, scale_factor: int = 4, model_name: str = "GANModel", device: str = "cpu"):
        self.enabled = enabled
        self.scale_factor = scale_factor
        self.model_name = model_name
        
        # Force CPU mode for clean, warning-free operation
        self.device = "cpu"
        self.upsampler = None
        self.fallback_mode = False
        
        # print(f"GAN Refiner initializing (enabled: {enabled}, scale: {scale_factor}x, model: {model_name}, device: {self.device})")
        # print("INFO: Using CPU for clean, warning-free operation")
        
        if self.enabled:
            self._load_model()

    def _load_model(self):
        """Load the RealESRGAN model"""
        try:
            # Use RealESRGAN directly (RuntimeWarning is harmless)
            from RealESRGAN import RealESRGAN
            
            #print("Loading RealESRGAN model...")
            #print("NOTE: RuntimeWarning may appear but doesn't affect functionality")
            
            # Get the directory of this file and workspace root
            current_dir = os.path.dirname(os.path.abspath(__file__))
            workspace_root = os.path.dirname(current_dir)
            
            # Define model configurations with local paths
            model_configs = {
                "GANModel": {
                    "model_path": os.path.join(workspace_root, "GANModel.pth"),
                    "alternative_path": os.path.join(workspace_root, "RealESRGAN_x4.pth"),
                    "scale": 4
                },
                "RealESRGAN_x4": {
                    "model_path": os.path.join(workspace_root, "RealESRGAN_x4.pth"),
                    "alternative_path": os.path.join(workspace_root, "GANModel.pth"),
                    "scale": 4
                },
                "RealESRGAN_x2plus": {
                    "model_path": os.path.join(workspace_root, "RealESRGAN_x2plus.pth"),
                    "scale": 2
                }
            }
            
            config = model_configs.get(self.model_name, model_configs["GANModel"])
            
            # Initialize RealESRGAN model (original, working version)
            self.upsampler = RealESRGAN(self.device, scale=config["scale"])
            
            # Try to load model weights (check both primary and alternative paths)
            model_loaded = False
            
            # Try primary model path
            if os.path.exists(config["model_path"]):
                print(f"Using model file: {config['model_path']}")
                self.upsampler.load_weights(config["model_path"], download=False)
                model_loaded = True
            # Try alternative path if available
            elif "alternative_path" in config and os.path.exists(config["alternative_path"]):
                print(f"Using alternative model file: {config['alternative_path']}")
                self.upsampler.load_weights(config["alternative_path"], download=False)
                model_loaded = True
            
            if model_loaded:
                print("GAN model loaded successfully from local file!")
            else:
                print(f"WARNING: Model file not found: {config['model_path']}")
                if "alternative_path" in config:
                    print(f"WARNING: Alternative file not found: {config['alternative_path']}")
                print("Attempting to download model weights...")
                try:
                    # Let RealESRGAN download the model
                    self.upsampler.load_weights(self.model_name, download=True)
                    print("RealESRGAN model downloaded and loaded successfully!")
                except Exception as download_error:
                    print(f"ERROR: Failed to download model: {download_error}")
                    print("Falling back to enhanced PIL-based upscaling")
                    self.fallback_mode = True
                    return
            
        except ImportError as e:
            print(f"WARNING: RealESRGAN package not available: {e}")
            print("INFO: To install RealESRGAN:")
            print("   pip install RealESRGAN")
            print("Using enhanced PIL-based upscaling instead")
            self.fallback_mode = True
        except Exception as e:
            print(f"ERROR: Failed to load RealESRGAN model: {e}")
            print("Falling back to enhanced PIL-based upscaling")
            self.fallback_mode = True

    def refine(self, img: Image.Image, sharpness: float = 1.2, 
               color_boost: float = 1.1, contrast_boost: float = 1.05) -> Image.Image:
        """
        Apply RealESRGAN refinement to the image.
        Falls back to enhanced PIL operations if RealESRGAN is not available.
        """
        if not self.enabled:
            return img
        
        if self.fallback_mode or self.upsampler is None:
            return self._fallback_refine(img, sharpness, color_boost, contrast_boost)
        
        try:
            return self._realesrgan_refine(img)
        except Exception as e:
            print(f"WARNING: RealESRGAN refinement failed: {e}")
            print("Using fallback refinement")
            return self._fallback_refine(img, sharpness, color_boost, contrast_boost)

    def _realesrgan_refine(self, img: Image.Image) -> Image.Image:
        """Apply RealESRGAN-based refinement"""
        try:
            print("Applying RealESRGAN enhancement...")
            
            # Convert PIL to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Apply RealESRGAN enhancement
            enhanced_img = self.upsampler.predict(img)
            
            print(f"RealESRGAN enhancement complete: {img.size} -> {enhanced_img.size}")
            return enhanced_img
            
        except Exception as e:
            print(f"ERROR: RealESRGAN enhancement failed: {e}")
            raise

    def _fallback_refine(self, img: Image.Image, sharpness: float = 1.2, 
                        color_boost: float = 1.1, contrast_boost: float = 1.05) -> Image.Image:
        """
        Enhanced PIL-based fallback refinement with professional-grade processing.
        """
        try:
            print("Applying enhanced PIL-based refinement...")
            
            # Convert to RGB if needed
            if img.mode != 'RGB':
                img = img.convert('RGB')
            
            # Calculate target size
            original_size = img.size
            target_size = (original_size[0] * self.scale_factor, original_size[1] * self.scale_factor)
            
            # Multi-stage upscaling for better quality
            current_img = img
            stages = []
            
            if self.scale_factor == 4:
                stages = [2, 2]  # 2x then 2x
            elif self.scale_factor == 2:
                stages = [2]     # Single 2x
            else:
                stages = [self.scale_factor]  # Direct scaling
            
            for stage_scale in stages:
                stage_size = (current_img.size[0] * stage_scale, current_img.size[1] * stage_scale)
                
                # Use high-quality Lanczos resampling
                current_img = current_img.resize(stage_size, Image.LANCZOS)
                
                # Apply light sharpening after each stage
                if stage_scale > 1:
                    enhancer = ImageEnhance.Sharpness(current_img)
                    current_img = enhancer.enhance(1.1)
            
            # Final enhancements
            result_img = current_img
            
            # Sharpness enhancement
            if sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(result_img)
                result_img = enhancer.enhance(sharpness)
            
            # Color enhancement
            if color_boost != 1.0:
                enhancer = ImageEnhance.Color(result_img)
                result_img = enhancer.enhance(color_boost)
            
            # Contrast enhancement
            if contrast_boost != 1.0:
                enhancer = ImageEnhance.Contrast(result_img)
                result_img = enhancer.enhance(contrast_boost)
            
            # Final detail enhancement using unsharp mask
            try:
                # Convert to numpy for advanced processing
                img_array = np.array(result_img)
                
                # Apply subtle unsharp mask
                blurred = cv2.GaussianBlur(img_array, (0, 0), 0.8)
                unsharp_mask = cv2.addWeighted(img_array, 1.5, blurred, -0.5, 0)
                unsharp_mask = np.clip(unsharp_mask, 0, 255).astype(np.uint8)
                
                result_img = Image.fromarray(unsharp_mask)
                
            except Exception as e:
                print(f"⚠️  Advanced processing skipped: {e}")
                # Continue with basic enhancement
            
            print(f"✅ Enhanced PIL refinement complete: {original_size} -> {result_img.size}")
            return result_img
            
        except Exception as e:
            print(f"❌ Fallback refinement failed: {e}")
            # Return original image as last resort
            return img

    def get_model_info(self) -> dict:
        """Get information about the current model configuration"""
        return {
            "enabled": self.enabled,
            "model_name": self.model_name,
            "scale_factor": self.scale_factor,
            "fallback_mode": self.fallback_mode,
            "model_loaded": self.upsampler is not None
        }

    def is_available(self) -> bool:
        """Check if GAN refinement is available"""
        return self.enabled and (not self.fallback_mode or self.upsampler is not None)

    def estimate_processing_time(self, image_size: tuple) -> float:
        """Estimate processing time in seconds based on image size"""
        pixels = image_size[0] * image_size[1]
        
        if self.fallback_mode:
            # PIL-based processing is faster
            return max(0.5, pixels / 1000000)  # ~0.5-2 seconds for typical images
        else:
            # RealESRGAN processing time depends on device and model
            base_time = pixels / 500000  # Base estimate
            if self.device == "cpu":
                base_time *= 3  # CPU is slower
            return max(1.0, base_time)  # Minimum 1 second