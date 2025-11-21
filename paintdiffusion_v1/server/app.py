import os
import io
import base64
from typing import Optional, List
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from PIL import Image
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))


from .ldm import LatentDiffusion
from .text_enhancer import PromptEnhancer
from .gan_refiner import GANRefiner
from .utils.image_io import b64_to_pil, pil_to_b64, ensure_rgb


# ---------- Models (load once) ----------
# DEVICE = "cuda" if os.getenv("DEVICE", "cuda") == "cuda" else "cpu"
print("Loading models...")


DEVICE = "cuda"
ldm = LatentDiffusion(device=DEVICE)


# Initialize text enhancer with error handling
try:
    text_enhancer = PromptEnhancer(device=DEVICE)
except Exception as e:
    print(f"Warning: Failed to initialize text enhancer: {e}")
    text_enhancer = None

# Initialize CPU-only GAN refiner
try:
    gan = GANRefiner(enabled=True, scale_factor=4, model_name="GANModel")
   # print("CPU-only GAN refiner initialized successfully!")
except Exception as e:
    print(f"Warning: Failed to initialize GAN refiner: {e}")
    gan = None

app = FastAPI(title="PaintDiffusion API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # or ["http://localhost:3000"] for security
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    prompt: str
    negative_prompt: Optional[str] = None
    guidance_scale: float = 7.0
    steps: int = Field(30, ge=1, le=150)
    height: int = Field(512, ge=128, le=1536)
    width: int = Field(512, ge=128, le=1536)
    seed: Optional[int] = None
    init_image_b64: Optional[str] = None
    strength: float = Field(0.7, ge=0.0, le=1.0)
    control_image_b64: Optional[str] = None
    mask_b64: Optional[str] = None
    sketch_image_b64: Optional[str] = Field(default=None, description="Base64 encoded sketch/drawing image")
    sketch_type: str = Field(default="canny", description="Sketch processing type: canny, scribble, lineart")
    controlnet_conditioning_scale: float = Field(default=1.0, ge=0.0, le=2.0, description="ControlNet conditioning strength")
    enhance_prompt: bool = Field(default=False, description="Whether to enhance the prompt using GPT-2")
    use_gan_refiner: bool = Field(default=False, description="Whether to apply GAN refining and show both results")
    selected_model: Optional[str] = Field(default=None, description="User-selected model name for display")

class GenerateResponse(BaseModel):
    image_b64: str
    refined_image_b64: Optional[str] = None
    used_gan_refiner: bool = False
    enhanced_prompt_used: Optional[str] = None
    used_sketch_processing: bool = False
    sketch_type_used: Optional[str] = None
    processed_control_image_b64: Optional[str] = None

class UpscaleRequest(BaseModel):
    image_b64: str

class UpscaleResponse(BaseModel):
    image_b64: str

class EnhancePromptRequest(BaseModel):
    prompt: str
    max_length: int = Field(100, ge=20, le=200, description="Maximum length of enhancement")
    temperature: float = Field(0.8, ge=0.1, le=2.0, description="Sampling temperature")

class EnhancePromptResponse(BaseModel):
    original_prompt: str
    enhanced_prompt: str

class PromptVariationsRequest(BaseModel):
    prompt: str
    num_variations: int = Field(3, ge=1, le=5, description="Number of variations to generate")

class PromptVariationsResponse(BaseModel):
    original_prompt: str
    variations: List[str]

class GanRefineRequest(BaseModel):
    image_b64: str
    sharpness: float = Field(1.2, ge=0.5, le=2.0, description="Sharpness enhancement factor")
    color_boost: float = Field(1.1, ge=0.8, le=1.5, description="Color enhancement factor")
    contrast_boost: float = Field(1.05, ge=0.8, le=1.3, description="Contrast enhancement factor")

class GanRefineResponse(BaseModel):
    original_image_b64: str
    refined_image_b64: str

class SketchProcessRequest(BaseModel):
    sketch_image_b64: str
    sketch_type: str = Field(default="canny", description="Sketch processing type: canny, scribble, lineart")
    enhance_sketch: bool = Field(default=True, description="Whether to enhance the sketch before processing")

class SketchProcessResponse(BaseModel):
    original_sketch_b64: str
    processed_control_b64: str
    sketch_type_used: str

class VAEEncodeRequest(BaseModel):
    image_b64: str

class VAEEncodeResponse(BaseModel):
    latents_info: dict
    success: bool

class PipelineInfoResponse(BaseModel):
    pipeline_info: dict

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    # Print selected model to console
    if req.selected_model:
        print(f"Using Model: {req.selected_model}")
    
    # Enhance prompt if requested
    prompt = req.prompt
    if req.enhance_prompt and text_enhancer is not None:
        try:
            prompt = text_enhancer.enhance_prompt(req.prompt)
        except Exception as e:
            print(f"Warning: Prompt enhancement failed: {e}")
            prompt = req.prompt  # Fall back to original prompt
    elif req.enhance_prompt and text_enhancer is None:
        print("Warning: Text enhancer not available, using original prompt")
    
    init_image = b64_to_pil(req.init_image_b64) if req.init_image_b64 else None
    control_image = b64_to_pil(req.control_image_b64) if req.control_image_b64 else None
    mask_image = b64_to_pil(req.mask_b64) if req.mask_b64 else None
    sketch_image = b64_to_pil(req.sketch_image_b64) if req.sketch_image_b64 else None

    # Ensure proper color spaces
    if init_image is not None:
        init_image = ensure_rgb(init_image)
    if control_image is not None:
        control_image = ensure_rgb(control_image)
    if mask_image is not None:
        mask_image = mask_image.convert("L")  # single-channel mask
    if sketch_image is not None:
        sketch_image = ensure_rgb(sketch_image)

    # Generate image with all parameters
    img = ldm.generate(
        prompt=prompt,  # Use the potentially enhanced prompt
        negative_prompt=req.negative_prompt,
        guidance_scale=req.guidance_scale,
        num_inference_steps=req.steps,
        height=req.height,
        width=req.width,
        seed=req.seed,
        init_image=init_image,
        strength=req.strength,
        control_image=control_image,
        mask_image=mask_image,
        sketch_image=sketch_image,
        sketch_type=req.sketch_type,
        controlnet_conditioning_scale=req.controlnet_conditioning_scale,
    )

    # Store original image and processing info
    original_img_b64 = pil_to_b64(img)
    refined_img_b64 = None
    used_gan_refiner = False
    used_sketch_processing = sketch_image is not None
    processed_control_image_b64 = None
    
    # If sketch was used, also return the processed control image
    if sketch_image is not None and hasattr(ldm, 'process_sketch'):
        try:
            processed_control = ldm.process_sketch(sketch_image, req.sketch_type)
            processed_control_image_b64 = pil_to_b64(processed_control)
        except Exception as e:
            print(f"Warning: Failed to generate processed control image: {e}")
    
    gan.refine_enabled = req.use_gan_refiner
    if req.use_gan_refiner and gan is not None and gan.is_available():
        try:
            refined_img = gan.refine(img)
            if refined_img is not None:
                refined_img_b64 = pil_to_b64(refined_img)
                used_gan_refiner = True
            else:
                print("Warning: GAN refiner returned None, skipping refinement")
        except Exception as e:
            print(f"Warning: GAN refinement failed: {e}")
    elif req.use_gan_refiner and gan is None:
        print("Warning: GAN refiner not available, skipping refinement")
   

    return GenerateResponse(
        image_b64=original_img_b64,
        refined_image_b64=refined_img_b64,
        used_gan_refiner=used_gan_refiner,
        enhanced_prompt_used=prompt if req.enhance_prompt else None,
        used_sketch_processing=used_sketch_processing,
        sketch_type_used=req.sketch_type if used_sketch_processing else None,
        processed_control_image_b64=processed_control_image_b64
    )

@app.post("/enhance-prompt", response_model=EnhancePromptResponse)
def enhance_prompt_endpoint(req: EnhancePromptRequest):
    """
    Enhance a prompt using GPT-2 for better image generation results.
    """
    if text_enhancer is None:
        return EnhancePromptResponse(
            original_prompt=req.prompt,
            enhanced_prompt=f"{req.prompt}, highly detailed, masterpiece quality"  # Basic enhancement
        )
    
    try:
        enhanced_prompt = text_enhancer.enhance_prompt(
            prompt=req.prompt,
            max_length=req.max_length,
            temperature=req.temperature
        )
        return EnhancePromptResponse(
            original_prompt=req.prompt,
            enhanced_prompt=enhanced_prompt
        )
    except Exception as e:
        print(f"Error enhancing prompt: {e}")
        return EnhancePromptResponse(
            original_prompt=req.prompt,
            enhanced_prompt=req.prompt  # Return original if enhancement fails
        )

@app.post("/prompt-variations", response_model=PromptVariationsResponse)
def prompt_variations_endpoint(req: PromptVariationsRequest):
    """
    Generate multiple artistic variations of a prompt.
    """
    if text_enhancer is None:
        # Basic variations when GPT-2 is not available
        basic_variations = [
            f"{req.prompt}, highly detailed, masterpiece quality",
            f"{req.prompt}, vivid colors, beautiful lighting",
            f"{req.prompt}, professional artwork, stunning composition"
        ]
        return PromptVariationsResponse(
            original_prompt=req.prompt,
            variations=basic_variations[:req.num_variations]
        )
    
    try:
        variations = text_enhancer.generate_artistic_variations(
            prompt=req.prompt,
            num_variations=req.num_variations
        )
        return PromptVariationsResponse(
            original_prompt=req.prompt,
            variations=variations
        )
    except Exception as e:
        print(f"Error generating prompt variations: {e}")
        return PromptVariationsResponse(
            original_prompt=req.prompt,
            variations=[req.prompt]  # Return original if generation fails
        )

@app.get("/gan-info")
def gan_info():
    """
    Get information about the GAN refiner status and capabilities.
    """
    if gan is None:
        return {
            "available": False,
            "reason": "GAN refiner not initialized",
            "model": None,
            "scale_factor": None,
            "device": None
        }
    
    return {
        "available": gan.is_available(),
        "model": gan.model_name if hasattr(gan, 'model_name') else "GANModel",
        "scale_factor": gan.scale_factor if hasattr(gan, 'scale_factor') else 4,
        "device": "cpu",
        "description": "CPU-only GANModel for clean, warning-free 4x super-resolution"
    }

@app.post("/upscale", response_model=UpscaleResponse)
def upscale_image(req: UpscaleRequest):
    """
    Upscale an image using GAN refiner.
    """
    if gan is None or not gan.is_available():
        # Fallback to simple PIL upscaling
        img = b64_to_pil(req.image_b64)
        upscaled = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        return UpscaleResponse(image_b64=pil_to_b64(upscaled))
    
    try:
        img = b64_to_pil(req.image_b64)
        refined_img = gan.refine(img)
        if refined_img is not None:
            return UpscaleResponse(image_b64=pil_to_b64(refined_img))
        else:
            # Fallback if refinement fails
            upscaled = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
            return UpscaleResponse(image_b64=pil_to_b64(upscaled))
    except Exception as e:
        print(f"Error during upscaling: {e}")
        # Fallback to simple upscaling
        img = b64_to_pil(req.image_b64)
        upscaled = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        return UpscaleResponse(image_b64=pil_to_b64(upscaled))

@app.post("/gan-refine", response_model=GanRefineResponse)
def gan_refine_image(req: GanRefineRequest):
    """
    Apply GAN refinement to an image with additional enhancement parameters.
    """
    img = b64_to_pil(req.image_b64)
    original_b64 = req.image_b64
    
    if gan is None or not gan.is_available():
        # Apply basic PIL enhancements as fallback
        from PIL import ImageEnhance
        enhanced_img = img
        
        # Apply sharpness
        if req.sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced_img)
            enhanced_img = enhancer.enhance(req.sharpness)
        
        # Apply color boost
        if req.color_boost != 1.0:
            enhancer = ImageEnhance.Color(enhanced_img)
            enhanced_img = enhancer.enhance(req.color_boost)
        
        # Apply contrast boost
        if req.contrast_boost != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced_img)
            enhanced_img = enhancer.enhance(req.contrast_boost)
        
        # Simple 2x upscale
        enhanced_img = enhanced_img.resize((enhanced_img.width * 2, enhanced_img.height * 2), Image.LANCZOS)
        
        return GanRefineResponse(
            original_image_b64=original_b64,
            refined_image_b64=pil_to_b64(enhanced_img)
        )
    
    try:
        # Use GAN refiner for upscaling
        refined_img = gan.refine(img)
        
        if refined_img is not None:
            # Apply additional enhancements to the refined image
            from PIL import ImageEnhance
            enhanced_img = refined_img
            
            # Apply sharpness
            if req.sharpness != 1.0:
                enhancer = ImageEnhance.Sharpness(enhanced_img)
                enhanced_img = enhancer.enhance(req.sharpness)
            
            # Apply color boost
            if req.color_boost != 1.0:
                enhancer = ImageEnhance.Color(enhanced_img)
                enhanced_img = enhancer.enhance(req.color_boost)
            
            # Apply contrast boost
            if req.contrast_boost != 1.0:
                enhancer = ImageEnhance.Contrast(enhanced_img)
                enhanced_img = enhancer.enhance(req.contrast_boost)
            
            return GanRefineResponse(
                original_image_b64=original_b64,
                refined_image_b64=pil_to_b64(enhanced_img)
            )
        else:
            raise Exception("GAN refiner returned None")
    
    except Exception as e:
        print(f"Error during GAN refinement: {e}")
        # Fallback to basic enhancement
        from PIL import ImageEnhance
        enhanced_img = img
        
        # Apply enhancements
        if req.sharpness != 1.0:
            enhancer = ImageEnhance.Sharpness(enhanced_img)
            enhanced_img = enhancer.enhance(req.sharpness)
        
        if req.color_boost != 1.0:
            enhancer = ImageEnhance.Color(enhanced_img)
            enhanced_img = enhancer.enhance(req.color_boost)
        
        if req.contrast_boost != 1.0:
            enhancer = ImageEnhance.Contrast(enhanced_img)
            enhanced_img = enhancer.enhance(req.contrast_boost)
        
        # Simple 2x upscale as fallback
        enhanced_img = enhanced_img.resize((enhanced_img.width * 2, enhanced_img.height * 2), Image.LANCZOS)
        
        return GanRefineResponse(
            original_image_b64=original_b64,
            refined_image_b64=pil_to_b64(enhanced_img)
        )

@app.post("/process-sketch", response_model=SketchProcessResponse)
def process_sketch_endpoint(req: SketchProcessRequest):
    """
    Process a sketch image for ControlNet input.
    """
    sketch_image = b64_to_pil(req.sketch_image_b64)
    
    # Enhance sketch if requested
    if req.enhance_sketch and hasattr(ldm, 'enhance_sketch'):
        try:
            sketch_image = ldm.enhance_sketch(sketch_image)
        except Exception as e:
            print(f"Warning: Sketch enhancement failed: {e}")
    
    # Process sketch for ControlNet
    try:
        processed_control = ldm.process_sketch(sketch_image, req.sketch_type)
        
        return SketchProcessResponse(
            original_sketch_b64=req.sketch_image_b64,
            processed_control_b64=pil_to_b64(processed_control),
            sketch_type_used=req.sketch_type
        )
    except Exception as e:
        print(f"Error processing sketch: {e}")
        # Return original as fallback
        return SketchProcessResponse(
            original_sketch_b64=req.sketch_image_b64,
            processed_control_b64=req.sketch_image_b64,
            sketch_type_used=req.sketch_type
        )

@app.post("/vae-encode", response_model=VAEEncodeResponse)
def vae_encode_endpoint(req: VAEEncodeRequest):
    """
    Encode an image using the VAE encoder.
    """
    try:
        image = b64_to_pil(req.image_b64)
        latents = ldm.encode_image_with_vae(image)
        
        return VAEEncodeResponse(
            latents_info={
                "shape": list(latents.shape),
                "dtype": str(latents.dtype),
                "device": str(latents.device),
                "mean": float(latents.mean().item()),
                "std": float(latents.std().item()),
                "min": float(latents.min().item()),
                "max": float(latents.max().item()),
            },
            success=True
        )
    except Exception as e:
        print(f"Error encoding image with VAE: {e}")
        return VAEEncodeResponse(
            latents_info={"error": str(e)},
            success=False
        )

@app.get("/pipeline-info", response_model=PipelineInfoResponse)
def pipeline_info_endpoint():
    """
    Get information about the loaded pipeline and capabilities.
    """
    try:
        pipeline_info = ldm.get_pipeline_info()
        return PipelineInfoResponse(pipeline_info=pipeline_info)
    except Exception as e:
        print(f"Error getting pipeline info: {e}")
        return PipelineInfoResponse(
            pipeline_info={
                "error": str(e),
                "basic_info": {
                    "model": getattr(ldm, 'sd_model', 'unknown'),
                    "device": getattr(ldm, 'device', 'unknown'),
                }
            }
        )

@app.get("/sketch-types")
def get_sketch_types():
    """
    Get available sketch processing types.
    """
    return {
        "sketch_types": [
            {
                "name": "canny",
                "description": "Canny edge detection - extracts clean edges from sketches",
                "recommended_for": ["line drawings", "architectural sketches", "technical drawings"]
            },
            {
                "name": "scribble",
                "description": "Scribble processing - handles rough sketches and doodles",
                "recommended_for": ["rough sketches", "concept art", "quick doodles"]
            },
            {
                "name": "lineart",
                "description": "Line art processing - cleans and enhances line art",
                "recommended_for": ["clean line art", "manga/anime style", "illustrations"]
            }
        ]
    }
