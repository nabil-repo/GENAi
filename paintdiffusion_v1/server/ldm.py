import os
import torch
import numpy as np
from typing import Optional
from PIL import Image, ImageOps, ImageFilter
import cv2
from diffusers import (
    StableDiffusionPipeline,
    StableDiffusionControlNetPipeline,
    StableDiffusionImg2ImgPipeline,
    ControlNetModel,
    DDIMScheduler,
    DPMSolverMultistepScheduler,
    AutoencoderKL,
)

class LatentDiffusion:
    def __init__(self, sd_model: Optional[str] = None, controlnet_model: Optional[str] = None, vae_model: Optional[str] = None, device: str = "cuda"):
        # ---- device + dtype (force FP32 on CUDA to avoid black images on 16-series GPUs) ----
        use_cuda = (device == "cuda") and torch.cuda.is_available()
        self.device = "cuda" if use_cuda else "cpu"
    # Conservative memory mode for 4GB GPUs unless explicitly disabled
        self.low_vram = os.getenv("LOW_VRAM", "1").lower() not in ("0", "false", "no")
        self.dtype = torch.float32  # CRITICAL: do NOT use float16 on GTX 1650

        print("=== Torch / CUDA ===")
        print("torch:", torch.__version__)
        print("cuda version:", torch.version.cuda)
        print("cuda available:", torch.cuda.is_available())
        if use_cuda:
            print("gpu:", torch.cuda.get_device_name(0))
            # modest perf boost, safe on FP32:
            torch.backends.cuda.matmul.allow_tf32 = True

        self.sd_model = sd_model or os.getenv("SD_MODEL", "runwayml/stable-diffusion-v1-5")
        self.controlnet_model = controlnet_model or os.getenv("CONTROLNET", "lllyasviel/sd-controlnet-canny")
        self.vae_model = vae_model or os.getenv("VAE_MODEL", None)  # Optional better VAE

        # ---- Load VAE if specified ----
        vae = None
        if self.vae_model:
            print(f"Loading custom VAE: {self.vae_model}")
            try:
                vae = AutoencoderKL.from_pretrained(
                    self.vae_model,
                    torch_dtype=self.dtype,
                )
                print("Custom VAE loaded successfully!")
            except Exception as e:
                print(f"WARNING: Failed to load custom VAE: {e}")
                vae = None

        # ---- Load ControlNet for sketch processing ----
        controlnet = None
        if self.controlnet_model:
            # print(f"Loading ControlNet: {self.controlnet_model}")
            try:
                controlnet = ControlNetModel.from_pretrained(
                    self.controlnet_model,
                    torch_dtype=self.dtype,
                )
                print("ControlNet loaded successfully!")
            except Exception as e:
                print(f"WARNING: Failed to load ControlNet: {e}")
                controlnet = None

        # ---- load pipeline in FP32 ----
        # Build common kwargs and ONLY pass 'vae' if we actually loaded a custom VAE.
        common_kwargs = {
            "torch_dtype": self.dtype,
            "safety_checker": None,
        }
        if controlnet is not None:
            pipe_kwargs = {**common_kwargs, "controlnet": controlnet}
            if vae is not None:
                pipe_kwargs["vae"] = vae
            self.pipe = StableDiffusionControlNetPipeline.from_pretrained(
                self.sd_model,
                **pipe_kwargs,
            )
            print("ControlNet pipeline loaded for sketch-to-image")
        else:
            pipe_kwargs = {**common_kwargs}
            if vae is not None:
                pipe_kwargs["vae"] = vae
            self.pipe = StableDiffusionPipeline.from_pretrained(
                self.sd_model,
                **pipe_kwargs,
            )
            print("Standard pipeline loaded")
        
        # ---- Also create img2img pipeline ----
        img2img_kwargs = {
            "torch_dtype": self.dtype,
            "safety_checker": None,
        }
        if vae is not None:
            img2img_kwargs["vae"] = vae
        self.img2img_pipe = StableDiffusionImg2ImgPipeline.from_pretrained(
            self.sd_model,
            **img2img_kwargs,
        )
        #print("Img2Img pipeline loaded")

        # ---- scheduler ----
        try:
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(self.pipe.scheduler.config)
        except Exception:
            self.pipe.scheduler = DDIMScheduler.from_config(self.pipe.scheduler.config)

        # ---- memory helpers (important for 4 GB VRAM) ----
        # keeps attention blocks streamed to save memory
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing()
        # enable xformers attention if available (optional)
        if hasattr(self.pipe, "enable_xformers_memory_efficient_attention"):
            try:
                self.pipe.enable_xformers_memory_efficient_attention()
            except Exception:
                pass
        # process VAE in tiles (prevents VAE OOM/blank output)
        if getattr(self.pipe, "vae", None) is not None and hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
        # Move/offload models
        if self.device == "cuda" and self.low_vram and hasattr(self.pipe, "enable_model_cpu_offload"):
            try:
                self.pipe.enable_model_cpu_offload()
            except Exception:
                # fallback (slower)
                if hasattr(self.pipe, "enable_sequential_cpu_offload"):
                    self.pipe.enable_sequential_cpu_offload()
        else:
            self.pipe = self.pipe.to(self.device)
        # Match img2img placement/offload
        if self.device == "cuda" and self.low_vram and hasattr(self.img2img_pipe, "enable_model_cpu_offload"):
            try:
                self.img2img_pipe.enable_model_cpu_offload()
            except Exception:
                if hasattr(self.img2img_pipe, "enable_sequential_cpu_offload"):
                    self.img2img_pipe.enable_sequential_cpu_offload()
        else:
            self.img2img_pipe = self.img2img_pipe.to(self.device)
        
    def process_sketch(self, sketch_image: Image.Image, sketch_type: str = "canny") -> Image.Image:
        """
        Process a sketch image for ControlNet input.
        
        Args:
            sketch_image: Input sketch/drawing image
            sketch_type: Type of processing ("canny", "scribble", "lineart")
        
        Returns:
            Processed control image
        """
        # Convert to numpy array
        image_array = np.array(sketch_image)
        
        if sketch_type == "canny":
            # Apply Canny edge detection
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Apply Canny edge detection
            edges = cv2.Canny(gray, 50, 200)
            
            # Convert back to PIL Image
            control_image = Image.fromarray(edges).convert("RGB")
            
        elif sketch_type == "scribble":
            # Process as scribble/sketch
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Invert if needed (white background -> black background)
            if np.mean(gray) > 127:
                gray = 255 - gray
            
            # Convert back to PIL Image
            control_image = Image.fromarray(gray).convert("RGB")
            
        elif sketch_type == "lineart":
            # Clean line art processing
            if len(image_array.shape) == 3:
                gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
            else:
                gray = image_array
            
            # Apply bilateral filter to smooth while preserving edges
            filtered = cv2.bilateralFilter(gray, 9, 75, 75)
            
            # Convert back to PIL Image
            control_image = Image.fromarray(filtered).convert("RGB")
            
        else:
            # Default: just convert to RGB
            control_image = sketch_image.convert("RGB")
        
        return control_image
    
    def enhance_sketch(self, sketch_image: Image.Image) -> Image.Image:
        """
        Enhance a sketch image for better ControlNet processing.
        """
        # Apply some enhancement filters
        enhanced = sketch_image.convert("L")  # Convert to grayscale
        
        # Sharpen the sketch
        enhanced = enhanced.filter(ImageFilter.SHARPEN)
        
        # Increase contrast
        enhanced = ImageOps.autocontrast(enhanced, cutoff=2)
        
        # Convert back to RGB
        return enhanced.convert("RGB")

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        guidance_scale: float = 7.0,
        num_inference_steps: int = 30,
        height: int = 512,
        width: int = 512,
        seed: Optional[int] = None,
        init_image: Optional[Image.Image] = None,
        strength: float = 0.7,
        control_image: Optional[Image.Image] = None,
        mask_image: Optional[Image.Image] = None,
        sketch_image: Optional[Image.Image] = None,
        sketch_type: str = "canny",
        controlnet_conditioning_scale: float = 1.0,
    ) -> Image.Image:
        # Free cached blocks before heavy op
        if self.device == "cuda":
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass

        # In low VRAM mode, cap resolution/steps a bit for stability
        if self.low_vram:
            height = min(height, 448)
            width = min(width, 448)
            num_inference_steps = min(num_inference_steps, 25)

        # enforce multiples of 8
        height = int(round(height / 8) * 8)
        width = int(round(width / 8) * 8)
        
        generator = torch.Generator(device=self.device)
        if seed is not None:
            generator = generator.manual_seed(seed)
            print(f"Using seed {seed}")

        kwargs = dict(
            prompt=prompt,
            negative_prompt=negative_prompt,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            height=height,
            width=width,
            generator=generator,
        )

        # Helper to execute a pipeline call with an OOM fallback
        def _run_with_retry(callable_fn, first_kwargs):
            try:
                return callable_fn(**first_kwargs)
            except (torch.cuda.OutOfMemoryError, RuntimeError) as e:
                msg = str(e)
                if "out of memory" not in msg.lower():
                    raise
                # Retry at lower cost
                if self.device == "cuda":
                    try:
                        torch.cuda.empty_cache()
                    except Exception:
                        pass
                retry_kwargs = dict(first_kwargs)
                retry_kwargs["height"] = max(256, min(384, first_kwargs.get("height", height)))
                retry_kwargs["width"] = max(256, min(384, first_kwargs.get("width", width)))
                retry_kwargs["num_inference_steps"] = min(first_kwargs.get("num_inference_steps", num_inference_steps), 20)
                return callable_fn(**retry_kwargs)

        # Handle sketch input with ControlNet
        if sketch_image is not None and hasattr(self.pipe, "controlnet"):
            print(f"Processing sketch with {sketch_type} method")
            control_image = self.process_sketch(sketch_image, sketch_type)
            kwargs["image"] = control_image
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            # Use ControlNet pipeline for sketch-to-image with OOM retry
            out = _run_with_retry(self.pipe, kwargs)
            
        # ControlNet input (expects an image) - legacy support
        elif control_image is not None and hasattr(self.pipe, "controlnet"):
            kwargs["image"] = control_image
            kwargs["controlnet_conditioning_scale"] = controlnet_conditioning_scale
            out = _run_with_retry(self.pipe, kwargs)

        # Img2Img processing
        elif init_image is not None:
            # Use dedicated img2img pipeline for better results
            img2img_kwargs = {
                "prompt": prompt,
                "negative_prompt": negative_prompt,
                "image": init_image,
                "strength": strength,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "generator": generator,
            }
            out = _run_with_retry(self.img2img_pipe, img2img_kwargs)
            
        else:
            # Standard txt2img
            out = _run_with_retry(self.pipe, kwargs)

        img = out.images[0]
        return img
    
    def encode_image_with_vae(self, image: Image.Image) -> torch.Tensor:
       
        # Convert PIL to tensor
        image_array = np.array(image)
        image_tensor = torch.from_numpy(image_array).float() / 255.0
        image_tensor = image_tensor.permute(2, 0, 1).unsqueeze(0)  # [1, C, H, W]
        image_tensor = image_tensor.to(self.device)
        
        # Normalize to [-1, 1]
        image_tensor = 2.0 * image_tensor - 1.0
        
        # Encode with VAE
        with torch.no_grad():
            latents = self.pipe.vae.encode(image_tensor).latent_dist.sample()
            latents = latents * self.pipe.vae.config.scaling_factor
        
        return latents
    
    def decode_latents_with_vae(self, latents: torch.Tensor) -> Image.Image:
        """
        Decode latents using the VAE decoder.
        """
        # Scale latents
        latents = latents / self.pipe.vae.config.scaling_factor
        
        # Decode with VAE
        with torch.no_grad():
            image_tensor = self.pipe.vae.decode(latents).sample
        
        # Convert to PIL
        image_tensor = (image_tensor / 2 + 0.5).clamp(0, 1)
        image_tensor = image_tensor.cpu().permute(0, 2, 3, 1).numpy()
        images = (image_tensor * 255).round().astype("uint8")
        
        return Image.fromarray(images[0])
    
    def get_pipeline_info(self) -> dict:
        """
        Get information about the loaded pipeline and capabilities.
        """
        return {
            "model": self.sd_model,
            "device": self.device,
            "dtype": str(self.dtype),
            "has_controlnet": hasattr(self.pipe, "controlnet"),
            "has_custom_vae": self.vae_model is not None,
            "controlnet_model": self.controlnet_model if hasattr(self.pipe, "controlnet") else None,
            "vae_model": self.vae_model,
            "capabilities": {
                "txt2img": True,
                "img2img": True,
                "sketch2img": hasattr(self.pipe, "controlnet"),
                "controlnet": hasattr(self.pipe, "controlnet"),
                "vae_encoding": True,
            }
        }
