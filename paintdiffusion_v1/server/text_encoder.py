import os
import torch
from transformers import CLIPTextModel, CLIPTokenizer

class TextEncoder:
    def __init__(self, model_name: str | None = None, device: str = "cuda"):
        self.model_name = model_name or os.getenv("TEXT_ENCODER", "openai/clip-vit-large-patch14")
        self.device = device if torch.cuda.is_available() and device == "cuda" else "cpu"
        self.tokenizer = CLIPTokenizer.from_pretrained(self.model_name)
        self.model = CLIPTextModel.from_pretrained(self.model_name).to(self.device)

    @torch.inference_mode()
    def encode(self, prompt: str) -> torch.Tensor:
        tokens = self.tokenizer([prompt], return_tensors="pt", padding=True, truncation=True)
        tokens = {k: v.to(self.device) for k, v in tokens.items()}
        out = self.model(**tokens)
        # Use last hidden state (sequence) for cross-attn
        return out.last_hidden_state  # [B, L, D]
