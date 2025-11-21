"""
Text enhancement module using GPT-2 for prompt improvement.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from typing import Optional
import logging

logger = logging.getLogger(__name__)

class PromptEnhancer:
    """
    A class to enhance prompts using GPT-2 model for better image generation results.
    """
    
    def __init__(self, device: str = "cuda"):
        """
        Initialize the prompt enhancer with GPT-2 model.
        
        Args:
            device: Device to run the model on ('cuda' or 'cpu')
        """
        self.device = device if torch.cuda.is_available() else "cpu"
        # print(f"Loading GPT-2 model on {self.device}...")
        
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("gpt2")
            self.model = AutoModelForCausalLM.from_pretrained("gpt2")
            self.model.to(self.device)
            self.model.eval()  # Set to evaluation mode
            
            # Set pad_token if it doesn't exist
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
                
            # print("GPT-2 model loaded successfully for prompt enhancement!")
        except Exception as e:
            # print(f"ERROR: Failed to load GPT-2 model: {e}")
            # print("INFO: Falling back to basic prompt enhancement...")
            self.model = None
            self.tokenizer = None
    
    def enhance_prompt(self, 
                      prompt: str, 
                      max_length: int = 100,
                      num_return_sequences: int = 1,
                      temperature: float = 0.8,
                      top_p: float = 0.9,
                      do_sample: bool = True) -> str:
        """
        Enhance the given prompt using GPT-2 text generation.
        
        Args:
            prompt: The original prompt to enhance
            max_length: Maximum length of the generated text
            num_return_sequences: Number of sequences to generate
            temperature: Sampling temperature (higher = more random)
            top_p: Nucleus sampling parameter
            do_sample: Whether to use sampling or greedy decoding
            
        Returns:
            Enhanced prompt string
        """
        # Fallback if model is not loaded
        if self.model is None or self.tokenizer is None:
            return self._basic_enhance_prompt(prompt)
            
        try:
            # Prepare input for art/image generation context
            enhanced_prompt_prefix = f"Create a detailed, vivid description for an artwork: {prompt}. The image should be"
            
            inputs = self.tokenizer.encode(enhanced_prompt_prefix, return_tensors="pt").to(self.device)
            
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs,
                    max_length=len(inputs[0]) + max_length,
                    num_return_sequences=num_return_sequences,
                    temperature=temperature,
                    top_p=top_p,
                    do_sample=do_sample,
                    pad_token_id=self.tokenizer.eos_token_id,
                    repetition_penalty=1.1,
                    length_penalty=1.0
                )
            
            # Decode and clean the output
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # Extract the enhanced part (remove the prefix)
            enhanced_part = generated_text[len(enhanced_prompt_prefix):].strip()
            
            # Combine original prompt with enhancement
            if enhanced_part:
                enhanced_prompt = f"{prompt}, {enhanced_part}"
            else:
                enhanced_prompt = prompt
                
            # Clean up the prompt (remove incomplete sentences, etc.)
            enhanced_prompt = self._clean_prompt(enhanced_prompt)
            
            logger.info(f"Original prompt: {prompt}")
            logger.info(f"Enhanced prompt: {enhanced_prompt}")
            
            return enhanced_prompt
            
        except Exception as e:
            logger.error(f"Error enhancing prompt: {e}")
            return prompt  # Return original prompt if enhancement fails
    
    def _clean_prompt(self, prompt: str) -> str:
        """
        Clean the enhanced prompt by removing incomplete sentences and redundant text.
        
        Args:
            prompt: The prompt to clean
            
        Returns:
            Cleaned prompt string
        """
        # Remove incomplete sentences (those not ending with proper punctuation)
        sentences = prompt.split('.')
        complete_sentences = []
        
        for sentence in sentences[:-1]:  # Exclude the last potentially incomplete sentence
            sentence = sentence.strip()
            if sentence and len(sentence) > 3:
                complete_sentences.append(sentence)
        
        # Check if the last sentence is complete
        last_sentence = sentences[-1].strip()
        if last_sentence and (last_sentence.endswith('.') or last_sentence.endswith('!') or last_sentence.endswith('?')):
            complete_sentences.append(last_sentence.rstrip('.!?'))
        
        cleaned_prompt = '. '.join(complete_sentences)
        
        # Limit length to prevent overly long prompts
        if len(cleaned_prompt) > 200:
            cleaned_prompt = cleaned_prompt[:200].rsplit(' ', 1)[0] + "..."
        
        return cleaned_prompt.strip()
    
    def generate_artistic_variations(self, prompt: str, num_variations: int = 3) -> list:
        """
        Generate multiple artistic variations of the prompt.
        
        Args:
            prompt: The original prompt
            num_variations: Number of variations to generate
            
        Returns:
            List of enhanced prompt variations
        """
        variations = []
        artistic_styles = [
            "in the style of a Renaissance painting",
            "as a modern digital art piece",
            "in a photorealistic style",
            "as an abstract artistic interpretation",
            "in the style of impressionist art"
        ]
        
        for i in range(num_variations):
            style = artistic_styles[i % len(artistic_styles)]
            enhanced_prompt = f"{prompt}, {style}"
            variations.append(self.enhance_prompt(enhanced_prompt, max_length=50))
        
        return variations
    
    def _basic_enhance_prompt(self, prompt: str) -> str:
        """
        Basic prompt enhancement when GPT-2 model is not available.
        
        Args:
            prompt: The original prompt
            
        Returns:
            Enhanced prompt with basic artistic descriptors
        """
        artistic_enhancers = [
            "highly detailed",
            "masterpiece quality",
            "vivid colors",
            "beautiful lighting",
            "professional artwork",
            "stunning visual",
            "artistic composition"
        ]
        
        # Add some basic enhancements
        import random
        enhancer = random.choice(artistic_enhancers)
        enhanced = f"{prompt}, {enhancer}"
        
        print(f"Using basic enhancement: {prompt} -> {enhanced}")
        return enhanced