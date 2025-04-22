"""
Model interface helper module for the E-Soul project.
Provides integration with Hugging Face Transformers models.
"""

import asyncio
import json
import logging
import os
import time
from pathlib import Path
from typing import Dict, List, Optional, Any, Union, Tuple

import torch
from tqdm import tqdm

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("e_soul.model_helper")

# Check if transformers is available
try:
    import transformers
    from transformers import AutoModelForCausalLM, AutoTokenizer
    HAS_TRANSFORMERS = True
except ImportError:
    logger.warning("Transformers not available. Model functionality will be limited.")
    HAS_TRANSFORMERS = False

class ModelInterfaceHelper:
    """Helper class for interfacing with Transformers models."""
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 model_id: Optional[str] = None,
                 device: Optional[str] = None,
                 max_tokens: int = 1024,
                 temperature: float = 0.7):
        """Initialize the model interface helper.
        
        Args:
            model_path: Local path to the model
            model_id: Hugging Face model ID (used if model_path not provided)
            device: Device to use for inference ('cuda', 'cpu', etc.)
            max_tokens: Maximum number of tokens to generate
            temperature: Temperature for generation
        """
        if not HAS_TRANSFORMERS:
            raise ImportError("Transformers library is required for ModelInterfaceHelper")
        
        self.model_path = model_path
        self.model_id = model_id or "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # Determine device
        if device:
            self.device = device
        else:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        # Initialize to None, will be loaded on demand
        self.model = None
        self.tokenizer = None
        self.is_loaded = False
        
        logger.info(f"Model interface helper initialized (device: {self.device})")
    
    async def load_model(self) -> bool:
        """Load the model and tokenizer.
        
        Returns:
            Success status
        """
        if self.is_loaded:
            return True
            
        try:
            # Run in a separate thread to avoid blocking
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._load_model_sync)
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            return False
    
    def _load_model_sync(self) -> bool:
        """Synchronous method to load the model (runs in executor).
        
        Returns:
            Success status
        """
        try:
            logger.info(f"Loading model from {self.model_path or self.model_id}")
            
            # Load tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path or self.model_id,
                use_fast=True
            )
            
            # Load model with 8-bit quantization if available
            try:
                import bitsandbytes
                has_bitsandbytes = True
            except ImportError:
                has_bitsandbytes = False
            
            # Configuration for model loading
            kwargs = {
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32,
                "device_map": "auto",
                "trust_remote_code": True,
            }
            
            # Add quantization if available
            if has_bitsandbytes and self.device == "cuda":
                kwargs["load_in_8bit"] = True
                logger.info("Using 8-bit quantization")
            
            # Load model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path or self.model_id,
                **kwargs
            )
            
            self.is_loaded = True
            logger.info(f"Model loaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in synchronous model loading: {e}")
            return False
    
    async def generate_text(self, 
                          prompt: str, 
                          max_tokens: Optional[int] = None,
                          temperature: Optional[float] = None,
                          system_prompt: Optional[str] = None) -> str:
        """Generate text from a prompt.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate (overrides default)
            temperature: Temperature for generation (overrides default)
            system_prompt: Optional system prompt to prepend
            
        Returns:
            Generated text
        """
        if not self.is_loaded:
            success = await self.load_model()
            if not success:
                return "Error: Failed to load model"
        
        try:
            # Run generation in a separate thread
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                None, 
                lambda: self._generate_text_sync(
                    prompt, 
                    max_tokens or self.max_tokens,
                    temperature or self.temperature,
                    system_prompt
                )
            )
        except Exception as e:
            logger.error(f"Error generating text: {e}")
            return f"Error during text generation: {str(e)}"
    
    def _generate_text_sync(self, 
                          prompt: str, 
                          max_tokens: int,
                          temperature: float,
                          system_prompt: Optional[str] = None) -> str:
        """Synchronous method to generate text (runs in executor).
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Temperature for generation
            system_prompt: Optional system prompt
            
        Returns:
            Generated text
        """
        try:
            # Format prompt with system prompt if provided
            full_prompt = prompt
            if system_prompt:
                # Different models have different system prompt formats
                # Here's a generic approach that works for many models
                full_prompt = f"{system_prompt}\n\n{prompt}"
            
            # Tokenize input
            inputs = self.tokenizer(full_prompt, return_tensors="pt").to(self.device)
            
            # Set generation parameters
            gen_kwargs = {
                "max_new_tokens": max_tokens,
                "temperature": temperature,
                "do_sample": temperature > 0,
                "top_p": 0.9,
                "top_k": 50,
                "repetition_penalty": 1.2,
                "no_repeat_ngram_size": 3
            }
            
            # Generate
            output = self.model.generate(
                **inputs,
                **gen_kwargs
            )
            
            # Decode and return only the generated part
            decoded = self.tokenizer.decode(output[0], skip_special_tokens=True)
            
            # Remove the prompt from the beginning to get only generated text
            if decoded.startswith(full_prompt):
                return decoded[len(full_prompt):].strip()
            
            return decoded.strip()
            
        except Exception as e:
            logger.error(f"Error in synchronous text generation: {e}")
            return f"Error: {str(e)}"
    
    async def count_tokens(self, text: str) -> int:
        """Count the number of tokens in a text.
        
        Args:
            text: Input text
            
        Returns:
            Number of tokens
        """
        if not self.is_loaded:
            success = await self.load_model()
            if not success:
                return 0
        
        try:
            # Run in a separate thread
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, lambda: len(self.tokenizer.encode(text)))
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            return 0
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get information about the loaded model.
        
        Returns:
            Model information dictionary
        """
        if not self.is_loaded:
            return {
                "status": "not_loaded",
                "model_id": self.model_id,
                "model_path": self.model_path,
                "device": self.device
            }
        
        # Get model size in parameters
        model_size = sum(p.numel() for p in self.model.parameters())
        
        # Get memory usage if on CUDA
        memory_usage = None
        if self.device == "cuda":
            try:
                memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # GB
                memory_reserved = torch.cuda.memory_reserved() / (1024 ** 3)    # GB
                memory_usage = {
                    "allocated_gb": memory_allocated,
                    "reserved_gb": memory_reserved
                }
            except Exception:
                pass
        
        return {
            "status": "loaded",
            "model_id": self.model_id,
            "model_path": self.model_path,
            "device": self.device,
            "parameters": model_size,
            "memory_usage": memory_usage,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature,
            "tokenizer_type": type(self.tokenizer).__name__,
            "model_type": type(self.model).__name__
        }
    
    async def unload_model(self) -> bool:
        """Unload the model to free memory.
        
        Returns:
            Success status
        """
        if not self.is_loaded:
            return True
        
        try:
            # Run in a separate thread
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self._unload_model_sync)
        except Exception as e:
            logger.error(f"Error unloading model: {e}")
            return False
    
    def _unload_model_sync(self) -> bool:
        """Synchronous method to unload the model (runs in executor).
        
        Returns:
            Success status
        """
        try:
            # Delete model and tokenizer
            if hasattr(self, 'model') and self.model is not None:
                del self.model
                self.model = None
                
            if hasattr(self, 'tokenizer') and self.tokenizer is not None:
                del self.tokenizer
                self.tokenizer = None
                
            # Force garbage collection
            import gc
            gc.collect()
            
            # Clear CUDA cache if applicable
            if self.device == "cuda":
                torch.cuda.empty_cache()
                
            self.is_loaded = False
            logger.info("Model unloaded successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in synchronous model unloading: {e}")
            return False


class ModelPool:
    """Manages a pool of models for efficient resource usage."""
    
    def __init__(self, max_models: int = 2):
        """Initialize the model pool.
        
        Args:
            max_models: Maximum number of loaded models at once
        """
        self.max_models = max_models
        self.models: Dict[str, ModelInterfaceHelper] = {}
        self.last_used: Dict[str, float] = {}
        self.lock = asyncio.Lock()
        
        logger.info(f"Model pool initialized (max models: {max_models})")
    
    async def get_model(self, 
                       model_path: Optional[str] = None,
                       model_id: Optional[str] = None) -> ModelInterfaceHelper:
        """Get a model from the pool, loading it if necessary.
        
        Args:
            model_path: Local path to the model
            model_id: Hugging Face model ID
            
        Returns:
            ModelInterfaceHelper instance
        """
        # Generate a key for the model
        model_key = model_path or model_id or "default"
        
        async with self.lock:
            # If model exists in pool, return it
            if model_key in self.models:
                self.last_used[model_key] = time.time()
                return self.models[model_key]
                
            # If pool is full, unload least recently used model
            if len(self.models) >= self.max_models:
                await self._unload_least_recently_used()
                
            # Create and load new model
            model = ModelInterfaceHelper(
                model_path=model_path,
                model_id=model_id
            )
            
            # Load the model
            await model.load_model()
            
            # Add to pool
            self.models[model_key] = model
            self.last_used[model_key] = time.time()
            
            return model
    
    async def _unload_least_recently_used(self) -> None:
        """Unload the least recently used model."""
        if not self.models:
            return
            
        # Find least recently used model
        lru_key = min(self.last_used.items(), key=lambda x: x[1])[0]
        
        # Unload it
        if lru_key in self.models:
            logger.info(f"Unloading least recently used model: {lru_key}")
            await self.models[lru_key].unload_model()
            del self.models[lru_key]
            del self.last_used[lru_key]
    
    async def unload_all(self) -> None:
        """Unload all models in the pool."""
        async with self.lock:
            for model_key, model in list(self.models.items()):
                logger.info(f"Unloading model: {model_key}")
                await model.unload_model()
                
            self.models.clear()
            self.last_used.clear()
            
        logger.info("All models unloaded")


# Module-level model pool instance for shared access
model_pool = ModelPool()


async def get_model(model_path: Optional[str] = None, model_id: Optional[str] = None) -> ModelInterfaceHelper:
    """Get a model from the shared pool.
    
    Args:
        model_path: Local path to the model
        model_id: Hugging Face model ID
        
    Returns:
        ModelInterfaceHelper instance
    """
    return await model_pool.get_model(model_path, model_id)


async def unload_all_models() -> None:
    """Unload all models from the shared pool."""
    await model_pool.unload_all()


async def test_model_generation(model_path: Optional[str] = None, model_id: Optional[str] = None) -> Dict[str, Any]:
    """Test model text generation.
    
    Args:
        model_path: Local path to the model
        model_id: Hugging Face model ID
        
    Returns:
        Test results
    """
    try:
        start_time = time.time()
        
        # Get model
        model = await get_model(model_path, model_id)
        
        # Generate text
        test_prompt = "Расскажи короткую историю о приключениях робота, который обрел сознание."
        generated_text = await model.generate_text(
            prompt=test_prompt,
            max_tokens=100,
            temperature=0.7
        )
        
        # Get model info
        model_info = model.get_model_info()
        
        # Calculate duration
        duration = time.time() - start_time
        
        return {
            "success": True,
            "model_info": model_info,
            "test_prompt": test_prompt,
            "generated_text": generated_text,
            "duration_seconds": duration
        }
        
    except Exception as e:
        logger.error(f"Error testing model: {e}")
        return {
            "success": False,
            "error": str(e)
        }
