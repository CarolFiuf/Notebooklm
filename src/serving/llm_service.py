import logging
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
from pathlib import Path
import threading

from llama_cpp import Llama

from config.config import settings
from src.utils.exceptions import LLMServiceError

logger = logging.getLogger(__name__)

class LlamaCppService:
    """llama.cpp service for fast LLM inference"""
    
    def __init__(self):
        self.model_path = settings.model_path
        self.max_tokens = settings.LLM_MAX_TOKENS
        self.temperature = settings.LLM_TEMPERATURE
        self.top_p = settings.LLM_TOP_P
        self.context_length = settings.LLM_CONTEXT_LENGTH
        
        # llama.cpp specific settings
        self.n_gpu_layers = settings.LLAMACPP_N_GPU_LAYERS
        self.n_batch = settings.LLAMACPP_N_BATCH  
        self.n_threads = settings.LLAMACPP_N_THREADS
        self.verbose = settings.LLAMACPP_VERBOSE
        
        self.llm = None
        self._lock = threading.Lock()
        self._ensure_model_exists()
        self._initialize_llm()
    
    def _ensure_model_exists(self):
        """Ensure model file exists, download if necessary"""
        try:
            if not self.model_path.exists():
                logger.info(f"Model not found at {self.model_path}")
                self._download_model()
            else:
                logger.info(f"Model found at {self.model_path}")
                
        except Exception as e:
            logger.error(f"Error checking model: {e}")
            raise LLMServiceError(f"Model check failed: {e}")
    
    def _download_model(self):
        """Download GGUF model from HuggingFace"""
        try:
            import requests
            from tqdm import tqdm
            
            model_url = settings.LLM_MODEL_URL
            if not model_url:
                raise LLMServiceError("No model URL specified for download")
            
            logger.info(f"Downloading model from {model_url}")
            logger.info("This may take a while depending on your connection...")
            
            # Create models directory
            self.model_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Download with progress bar
            response = requests.get(model_url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(self.model_path, 'wb') as f:
                with tqdm(total=total_size, unit='B', unit_scale=True, desc="Downloading") as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Model downloaded successfully to {self.model_path}")
            
        except Exception as e:
            # Clean up partial download
            if self.model_path.exists():
                self.model_path.unlink()
            logger.error(f"Model download failed: {e}")
            raise LLMServiceError(f"Model download failed: {e}")
    
    def _initialize_llm(self):
        """Initialize llama.cpp engine"""
        try:
            logger.info(f"Initializing llama.cpp with model: {self.model_path}")
            start_time = datetime.now()
            
            # Initialize Llama model with safe memory settings
            cpu_count = os.cpu_count() or 1
            safe_n_batch = min(self.n_batch, self.context_length)

            # llama.cpp on CPU can assert when the decode micro-batch is too large.
            # Empirically, staying at <=64 avoids the out_ids mismatch we observed.
            if self.n_gpu_layers == 0 and safe_n_batch > 64:
                logger.warning(
                    "Reducing n_batch from %s to 64 for CPU-only stability",
                    safe_n_batch,
                )
                safe_n_batch = 64

            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_length,  # Context window
                n_batch=safe_n_batch,  # Use safe batch size
                n_threads=min(self.n_threads, cpu_count),  # Cap to available CPUs
                n_gpu_layers=self.n_gpu_layers,  # GPU layers (0 = CPU only)
                verbose=self.verbose,
                use_mmap=True,  # Use memory mapping
                use_mlock=False,  # Don't lock memory
                seed=42,  # For reproducible results
                logits_all=False,  # Don't compute all logits
                embedding=False,  # Not used for embeddings
            )
            
            init_time = (datetime.now() - start_time).total_seconds()
            logger.info(f"llama.cpp engine initialized successfully in {init_time:.2f}s")
            
            # Log model info
            logger.info(f"Context length: {self.context_length}")
            logger.info(f"GPU layers: {self.n_gpu_layers}")
            logger.info(f"CPU threads: {self.n_threads}")
            logger.info(f"Batch size (n_batch): {safe_n_batch}")
            
        except Exception as e:
            # Include traceback for easier diagnosis in container logs
            logger.exception("Error initializing llama.cpp")
            raise LLMServiceError(f"llama.cpp initialization failed: {e}")
    
    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate response using llama.cpp
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences
            
        Returns:
            Generated text response
        """
        try:
            if not self.llm:
                raise LLMServiceError("LLM not initialized")
            
            # Prepare generation parameters
            generation_kwargs = {
                "prompt": prompt,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
                "echo": False,  # Don't echo the prompt
                "stop": stop or ["</s>", "<|endoftext|>", "<|im_end|>", "Human:", "User:"],
                "stream": False
            }
            
            # Generate response
            logger.debug(f"Generating response for prompt: {prompt[:100]}...")
            
            with self._lock:
                response = self.llm(**generation_kwargs)
            
            # Extract generated text
            if response and "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["text"].strip()
                
                logger.debug(f"Generated response: {generated_text[:100]}...")
                return generated_text
            else:
                logger.warning("No response generated from llama.cpp")
                return "I apologize, but I couldn't generate a response."
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            raise LLMServiceError(f"Response generation failed: {e}")
    
    def generate_batch(
        self,
        prompts: List[str],
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None
    ) -> List[str]:
        """Generate responses for multiple prompts (sequential for llama.cpp)"""
        try:
            if not prompts:
                return []
            
            responses = []
            for prompt in prompts:
                response = self.generate_response(
                    prompt=prompt,
                    max_tokens=max_tokens,
                    temperature=temperature,
                    top_p=top_p
                )
                responses.append(response)
            
            return responses
            
        except Exception as e:
            logger.error(f"Error generating batch responses: {e}")
            raise LLMServiceError(f"Batch generation failed: {e}")
    
    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None
    ):
        """Generate streaming response (generator)"""
        try:
            if not self.llm:
                raise LLMServiceError("LLM not initialized")
            
            generation_kwargs = {
                "prompt": prompt,
                "max_tokens": max_tokens or self.max_tokens,
                "temperature": temperature or self.temperature,
                "top_p": top_p or self.top_p,
                "echo": False,
                "stop": stop or ["</s>", "<|endoftext|>", "<|im_end|>"],
                "stream": True
            }
            
            # Generate streaming response
            with self._lock:
                for chunk in self.llm(**generation_kwargs):
                    if chunk and "choices" in chunk and len(chunk["choices"]) > 0:
                        token = chunk["choices"][0].get("text", "")
                        if token:
                            yield token
                        
        except Exception as e:
            logger.error(f"Error generating streaming response: {e}")
            raise LLMServiceError(f"Streaming generation failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        try:
            model_size = self.model_path.stat().st_size if self.model_path.exists() else 0
            
            return {
                'model_path': str(self.model_path),
                'model_name': settings.LLM_MODEL_NAME,
                'model_size_mb': round(model_size / (1024 * 1024), 2),
                'context_length': self.context_length,
                'n_gpu_layers': self.n_gpu_layers,
                'n_threads': self.n_threads,
                'n_batch': self.n_batch,
                'backend': 'llama.cpp',
                'is_initialized': self.llm is not None,
                'model_exists': self.model_path.exists()
            }
        except Exception as e:
            logger.error(f"Error getting model info: {e}")
            return {'error': str(e)}
    
    def health_check(self) -> bool:
        """Check if LLM service is healthy"""
        try:
            if not self.llm:
                return False
                
            # Simple test generation
            test_response = self.generate_response(
                "Hello", 
                max_tokens=5, 
                temperature=0.1
            )
            return bool(test_response and len(test_response) > 0)
        except Exception as e:
            logger.error(f"LLM health check failed: {e}")
            return False
    
    def estimate_tokens(self, text: str) -> int:
        """Rough token count estimation"""
        # Simple estimation: ~4 characters per token for most languages
        return len(text) // 4
    
    def get_context_window_usage(self, prompt: str, max_tokens: int = None) -> Dict[str, Any]:
        """Calculate context window usage"""
        max_tokens = max_tokens or self.max_tokens
        prompt_tokens = self.estimate_tokens(prompt)
        total_available = self.context_length
        
        return {
            'prompt_tokens': prompt_tokens,
            'max_new_tokens': max_tokens,
            'total_tokens': prompt_tokens + max_tokens,
            'context_window': total_available,
            'remaining_tokens': total_available - prompt_tokens - max_tokens,
            'usage_percentage': ((prompt_tokens + max_tokens) / total_available) * 100
        }

# Alias for backward compatibility
LLMService = LlamaCppService
