import logging
from typing import List, Dict, Any, Optional
import os
from datetime import datetime
from pathlib import Path
import threading

from llama_cpp import Llama

from config.settings import settings
from src.utils.exceptions import LLMServiceError
from src.utils.retry_utils import llm_retry

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
        """Check that model file exists (no auto-download)"""
        try:
            if not self.model_path.exists():
                raise LLMServiceError(
                    f"Model not found at {self.model_path}\n"
                    f"Please download the model manually or place it in {self.model_path.parent}"
                )
            else:
                logger.info(f"Model found at {self.model_path}")

        except Exception as e:
            logger.error(f"Error checking model: {e}")
            raise LLMServiceError(f"Model check failed: {e}")
    
    def _strip_thinking_content(self, text: str) -> str:
        """
        ✅ IMPROVED: Remove thinking/reasoning content from Qwen model output

        Qwen models can output thinking in multiple ways:
        1. <think>...</think> tags
        2. Thinking paragraphs starting with "Okay", "Let me", "I need to", etc.
        3. Thinking content appearing ANYWHERE in the response (not just at start)

        Args:
            text: Raw generated text

        Returns:
            Text with thinking content removed
        """
        import re

        # Pattern 1: Remove <think>...</think> blocks
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL | re.IGNORECASE)

        # Pattern 2: Remove thinking content that appears ANYWHERE in response
        # Common thinking patterns (case-insensitive):
        thinking_patterns = [
            # English thinking patterns
            r'\n\s*Okay,\s+let[\'s]?\s+',  # "Okay, let's" or "Okay, let"
            r'\n\s*Let\s+me\s+',            # "Let me"
            r'\n\s*I\s+need\s+to\s+',       # "I need to"
            r'\n\s*First,?\s+I\s+',         # "First, I" or "First I"
            r'\n\s*Wait,?\s+',              # "Wait," or "Wait"
            r'\n\s*I\s+should\s+',          # "I should"
            r'\n\s*Alright,?\s+',           # "Alright," or "Alright"
            r'\n\s*So,?\s+the\s+user',      # "So, the user" or "So the user"
            r'\n\s*The\s+user[\'s]?\s+(question|message|asked)',  # "The user's question"
            r'\n\s*Okay\s+I\s+need\s+to',   # "Okay I need to"
        ]

        # Find the earliest thinking pattern in the text
        earliest_pos = len(text)
        for pattern in thinking_patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE)
            if match:
                pos = match.start()
                if pos < earliest_pos:
                    earliest_pos = pos

        # If thinking pattern found, cut everything from that point
        if earliest_pos < len(text):
            text = text[:earliest_pos].strip()
            logger.debug(f"✂️ Stripped thinking content starting at position {earliest_pos}")

        # Pattern 3: If text STARTS with thinking-like content (fallback)
        # Some models might not have newlines before thinking
        thinking_starts = [
            'okay, let', 'let me', 'i need to', 'first, i',
            'wait,', 'i should', 'alright,', 'so, the user'
        ]
        text_lower = text.lower()
        for start_pattern in thinking_starts:
            if text_lower.startswith(start_pattern):
                # Try to find where actual answer starts (usually after multiple sentences)
                sentences = text.split('.')
                # Skip first few thinking sentences, keep rest
                if len(sentences) > 3:
                    text = '.'.join(sentences[3:]).strip()
                    logger.debug(f"✂️ Stripped thinking at start with pattern: {start_pattern}")
                break

        # Clean up extra whitespace
        text = re.sub(r'\n{3,}', '\n\n', text)  # Max 2 newlines
        text = text.strip()

        return text

    def _initialize_llm(self):
        """Initialize llama.cpp engine"""
        try:
            logger.info(f"Initializing llama.cpp with model: {self.model_path}")
            start_time = datetime.now()
            
            # Initialize Llama model with safe memory settings
            cpu_count = os.cpu_count() or 1
            safe_n_batch = self.n_batch

            self.llm = Llama(
                model_path=str(self.model_path),
                n_ctx=self.context_length,  # Context window
                n_batch=min(safe_n_batch, 512),  # Capped at 512 for stability
                n_threads=min(self.n_threads, cpu_count),  # Cap to available CPUs
                n_gpu_layers=self.n_gpu_layers,  # GPU layers (0 = CPU only)
                verbose=True,  # Enable verbose for debugging
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
    
    @llm_retry 
    def generate_response(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None
    ) -> str:
        """
        Generate response using llama.cpp with automatic retry on failures.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Top-p sampling parameter
            stop: Stop sequences

        Returns:
            Generated text response

        Note:
            This method automatically retries up to 5 times with exponential backoff
            (4-30 seconds) on transient failures.
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

            response = self.llm(**generation_kwargs)

            # Extract generated text
            if response and "choices" in response and len(response["choices"]) > 0:
                generated_text = response["choices"][0]["text"].strip()

                # 🔍 DEBUG: Log raw LLM output before stripping thinking content
                logger.info("=" * 80)
                logger.info("[RAW LLM OUTPUT - BEFORE STRIPPING]")
                logger.info("=" * 80)
                logger.info(generated_text)
                logger.info("=" * 80)

                # ✅ FIX: Remove thinking/reasoning content from Qwen models
                # Qwen models use <think>...</think> tags for reasoning
                generated_text = self._strip_thinking_content(generated_text)

                # 🔍 DEBUG: Log cleaned output after stripping
                logger.info("[CLEANED OUTPUT - AFTER STRIPPING]")
                logger.info("=" * 80)
                logger.info(generated_text)
                logger.info("=" * 80)

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
    
    @llm_retry  # ✅ Retry on transient failures
    def generate_stream(
        self,
        prompt: str,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        top_p: Optional[float] = None,
        stop: Optional[List[str]] = None
    ):
        """
        Generate streaming response (generator) with automatic retry.

        Note:
            This method automatically retries up to 5 times with exponential backoff
            (4-30 seconds) on transient failures.
        """
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
    
    @llm_retry  # ✅ Retry on transient failures
    def health_check(self) -> bool:
        """
        Check if LLM service is healthy with automatic retry.

        Note:
            This method automatically retries up to 5 times with exponential backoff
            (4-30 seconds) on transient failures.
        """
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
    
    def tokenize(self, text: str) -> List[int]:
        """
        Tokenize text using llama.cpp's built-in tokenizer

        Args:
            text: Text to tokenize

        Returns:
            List of token IDs
        """
        try:
            if not self.llm:
                raise LLMServiceError("LLM not initialized")

            # llama.cpp expects bytes
            tokens = self.llm.tokenize(text.encode('utf-8'))
            return tokens

        except Exception as e:
            logger.error(f"Error tokenizing text: {e}")
            return []

    def detokenize(self, tokens: List[int]) -> str:
        """
        Convert tokens back to text

        Args:
            tokens: List of token IDs

        Returns:
            Decoded text string
        """
        try:
            if not self.llm:
                raise LLMServiceError("LLM not initialized")

            # llama.cpp returns bytes
            text = self.llm.detokenize(tokens).decode('utf-8', errors='ignore')
            return text

        except Exception as e:
            logger.error(f"Error detokenizing: {e}")
            return ""

    def get_exact_token_count(self, text: str) -> int:
        """
        Get exact token count using model's tokenizer

        Args:
            text: Text to count tokens for

        Returns:
            Exact number of tokens
        """
        try:
            tokens = self.tokenize(text)
            return len(tokens)
        except Exception as e:
            logger.error(f"Error counting tokens: {e}")
            # Fallback to estimation
            return len(text) // 4

    def estimate_tokens(self, text: str) -> int:
        """
        Rough token count estimation (for backward compatibility)

        Note: Use get_exact_token_count() for accurate counting
        """
        # Simple estimation: ~4 characters per token for most languages
        return len(text) // 4

    def smart_truncate_to_tokens(
        self,
        text: str,
        max_tokens: int,
        from_end: bool = True
    ) -> str:
        """
        Truncate text to exact token count

        Args:
            text: Text to truncate
            max_tokens: Maximum number of tokens
            from_end: If True, keep beginning; if False, keep end

        Returns:
            Truncated text with exactly max_tokens or less
        """
        try:
            tokens = self.tokenize(text)

            if len(tokens) <= max_tokens:
                return text

            if from_end:
                # Keep beginning
                truncated_tokens = tokens[:max_tokens]
            else:
                # Keep end
                truncated_tokens = tokens[-max_tokens:]

            return self.detokenize(truncated_tokens)

        except Exception as e:
            logger.error(f"Error truncating text: {e}")
            # Fallback to character-based truncation
            char_limit = max_tokens * 4
            if from_end:
                return text[:char_limit]
            else:
                return text[-char_limit:]
    
    def get_context_window_usage(self, prompt: str, max_tokens: int = None) -> Dict[str, Any]:
        """
        Calculate context window usage with exact token counting

        Args:
            prompt: Input prompt text
            max_tokens: Maximum tokens to generate

        Returns:
            Dictionary with token usage statistics
        """
        max_tokens = max_tokens or self.max_tokens

        # Use exact token counting instead of estimation
        prompt_tokens = self.get_exact_token_count(prompt)
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
