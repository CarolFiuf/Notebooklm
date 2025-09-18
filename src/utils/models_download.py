import requests
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Optional

from config.config import settings

logger = logging.getLogger(__name__)

class ModelDownloader:
    """Utility for downloading GGUF models"""
    
    def __init__(self):
        self.models_dir = settings.MODELS_DIR
        self.models_dir.mkdir(parents=True, exist_ok=True)
    
    def download_model(
        self, 
        url: str, 
        filename: Optional[str] = None,
        force_download: bool = False
    ) -> Path:
        """
        Download GGUF model from URL
        
        Args:
            url: Download URL
            filename: Local filename (auto-detect if None)
            force_download: Re-download even if file exists
            
        Returns:
            Path to downloaded model
        """
        try:
            # Determine filename
            if not filename:
                filename = url.split('/')[-1]
                if not filename.endswith('.gguf'):
                    filename += '.gguf'
            
            model_path = self.models_dir / filename
            
            # Check if already exists
            if model_path.exists() and not force_download:
                logger.info(f"Model already exists: {model_path}")
                return model_path
            
            logger.info(f"Downloading model from {url}")
            logger.info(f"Saving to: {model_path}")
            
            # Download with progress bar
            response = requests.get(url, stream=True)
            response.raise_for_status()
            
            total_size = int(response.headers.get('content-length', 0))
            
            with open(model_path, 'wb') as f:
                with tqdm(
                    total=total_size, 
                    unit='B', 
                    unit_scale=True, 
                    desc=f"Downloading {filename}"
                ) as pbar:
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            pbar.update(len(chunk))
            
            logger.info(f"Model downloaded successfully: {model_path}")
            return model_path
            
        except Exception as e:
            # Clean up partial download
            if model_path.exists():
                model_path.unlink()
            logger.error(f"Model download failed: {e}")
            raise
    
    def list_available_models(self) -> dict:
        """List some popular GGUF models available for download"""
        return {
            "qwen2.5-7b-instruct": {
                "q4_k_m": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q4_k_m.gguf",
                "q8_0": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-q8_0.gguf",
                "f16": "https://huggingface.co/Qwen/Qwen2.5-7B-Instruct-GGUF/resolve/main/qwen2.5-7b-instruct-f16.gguf"
            },
            "llama-3.1-8b-instruct": {
                "q4_k_m": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf",
                "q8_0": "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q8_0.gguf"
            },
            "mistral-7b-instruct": {
                "q4_k_m": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q4_K_M.gguf",
                "q8_0": "https://huggingface.co/TheBloke/Mistral-7B-Instruct-v0.2-GGUF/resolve/main/mistral-7b-instruct-v0.2.Q8_0.gguf"
            }
        }
    
    def download_recommended_model(self) -> Path:
        """Download the default recommended model"""
        return self.download_model(
            url=settings.LLM_MODEL_URL,
            filename=settings.LLM_MODEL_NAME
        )

# Usage example
def download_default_model():
    """Download default model specified in settings"""
    downloader = ModelDownloader()
    return downloader.download_recommended_model()