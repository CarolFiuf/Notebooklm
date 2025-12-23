"""
Custom Embeddings class for FPT Cloud API
Bypasses langchain-openai's tokenization logic which is incompatible with LiteLLM proxy
"""

from typing import List
from langchain_core.embeddings import Embeddings
from openai import OpenAI


class FPTEmbeddings(Embeddings):
    """
    Custom embeddings class for FPT Cloud Vietnamese_Embedding model.

    This class directly calls the FPT API without tokenization preprocessing,
    which avoids the token ID issue with langchain-openai's OpenAIEmbeddings.
    """

    def __init__(
        self,
        model: str,
        api_key: str,
        base_url: str,
        **kwargs
    ):
        """
        Initialize FPT Embeddings.

        Args:
            model: Model name (e.g., "Vietnamese_Embedding")
            api_key: FPT Cloud API key
            base_url: FPT Cloud API base URL
        """
        self.model = model
        self.client = OpenAI(
            api_key=api_key,
            base_url=base_url,
            **kwargs
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a list of documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        # Call API directly - NO tokenization!
        response = self.client.embeddings.create(
            model=self.model,
            input=texts  # Send strings directly, not token IDs
        )

        # Extract embeddings from response
        embeddings = [item.embedding for item in response.data]
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        """
        Embed a single query text.

        Args:
            text: Text string to embed

        Returns:
            Embedding vector
        """
        # Use embed_documents for single text and return first result
        return self.embed_documents([text])[0]
