"""
Local embedding generation using Sentence Transformers.
This is a fallback when Gemini API is unavailable or too slow.
"""

import numpy as np
from typing import List
from loguru import logger

try:
    from sentence_transformers import SentenceTransformer
    SENTENCE_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SENTENCE_TRANSFORMERS_AVAILABLE = False
    logger.warning("sentence-transformers not installed. Install with: pip install sentence-transformers")


class LocalEmbedder:
    """Generate embeddings using local Sentence Transformers model."""
    
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """
        Initialize local embedder.
        
        Args:
            model_name: Sentence Transformers model name
                       Default: all-MiniLM-L6-v2 (384-dim, fast, good quality)
                       Alternative: all-mpnet-base-v2 (768-dim, slower, better quality)
        """
        if not SENTENCE_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "sentence-transformers is required for local embeddings. "
                "Install with: pip install sentence-transformers"
            )
        
        logger.info(f"Loading local embedding model: {model_name}")
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name
        self.embedding_dim = self.model.get_sentence_embedding_dimension()
        logger.info(f"Initialized LocalEmbedder with {self.embedding_dim}-dimensional embeddings")
    
    def embed_text(self, text: str) -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text
            
        Returns:
            Embedding vector
        """
        try:
            embedding = self.model.encode(text, convert_to_numpy=True)
            return embedding
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 32, show_progress: bool = True) -> np.ndarray:
        """
        Generate embeddings for multiple texts.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for encoding
            show_progress: Show progress bar
            
        Returns:
            Array of embeddings (n_texts, embedding_dim)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        logger.info(f"Using batch size: {batch_size}")
        
        try:
            embeddings = self.model.encode(
                texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True
            )
            
            logger.info(f"✅ Generated {len(embeddings)} embeddings")
            return embeddings
            
        except Exception as e:
            logger.error(f"Batch embedding failed: {e}")
            raise
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for query.
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query)


# Recommended models:
# - all-MiniLM-L6-v2: 384-dim, fast, good for most tasks
# - all-mpnet-base-v2: 768-dim, slower, better quality (matches old Gemini dimension!)
# - paraphrase-multilingual-MiniLM-L12-v2: 384-dim, supports multiple languages
