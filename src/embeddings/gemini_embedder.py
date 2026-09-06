"""
Gemini embedding generation.
"""

import google.generativeai as genai
import numpy as np
from typing import List
from loguru import logger
import time


class GeminiEmbedder:
    """Generate embeddings using Gemini API."""
    
    def __init__(self, api_key: str, model_name: str = "models/gemini-embedding-001"):
        """
        Initialize embedder.
        
        Args:
            api_key: Gemini API key
            model_name: Embedding model name (e.g., models/embedding-001)
        """
        genai.configure(api_key=api_key)
        self.model_name = model_name
        logger.info(f"Initialized GeminiEmbedder with model: {model_name}")
    
    def embed_text(self, text: str, task_type: str = "retrieval_document") -> np.ndarray:
        """
        Generate embedding for single text.
        
        Args:
            text: Input text
            task_type: Task type (retrieval_document or retrieval_query)
            
        Returns:
            Embedding vector (768-dim)
        """
        try:
            result = genai.embed_content(
                model=self.model_name,
                content=text,
                task_type=task_type
            )
            return np.array(result['embedding'])
        
        except Exception as e:
            logger.error(f"Embedding generation failed: {e}")
            raise
    
    def embed_batch(self, texts: List[str], batch_size: int = 100, delay: float = 0.1) -> np.ndarray:
        """
        Generate embeddings for multiple texts using batch API.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for API calls (Gemini supports up to 100)
            delay: Delay between batch API calls (seconds)
            
        Returns:
            Array of embeddings (n_texts, 768)
        """
        logger.info(f"Generating embeddings for {len(texts)} texts")
        logger.info(f"Using batch size: {batch_size}, delay: {delay}s per batch")
        
        embeddings = []
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = i//batch_size + 1
            total_batches = (len(texts)-1)//batch_size + 1
            
            try:
                # Use batch embedding API (much faster!)
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                
                # Extract embeddings from batch result
                batch_embeddings = [np.array(emb) for emb in result['embedding']]
                embeddings.extend(batch_embeddings)
                
                logger.info(f"✅ Batch {batch_num}/{total_batches} complete ({len(embeddings)}/{len(texts)} texts)")
                
                # Rate limiting between batches
                if i + batch_size < len(texts):
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"Batch {batch_num} failed: {e}")
                logger.warning("Falling back to individual embedding...")
                
                # Fallback: process individually
                for text in batch:
                    try:
                        emb = self.embed_text(text, task_type="retrieval_document")
                        embeddings.append(emb)
                        time.sleep(delay)
                    except Exception as e2:
                        logger.error(f"Individual embedding failed: {e2}")
                        # Use zero vector as placeholder
                        embeddings.append(np.zeros(768))
        
        return np.array(embeddings)
    
    def embed_query(self, query: str) -> np.ndarray:
        """
        Generate embedding for query (uses different task type).
        
        Args:
            query: Query text
            
        Returns:
            Query embedding vector
        """
        return self.embed_text(query, task_type="retrieval_query")
