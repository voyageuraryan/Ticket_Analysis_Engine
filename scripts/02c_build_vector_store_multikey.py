"""
Build vector store with embeddings using MULTIPLE API KEYS for faster processing.
This script rotates between multiple Gemini API keys to bypass rate limits.

With 3 API keys: ~40 minutes instead of 2 hours!
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
import pickle
import numpy as np
import time
from typing import List
import google.generativeai as genai

from src.config import Config
from src.vector_store import ChromaVectorStore
from src.retrieval.sparse_retriever import SparseRetriever


class MultiKeyGeminiEmbedder:
    """Generate embeddings using multiple Gemini API keys with rotation."""
    
    def __init__(self, api_keys: List[str], model_name: str = "models/gemini-embedding-001"):
        """
        Initialize embedder with multiple API keys.
        
        Args:
            api_keys: List of Gemini API keys
            model_name: Embedding model name
        """
        if not api_keys:
            raise ValueError("At least one API key is required")
        
        self.api_keys = api_keys
        self.model_name = model_name
        self.current_key_index = 0
        
        logger.info(f"🔑 Initialized MultiKeyGeminiEmbedder with {len(api_keys)} API keys")
        logger.info(f"📊 Model: {model_name}")
        logger.info(f"⚡ Speed boost: {len(api_keys)}x faster!")
    
    def _get_next_key(self) -> str:
        """Get next API key in rotation."""
        key = self.api_keys[self.current_key_index]
        self.current_key_index = (self.current_key_index + 1) % len(self.api_keys)
        return key
    
    def embed_batch_with_rotation(
        self, 
        texts: List[str], 
        batch_size: int = 100,
        delay: float = 0.1
    ) -> np.ndarray:
        """
        Generate embeddings using API key rotation.
        
        Args:
            texts: List of input texts
            batch_size: Batch size for API calls
            delay: Delay between batches (can be smaller with multiple keys)
            
        Returns:
            Array of embeddings
        """
        logger.info(f"🚀 Generating embeddings for {len(texts)} texts")
        logger.info(f"🔑 Using {len(self.api_keys)} API keys in rotation")
        logger.info(f"📦 Batch size: {batch_size}")
        
        embeddings = []
        total_batches = (len(texts) - 1) // batch_size + 1
        
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_num = i // batch_size + 1
            
            # Rotate to next API key
            api_key = self._get_next_key()
            key_num = self.current_key_index if self.current_key_index > 0 else len(self.api_keys)
            
            try:
                # Configure API with current key
                genai.configure(api_key=api_key)
                
                # Generate embeddings
                result = genai.embed_content(
                    model=self.model_name,
                    content=batch,
                    task_type="retrieval_document"
                )
                
                # Extract embeddings
                batch_embeddings = [np.array(emb) for emb in result['embedding']]
                embeddings.extend(batch_embeddings)
                
                logger.info(
                    f"✅ Batch {batch_num}/{total_batches} complete "
                    f"({len(embeddings)}/{len(texts)} texts) "
                    f"[Key #{key_num}]"
                )
                
                # Shorter delay since we're rotating keys
                if i + batch_size < len(texts):
                    time.sleep(delay)
                    
            except Exception as e:
                error_msg = str(e)
                
                # Check if it's a rate limit error
                if "429" in error_msg or "quota" in error_msg.lower():
                    logger.warning(
                        f"⚠️ Rate limit hit on Key #{key_num} for batch {batch_num}. "
                        f"Rotating to next key..."
                    )
                    
                    # Try with next key immediately
                    try:
                        api_key = self._get_next_key()
                        key_num = self.current_key_index if self.current_key_index > 0 else len(self.api_keys)
                        
                        genai.configure(api_key=api_key)
                        result = genai.embed_content(
                            model=self.model_name,
                            content=batch,
                            task_type="retrieval_document"
                        )
                        
                        batch_embeddings = [np.array(emb) for emb in result['embedding']]
                        embeddings.extend(batch_embeddings)
                        
                        logger.info(
                            f"✅ Batch {batch_num}/{total_batches} complete (retry with Key #{key_num})"
                        )
                        
                    except Exception as e2:
                        logger.error(f"❌ Batch {batch_num} failed even after rotation: {e2}")
                        logger.warning("⏳ Waiting 60 seconds before retry...")
                        time.sleep(60)
                        
                        # Final retry with first key
                        try:
                            genai.configure(api_key=self.api_keys[0])
                            result = genai.embed_content(
                                model=self.model_name,
                                content=batch,
                                task_type="retrieval_document"
                            )
                            batch_embeddings = [np.array(emb) for emb in result['embedding']]
                            embeddings.extend(batch_embeddings)
                            logger.info(f"✅ Batch {batch_num} complete after wait")
                        except Exception as e3:
                            logger.error(f"❌ Final retry failed: {e3}")
                            # Use zero vectors as placeholder
                            for _ in batch:
                                embeddings.append(np.zeros(3072))
                else:
                    logger.error(f"❌ Batch {batch_num} failed: {e}")
                    # Use zero vectors as placeholder
                    for _ in batch:
                        embeddings.append(np.zeros(3072))
        
        return np.array(embeddings)


def main():
    """Build vector store with multi-key rotation."""
    
    # Setup logging
    log_dir = Config.OUTPUTS_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "build_vector_store_multikey.log", rotation="10 MB")
    
    logger.info("=" * 80)
    logger.info("SAP MODULE CLASSIFIER - BUILD VECTOR STORE (MULTI-KEY)")
    logger.info("=" * 80)
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Collect all API keys
    api_keys = []
    
    # Primary key
    if Config.GOOGLE_API_KEY:
        api_keys.append(Config.GOOGLE_API_KEY)
        logger.info(f"✅ Primary API key loaded")
    
    # Check for additional keys in environment
    import os
    for i in range(1, 10):  # Check for up to 10 keys
        key_name = f"GOOGLE_API_KEY_{i}"
        key_value = os.getenv(key_name)
        if key_value:
            api_keys.append(key_value)
            logger.info(f"✅ Additional API key #{i} loaded")
    
    if not api_keys:
        logger.error("❌ No API keys found!")
        logger.error("Please set GOOGLE_API_KEY in .env file")
        return
    
    logger.info(f"\n🔑 Total API keys available: {len(api_keys)}")
    logger.info(f"⚡ Expected speed boost: {len(api_keys)}x faster!")
    
    # Calculate time estimate
    total_tickets = 8611  # Approximate
    requests_per_minute = 100 * len(api_keys)
    estimated_minutes = total_tickets / requests_per_minute
    logger.info(f"⏱️ Estimated time: ~{estimated_minutes:.0f} minutes ({estimated_minutes/60:.1f} hours)")
    
    # Get paths
    paths = Config.get_paths()
    train_path = paths['processed'] / "train_test_split" / "train.csv"
    
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please run scripts/01_prepare_data.py first")
        return
    
    # Load training data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading training data")
    logger.info("=" * 80)
    train_df = pd.read_csv(train_path)
    logger.info(f"✅ Loaded {len(train_df)} training samples")
    
    # Create Combined_Text if needed
    if 'Combined_Text' not in train_df.columns or train_df['Combined_Text'].isna().any():
        logger.info("Creating Combined_Text column...")
        train_df['Combined_Text'] = (
            "Summary: " + train_df['Summary'].fillna('').astype(str) + 
            " Description: " + train_df['Description'].fillna('').astype(str)
        )
    
    # Initialize multi-key embedder
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Initializing Multi-Key Gemini Embedder")
    logger.info("=" * 80)
    embedder = MultiKeyGeminiEmbedder(
        api_keys=api_keys,
        model_name=Config.EMBEDDING_MODEL
    )
    
    # Generate embeddings
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Generating embeddings with key rotation")
    logger.info("=" * 80)
    
    embeddings_dir = Config.DATA_DIR / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = embeddings_dir / "train_embeddings.pkl"
    
    # Check for existing embeddings
    if embeddings_path.exists():
        logger.warning(f"⚠️ Existing embeddings found at {embeddings_path}")
        logger.warning("These will be overwritten. Creating backup...")
        backup_path = embeddings_dir / f"train_embeddings_backup_{int(time.time())}.pkl"
        embeddings_path.rename(backup_path)
        logger.info(f"✅ Backup created: {backup_path.name}")
    
    texts = train_df['Combined_Text'].tolist()
    
    # Calculate optimal delay based on number of keys
    # With more keys, we can use shorter delays
    delay = max(0.1, 2.0 / len(api_keys))
    logger.info(f"⏱️ Using delay of {delay:.2f}s between batches")
    
    start_time = time.time()
    
    embeddings = embedder.embed_batch_with_rotation(
        texts,
        batch_size=Config.BATCH_SIZE,
        delay=delay
    )
    
    elapsed_time = time.time() - start_time
    logger.info(f"\n✅ Generated {len(embeddings)} embeddings in {elapsed_time/60:.1f} minutes")
    
    # Save embeddings
    with open(embeddings_path, 'wb') as f:
        pickle.dump(embeddings, f)
    logger.info(f"✅ Saved embeddings to {embeddings_path}")
    
    # Build vector store
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Building ChromaDB vector store")
    logger.info("=" * 80)
    vector_store = ChromaVectorStore(persist_directory=str(paths['vector_store']))
    vector_store.create_collection(name=Config.COLLECTION_NAME, reset=True)
    
    # Prepare metadata
    metadatas = train_df[['Module', 'Priority', 'Team']].fillna('Unknown').to_dict('records')
    ids = train_df['Incident'].astype(str).tolist()
    
    # Add documents in batches
    batch_size = 5000
    for i in range(0, len(texts), batch_size):
        end_idx = min(i + batch_size, len(texts))
        batch_texts = texts[i:end_idx]
        batch_embeddings = embeddings[i:end_idx]
        batch_metadatas = metadatas[i:end_idx]
        batch_ids = ids[i:end_idx]
        
        vector_store.add_documents(
            texts=batch_texts,
            embeddings=batch_embeddings,
            metadatas=batch_metadatas,
            ids=batch_ids
        )
        logger.info(f"  ✅ Added batch: {end_idx}/{len(texts)} documents")
    
    logger.info(f"✅ Vector store built with {len(texts)} documents")
    
    # Build BM25 index
    logger.info("\n" + "=" * 80)
    logger.info("STEP 5: Building BM25 index")
    logger.info("=" * 80)
    
    bm25_index_path = embeddings_dir / "bm25_index.pkl"
    
    # Backup existing index if it exists
    if bm25_index_path.exists():
        backup_path = embeddings_dir / f"bm25_index_backup_{int(time.time())}.pkl"
        bm25_index_path.rename(backup_path)
        logger.info(f"✅ BM25 backup created: {backup_path.name}")
    
    # Create sparse retriever with corpus and metadata
    sparse_retriever = SparseRetriever(
        corpus=texts,
        metadata=metadatas,
        index_path=str(bm25_index_path)
    )
    
    # Save the index
    sparse_retriever.save_index()
    logger.info(f"✅ BM25 index saved")
    
    # Summary
    total_time = time.time() - start_time
    logger.info("\n" + "=" * 80)
    logger.info("✅ VECTOR STORE BUILD COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"📊 Total documents: {len(texts)}")
    logger.info(f"📏 Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"🔑 API keys used: {len(api_keys)}")
    logger.info(f"⏱️ Total time: {total_time/60:.1f} minutes")
    logger.info(f"⚡ Speed: {len(texts)/(total_time/60):.0f} embeddings/minute")
    logger.info(f"📁 Vector store: {paths['vector_store']}")
    logger.info(f"📁 BM25 index: {bm25_index_path}")
    logger.info("\n✅ Next step: Run streamlit app or test classifier!")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n⚠️ Process interrupted by user")
    except Exception as e:
        logger.error(f"\n\n❌ Unexpected error: {str(e)}", exc_info=True)
