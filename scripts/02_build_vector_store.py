"""
Build vector store with embeddings.
Run this after data preparation.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
import pickle

from src.config import Config
from src.embeddings import GeminiEmbedder
from src.vector_store import ChromaVectorStore
from src.retrieval import SparseRetriever


def main():
    """Build vector store and BM25 index."""
    
    # Setup logging
    log_dir = Config.OUTPUTS_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "build_vector_store.log", rotation="10 MB")
    
    logger.info("=" * 80)
    logger.info("SAP MODULE CLASSIFIER - BUILD VECTOR STORE")
    logger.info("=" * 80)
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Get paths
    paths = Config.get_paths()
    train_path = paths['processed'] / "train_test_split" / "train.csv"
    
    if not train_path.exists():
        logger.error(f"Training data not found: {train_path}")
        logger.error("Please run scripts/01_prepare_data.py first")
        return
    
    # Load training data
    logger.info("\nLoading training data...")
    train_df = pd.read_csv(train_path)
    logger.info(f"Loaded {len(train_df)} training samples")
    
    # Create Combined_Text if it doesn't exist
    if 'Combined_Text' not in train_df.columns or train_df['Combined_Text'].isna().any():
        logger.info("Creating Combined_Text column...")
        train_df['Combined_Text'] = (
            "Summary: " + train_df['Summary'].fillna('').astype(str) + 
            " Description: " + train_df['Description'].fillna('').astype(str)
        )
        logger.info(f"Created Combined_Text for {len(train_df)} records")
    
    # Initialize embedder
    logger.info("\nInitializing Gemini embedder...")
    embedder = GeminiEmbedder(
        api_key=Config.GOOGLE_API_KEY,
        model_name=Config.EMBEDDING_MODEL
    )
    
    # Prepare embeddings directory
    embeddings_dir = Config.DATA_DIR / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    embeddings_path = embeddings_dir / "train_embeddings.pkl"
    checkpoint_path = embeddings_dir / "embeddings_checkpoint.pkl"
    
    # Check for existing embeddings
    if embeddings_path.exists():
        logger.info(f"\n✅ Found existing embeddings at {embeddings_path}")
        logger.info("Loading existing embeddings...")
        with open(embeddings_path, 'rb') as f:
            embeddings = pickle.load(f)
        logger.info(f"Loaded {len(embeddings)} embeddings from cache")
        texts = train_df['Combined_Text'].tolist()
    else:
        # Generate embeddings
        logger.info("\nGenerating embeddings...")
        logger.info("This will use BATCH API (much faster!)")
        logger.info(f"Estimated time: ~{(len(train_df) // Config.BATCH_SIZE + 1) * Config.API_DELAY_SECONDS / 60:.1f} minutes")
        
        texts = train_df['Combined_Text'].tolist()
        embeddings = embedder.embed_batch(
            texts,
            batch_size=Config.BATCH_SIZE,
            delay=Config.API_DELAY_SECONDS
        )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Save embeddings immediately
        with open(embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"✅ Saved embeddings to {embeddings_path}")
    
    # Build vector store
    logger.info("\nBuilding ChromaDB vector store...")
    vector_store = ChromaVectorStore(persist_directory=str(paths['vector_store']))
    vector_store.create_collection(name=Config.COLLECTION_NAME, reset=True)
    
    # Prepare metadata
    metadatas = train_df[['Module', 'Priority', 'Team']].fillna('Unknown').to_dict('records')
    ids = train_df['Incident'].astype(str).tolist()
    
    # Add documents
    vector_store.add_documents(
        texts=texts,
        embeddings=embeddings,
        metadatas=metadatas,
        ids=ids
    )
    
    stats = vector_store.get_collection_stats()
    logger.info(f"Vector store built with {stats['total_documents']} documents")
    
    # Build BM25 index
    logger.info("\nBuilding BM25 index...")
    sparse_retriever = SparseRetriever(
        corpus=texts,
        metadata=metadatas,
        index_path=str(embeddings_dir / "bm25_index.pkl")
    )
    sparse_retriever.save_index()
    logger.info("BM25 index saved")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("VECTOR STORE BUILD COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Total documents: {len(texts)}")
    logger.info(f"Embedding dimension: {embeddings.shape[1]}")
    logger.info(f"Vector store location: {paths['vector_store']}")
    logger.info(f"BM25 index location: {embeddings_dir / 'bm25_index.pkl'}")
    
    logger.info("\n✅ Next step: Run scripts/03_create_raft_dataset.py (optional)")
    logger.info("   Or go directly to app/streamlit_app.py to test the classifier")


if __name__ == "__main__":
    main()
