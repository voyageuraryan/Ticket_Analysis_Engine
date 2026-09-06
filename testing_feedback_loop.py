"""
Testing Feedback Loop
=====================
This script processes validated classification results and adds corrected tickets
to the training data, then rebuilds the vector store so the RAG system learns from mistakes.

Workflow:
1. Load validated Excel file (with human corrections)
2. Extract rejected/corrected predictions
3. Add corrected tickets to training data
4. Rebuild vector store with updated data
5. Optionally re-test on the same tickets to verify improvement

Author: SAP Module Classifier System
Date: 2026-01-27
"""

import os
import sys
import logging
from pathlib import Path
from datetime import datetime
import pandas as pd
from typing import Dict, List, Optional, Tuple
from dotenv import load_dotenv
import pickle

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from src.config import Config
from src.embeddings.gemini_embedder import GeminiEmbedder
from src.vector_store.chroma_store import ChromaVectorStore
from src.retrieval.sparse_retriever import SparseRetriever

# ================================================================================
# LOGGING SETUP
# ================================================================================
log_dir = project_root / "outputs" / "logs"
log_dir.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_dir / "feedback_loop.log", encoding='utf-8'),
        logging.StreamHandler()
    ]
)

# Set console handler to use UTF-8 encoding for Windows
console_handler = logging.getLogger().handlers[1]
if hasattr(console_handler, 'stream') and hasattr(console_handler.stream, 'reconfigure'):
    try:
        console_handler.stream.reconfigure(encoding='utf-8')
    except:
        pass

# Set console to INFO level only
console = logging.getLogger().handlers[1]
console.setLevel(logging.INFO)

logger = logging.getLogger(__name__)

# ================================================================================
# CONFIGURATION
# ================================================================================
TESTING_RESULTS_DIR = project_root / "outputs" / "testing_results"
FEEDBACK_DATA_DIR = project_root / "data" / "feedback"
FEEDBACK_DATA_DIR.mkdir(parents=True, exist_ok=True)

# ================================================================================
# HELPER FUNCTIONS
# ================================================================================

def find_latest_validated_file() -> Optional[Path]:
    """Find the most recently modified classified tickets file."""
    try:
        excel_files = list(TESTING_RESULTS_DIR.glob("classified_tickets_*.xlsx"))
        if not excel_files:
            logger.error("No classified ticket files found in outputs/testing_results/")
            return None
        
        # Sort by modification time, most recent first
        latest_file = max(excel_files, key=lambda p: p.stat().st_mtime)
        logger.info(f"Found latest classified file: {latest_file.name}")
        return latest_file
    except Exception as e:
        logger.error(f"Error finding classified file: {str(e)}")
        return None


def load_validated_data(file_path: Path) -> Optional[pd.DataFrame]:
    """Load the validated classification results."""
    try:
        df = pd.read_excel(file_path)
        logger.info(f"Loaded {len(df)} tickets from {file_path.name}")
        return df
    except PermissionError:
        logger.error(f"Permission denied: Cannot read {file_path.name}")
        logger.error("Please close the Excel file if it's open and try again.")
        return None
    except Exception as e:
        logger.error(f"Error loading Excel file: {str(e)}")
        return None


def extract_corrections(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Extract corrected, approved, and pending tickets.
    
    Returns:
        Tuple of (corrected_df, approved_df, pending_df)
    """
    logger.info("=" * 80)
    logger.info("Analyzing validation results")
    logger.info("=" * 80)
    
    # Ensure Validated_Module is string type
    df['Validated_Module'] = df['Validated_Module'].fillna('').astype(str)
    df['Validation_Status'] = df['Validation_Status'].fillna('Pending').astype(str)
    
    # Corrected: Where human provided a different module than predicted
    corrected = df[
        (df['Validated_Module'].str.strip() != '') & 
        (df['Validated_Module'] != df['Predicted_Module'])
    ].copy()
    
    # Approved: Where prediction was correct (explicitly marked or validated module matches)
    approved = df[
        (df['Validation_Status'].str.lower().str.contains('approve|correct|ok', na=False)) |
        ((df['Validated_Module'].str.strip() != '') & (df['Validated_Module'] == df['Predicted_Module']))
    ].copy()
    
    # Pending: Not yet validated
    pending = df[
        (df['Validated_Module'].str.strip() == '') & 
        (df['Validation_Status'].str.lower() == 'pending')
    ].copy()
    
    logger.info(f"[OK] Corrected (wrong predictions): {len(corrected)} tickets")
    logger.info(f"[OK] Approved (correct predictions): {len(approved)} tickets")
    logger.info(f"[!] Pending (not yet validated): {len(pending)} tickets")
    
    if len(corrected) > 0:
        logger.info("\nCorrected tickets:")
        for _, row in corrected.iterrows():
            logger.info(f"  {row['Incident_Number']}: {row['Predicted_Module']} -> {row['Validated_Module']}")
    
    return corrected, approved, pending


def prepare_feedback_data(corrected_df: pd.DataFrame) -> pd.DataFrame:
    """
    Prepare corrected tickets in the format needed for training data.
    
    Expected columns: Incident, Summary, Description, Module, Priority, Team, Combined_Text
    """
    if len(corrected_df) == 0:
        return pd.DataFrame()
    
    feedback_df = pd.DataFrame({
        'Incident': corrected_df['Incident_Number'],
        'Summary': corrected_df['Summary'],
        'Description': corrected_df['Description'],
        'Module': corrected_df['Validated_Module'],  # Use the CORRECTED module
        'Priority': 'Medium',  # Default values
        'Team': corrected_df.get('Owner', 'Unknown'),
        'Source': 'feedback_loop',
        'Feedback_Date': datetime.now().strftime('%Y-%m-%d'),
        'Original_Prediction': corrected_df['Predicted_Module'],
        'Confidence': corrected_df['Confidence']
    })
    
    # Create Combined_Text
    feedback_df['Combined_Text'] = (
        "Summary: " + feedback_df['Summary'].fillna('').astype(str) + 
        " Description: " + feedback_df['Description'].fillna('').astype(str)
    )
    
    return feedback_df


def save_feedback_data(feedback_df: pd.DataFrame) -> Path:
    """Save feedback data to CSV."""
    timestamp = datetime.now().strftime("%Y_%m_%d_%H%M%S")
    feedback_path = FEEDBACK_DATA_DIR / f"feedback_corrections_{timestamp}.csv"
    
    feedback_df.to_csv(feedback_path, index=False)
    logger.info(f"[OK] Saved {len(feedback_df)} corrected tickets to: {feedback_path}")
    
    return feedback_path


def append_to_training_data(feedback_df: pd.DataFrame) -> bool:
    """
    Append corrected tickets to the training data.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        paths = Config.get_paths()
        train_path = paths['processed'] / "train_test_split" / "train.csv"
        
        if not train_path.exists():
            logger.error(f"Training data not found: {train_path}")
            return False
        
        # Load existing training data
        train_df = pd.read_csv(train_path)
        original_count = len(train_df)
        logger.info(f"Loaded existing training data: {original_count} tickets")
        
        # Ensure feedback data has the same columns as training data
        required_columns = ['Incident', 'Summary', 'Description', 'Module', 'Combined_Text']
        feedback_subset = feedback_df[required_columns].copy()
        
        # Add optional columns if they exist in training data
        for col in ['Priority', 'Team']:
            if col in train_df.columns:
                feedback_subset[col] = feedback_df.get(col, 'Unknown')
        
        # Append feedback data
        train_df_updated = pd.concat([train_df, feedback_subset], ignore_index=True)
        
        # Remove duplicates based on Incident number (keep the latest - feedback version)
        train_df_updated = train_df_updated.drop_duplicates(subset=['Incident'], keep='last')
        
        new_count = len(train_df_updated)
        added_count = new_count - original_count
        
        # Create backup of original training data
        backup_path = train_path.parent / f"train_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        train_df.to_csv(backup_path, index=False)
        logger.info(f"[OK] Backed up original training data to: {backup_path.name}")
        
        # Save updated training data
        train_df_updated.to_csv(train_path, index=False)
        logger.info(f"[OK] Updated training data: {original_count} -> {new_count} tickets (+{added_count} new)")
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to update training data: {str(e)}", exc_info=True)
        return False


def rebuild_vector_store() -> bool:
    """
    Rebuild the vector store with updated training data.
    
    Returns:
        True if successful, False otherwise
    """
    try:
        logger.info("=" * 80)
        logger.info("Rebuilding vector store with updated training data")
        logger.info("=" * 80)
        
        paths = Config.get_paths()
        train_path = paths['processed'] / "train_test_split" / "train.csv"
        
        # Load updated training data
        train_df = pd.read_csv(train_path)
        logger.info(f"Loaded {len(train_df)} training samples")
        
        # Ensure Combined_Text exists
        if 'Combined_Text' not in train_df.columns or train_df['Combined_Text'].isna().any():
            logger.info("Creating Combined_Text column...")
            train_df['Combined_Text'] = (
                "Summary: " + train_df['Summary'].fillna('').astype(str) + 
                " Description: " + train_df['Description'].fillna('').astype(str)
            )
        
        # Initialize embedder
        logger.info("Initializing Gemini embedder...")
        embedder = GeminiEmbedder(
            api_key=Config.GOOGLE_API_KEY,
            model_name=Config.EMBEDDING_MODEL
        )
        
        # Generate embeddings for ALL data (including new feedback)
        logger.info("Generating embeddings for updated dataset...")
        logger.info(f"This will take approximately {(len(train_df) // Config.BATCH_SIZE + 1) * Config.API_DELAY_SECONDS / 60:.1f} minutes")
        
        texts = train_df['Combined_Text'].tolist()
        embeddings = embedder.embed_batch(
            texts,
            batch_size=Config.BATCH_SIZE,
            delay=Config.API_DELAY_SECONDS
        )
        
        logger.info(f"Generated {len(embeddings)} embeddings")
        
        # Save embeddings
        embeddings_dir = Config.DATA_DIR / "embeddings"
        embeddings_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup old embeddings
        old_embeddings_path = embeddings_dir / "train_embeddings.pkl"
        if old_embeddings_path.exists():
            backup_path = embeddings_dir / f"train_embeddings_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            old_embeddings_path.rename(backup_path)
            logger.info(f"[OK] Backed up old embeddings to: {backup_path.name}")
        
        # Save new embeddings
        with open(old_embeddings_path, 'wb') as f:
            pickle.dump(embeddings, f)
        logger.info(f"[OK] Saved new embeddings to {old_embeddings_path}")
        
        # Rebuild ChromaDB vector store
        logger.info("Rebuilding ChromaDB vector store...")
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
            logger.info(f"  Added batch {i//batch_size + 1}: {end_idx}/{len(texts)} documents")
        
        logger.info(f"[OK] Vector store rebuilt with {len(texts)} documents")
        
        # Rebuild BM25 index
        logger.info("Rebuilding BM25 index...")
        from src.retrieval.sparse_retriever import SparseRetriever
        
        bm25_index_path = embeddings_dir / "bm25_index.pkl"
        
        # Backup old BM25 index
        if bm25_index_path.exists():
            backup_path = embeddings_dir / f"bm25_index_backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pkl"
            bm25_index_path.rename(backup_path)
            logger.info(f"[OK] Backed up old BM25 index to: {backup_path.name}")
        
        # Create new BM25 index
        sparse_retriever = SparseRetriever()
        sparse_retriever.build_index(texts, ids, metadatas)
        sparse_retriever.save_index(str(bm25_index_path))
        
        logger.info(f"[OK] BM25 index rebuilt and saved to {bm25_index_path}")
        logger.info("=" * 80)
        logger.info("[OK] Vector store rebuild complete!")
        logger.info("=" * 80)
        
        return True
        
    except Exception as e:
        logger.error(f"[ERROR] Failed to rebuild vector store: {str(e)}", exc_info=True)
        return False


def print_summary(corrected_df: pd.DataFrame, approved_df: pd.DataFrame, pending_df: pd.DataFrame):
    """Print feedback loop summary."""
    logger.info("=" * 80)
    logger.info("FEEDBACK LOOP SUMMARY")
    logger.info("=" * 80)
    
    total = len(corrected_df) + len(approved_df) + len(pending_df)
    
    logger.info(f"Total tickets processed: {total}")
    logger.info(f"  [OK] Approved (correct): {len(approved_df)} ({len(approved_df)/total*100:.1f}%)")
    logger.info(f"  [!] Corrected (wrong): {len(corrected_df)} ({len(corrected_df)/total*100:.1f}%)")
    logger.info(f"  [!] Pending validation: {len(pending_df)} ({len(pending_df)/total*100:.1f}%)")
    
    if len(corrected_df) > 0:
        logger.info("\nModule-wise corrections:")
        correction_stats = corrected_df.groupby(['Predicted_Module', 'Validated_Module']).size()
        for (pred, actual), count in correction_stats.items():
            logger.info(f"  {pred} -> {actual}: {count} tickets")
    
    logger.info("=" * 80)


# ================================================================================
# MAIN EXECUTION
# ================================================================================

def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("TESTING FEEDBACK LOOP")
    logger.info("=" * 80)
    logger.info(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info("=" * 80)
    
    # Load environment variables
    load_dotenv()
    
    # Step 1: Find the latest validated file
    logger.info("\nSTEP 1: Finding latest validated classification file")
    logger.info("=" * 80)
    validated_file = find_latest_validated_file()
    if not validated_file:
        logger.error("[ERROR] No classified file found.")
        logger.error("Please run testing_data_classifier.py first and validate the results.")
        return
    
    # Step 2: Load the validated data
    logger.info("\nSTEP 2: Loading validated data")
    logger.info("=" * 80)
    df = load_validated_data(validated_file)
    if df is None or len(df) == 0:
        logger.error("[ERROR] Failed to load data or no tickets found.")
        return
    
    # Step 3: Extract corrections
    logger.info("\nSTEP 3: Extracting corrections")
    logger.info("=" * 80)
    corrected_df, approved_df, pending_df = extract_corrections(df)
    
    if len(corrected_df) == 0:
        logger.info("\n[OK] No corrections found! All predictions were either correct or pending.")
        logger.info("No need to retrain. The model is performing well!")
        print_summary(corrected_df, approved_df, pending_df)
        return
    
    # Step 4: Prepare feedback data
    logger.info("\nSTEP 4: Preparing feedback data")
    logger.info("=" * 80)
    feedback_df = prepare_feedback_data(corrected_df)
    feedback_path = save_feedback_data(feedback_df)
    
    # Step 5: Add to training data
    logger.info("\nSTEP 5: Adding corrections to training data")
    logger.info("=" * 80)
    if not append_to_training_data(feedback_df):
        logger.error("[ERROR] Failed to update training data.")
        return
    
    # Step 6: Rebuild vector store
    logger.info("\nSTEP 6: Rebuilding vector store")
    logger.info("=" * 80)
    logger.info("[!] This will take several minutes...")
    if not rebuild_vector_store():
        logger.error("[ERROR] Failed to rebuild vector store.")
        return
    
    # Step 7: Print summary
    logger.info("\nSTEP 7: Summary")
    print_summary(corrected_df, approved_df, pending_df)
    
    # Final message
    logger.info("=" * 80)
    logger.info("[OK] FEEDBACK LOOP COMPLETE!")
    logger.info("=" * 80)
    logger.info(f"Corrected tickets: {len(corrected_df)}")
    logger.info(f"Feedback data saved: {feedback_path}")
    logger.info("\nThe RAG system has been updated with your corrections!")
    logger.info("You can now run testing_data_classifier.py again to verify improvements.")
    logger.info("=" * 80)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\n\n[!] Process interrupted by user")
    except Exception as e:
        logger.error(f"\n\n[ERROR] Unexpected error: {str(e)}", exc_info=True)
    finally:
        logger.info("\nScript execution completed")
