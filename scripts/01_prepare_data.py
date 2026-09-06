"""
Complete data preparation pipeline.
Run this first to clean, augment, and split the data.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger
import os

from src.config import Config
from src.data_preparation import TicketCleaner, TicketAugmenter, DataSplitter


def main():
    """Run complete data preparation pipeline."""
    
    # Setup logging
    log_dir = Config.OUTPUTS_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "data_preparation.log", rotation="10 MB")
    
    logger.info("=" * 80)
    logger.info("SAP MODULE CLASSIFIER - DATA PREPARATION")
    logger.info("=" * 80)
    
    # Validate config
    try:
        Config.validate()
    except ValueError as e:
        logger.error(f"Configuration error: {e}")
        return
    
    # Get paths
    paths = Config.get_paths()
    raw_data_path = paths['raw_data']
    processed_dir = paths['processed']
    processed_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Load raw data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 1: Loading raw data")
    logger.info("=" * 80)
    
    if not raw_data_path.exists():
        logger.error(f"Raw data file not found: {raw_data_path}")
        logger.error("Please place Historic_Data_CSV.xlsx in data/raw/ directory")
        return
    
    df_raw = pd.read_excel(raw_data_path)
    logger.info(f"Loaded {len(df_raw)} records from {raw_data_path}")
    logger.info(f"Columns: {df_raw.columns.tolist()}")
    
    # Step 2: Clean data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 2: Cleaning data")
    logger.info("=" * 80)
    
    cleaner = TicketCleaner()
    df_clean = cleaner.clean_dataset(df_raw)
    
    clean_path = processed_dir / "cleaned_tickets.csv"
    df_clean.to_csv(clean_path, index=False)
    logger.info(f"Saved cleaned data to {clean_path}")
    
    # Step 3: Augment data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 3: Augmenting data")
    logger.info("=" * 80)
    
    # Initialize augmenter based on LLM provider
    if Config.LLM_PROVIDER == "groq":
        logger.info(f"Using Groq API with model: {Config.GROQ_MODEL}")
        logger.info("⚡ This will be SUPER FAST (30-45 minutes for 6,000 tickets)!")
        augmenter = TicketAugmenter(
            llm_provider="groq",
            groq_api_key=Config.GROQ_API_KEY,
            groq_model=Config.GROQ_MODEL,
            target_counts=Config.TARGET_COUNTS
        )
    elif Config.LLM_PROVIDER == "ollama":
        logger.info(f"Using Ollama with model: {Config.LOCAL_LLM_MODEL}")
        augmenter = TicketAugmenter(
            llm_provider="ollama",
            use_local_llm=True,
            local_model=Config.LOCAL_LLM_MODEL,
            target_counts=Config.TARGET_COUNTS
        )
    else:  # gemini
        logger.info(f"Using Gemini API with model: {Config.LLM_MODEL}")
        logger.info("This may take a while due to API rate limiting...")
        augmenter = TicketAugmenter(
            llm_provider="gemini",
            api_key=Config.GOOGLE_API_KEY,
            model_name=Config.LLM_MODEL,
            target_counts=Config.TARGET_COUNTS
        )
    
    df_augmented = augmenter.augment_dataset(
        df_clean,
        delay_seconds=Config.API_DELAY_SECONDS
    )
    
    augmented_path = processed_dir / "augmented_tickets.csv"
    df_augmented.to_csv(augmented_path, index=False)
    logger.info(f"Saved augmented data to {augmented_path}")
    
    # Step 4: Split data
    logger.info("\n" + "=" * 80)
    logger.info("STEP 4: Splitting data")
    logger.info("=" * 80)
    
    splitter = DataSplitter(
        train_size=Config.TRAIN_SIZE,
        val_size=Config.VAL_SIZE,
        test_size=Config.TEST_SIZE,
        random_seed=Config.RANDOM_SEED
    )
    
    train_df, val_df, test_df = splitter.split(df_augmented)
    
    split_dir = processed_dir / "train_test_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    logger.info(f"Saved splits to {split_dir}")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Raw records: {len(df_raw)}")
    logger.info(f"Cleaned records: {len(df_clean)}")
    logger.info(f"Augmented records: {len(df_augmented)}")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    logger.info("\nFinal module distribution:")
    for module, count in df_augmented['Module'].value_counts().items():
        logger.info(f"  {module}: {count}")
    
    logger.info("\n✅ Next step: Run scripts/02_build_vector_store.py")


if __name__ == "__main__":
    main()
