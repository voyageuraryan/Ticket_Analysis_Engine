"""
Combine all augmented modules into final dataset.
Run this after augmenting all modules separately.
"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

import pandas as pd
from loguru import logger

from src.config import Config
from src.data_preparation import DataSplitter


def main():
    """Combine augmented modules and create train/val/test splits."""
    
    # Setup logging
    log_dir = Config.OUTPUTS_DIR / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    logger.add(log_dir / "combine_augmented.log", rotation="10 MB")
    
    logger.info("=" * 80)
    logger.info("COMBINING AUGMENTED MODULES")
    logger.info("=" * 80)
    
    # Get paths
    paths = Config.get_paths()
    processed_dir = paths['processed']
    
    # Load cleaned data
    clean_path = processed_dir / "cleaned_tickets.csv"
    if not clean_path.exists():
        logger.error(f"Cleaned data not found: {clean_path}")
        return
    
    df_clean = pd.read_csv(clean_path)
    logger.info(f"Loaded {len(df_clean)} original cleaned records")
    
    # Add 'Augmented' column to original data
    df_clean['Augmented'] = 'Original'
    
    # Load all augmented modules
    augmented_files = {
        'HR_Payroll': processed_dir / "augmented_HR_Payroll.csv",
        'Procurement': processed_dir / "augmented_Procurement.csv",
        'Connections': processed_dir / "augmented_Connections.csv",
        'FICO': processed_dir / "augmented_FICO.csv",
        'ABAP': processed_dir / "augmented_ABAP.csv"
    }
    
    augmented_dfs = []
    
    logger.info("\n" + "=" * 80)
    logger.info("Loading augmented modules")
    logger.info("=" * 80)
    
    for module_name, file_path in augmented_files.items():
        if file_path.exists():
            df_module = pd.read_csv(file_path)
            augmented_dfs.append(df_module)
            logger.info(f"✅ {module_name}: {len(df_module)} records")
        else:
            logger.warning(f"⚠️  {module_name}: Not found (skipping)")
    
    # Combine all data
    if augmented_dfs:
        df_all_augmented = pd.concat(augmented_dfs, ignore_index=True)
        logger.info(f"\nTotal augmented records: {len(df_all_augmented)}")
    else:
        logger.warning("No augmented data found!")
        df_all_augmented = pd.DataFrame()
    
    # Combine original and augmented
    if len(df_all_augmented) > 0:
        df_final = pd.concat([df_clean, df_all_augmented], ignore_index=True)
    else:
        df_final = df_clean
    
    logger.info(f"Final dataset: {len(df_final)} records")
    
    # Save combined dataset
    augmented_path = processed_dir / "augmented_tickets.csv"
    df_final.to_csv(augmented_path, index=False)
    logger.info(f"Saved combined dataset to {augmented_path}")
    
    # Show distribution
    logger.info("\n" + "=" * 80)
    logger.info("Final module distribution")
    logger.info("=" * 80)
    
    for module, count in df_final['Module'].value_counts().items():
        target = Config.TARGET_COUNTS.get(module, 'N/A')
        logger.info(f"  {module}: {count} (target: {target})")
    
    # Split data
    logger.info("\n" + "=" * 80)
    logger.info("Splitting data into train/val/test")
    logger.info("=" * 80)
    
    splitter = DataSplitter(
        train_size=Config.TRAIN_SIZE,
        val_size=Config.VAL_SIZE,
        test_size=Config.TEST_SIZE,
        random_seed=Config.RANDOM_SEED
    )
    
    train_df, val_df, test_df = splitter.split(df_final)
    
    split_dir = processed_dir / "train_test_split"
    split_dir.mkdir(parents=True, exist_ok=True)
    
    train_df.to_csv(split_dir / "train.csv", index=False)
    val_df.to_csv(split_dir / "val.csv", index=False)
    test_df.to_csv(split_dir / "test.csv", index=False)
    logger.info(f"Saved splits to {split_dir}")
    logger.info(f"  Train: {len(train_df)} records")
    logger.info(f"  Val: {len(val_df)} records")
    logger.info(f"  Test: {len(test_df)} records")
    
    # Summary
    logger.info("\n" + "=" * 80)
    logger.info("DATA PREPARATION COMPLETE")
    logger.info("=" * 80)
    logger.info(f"Original records: {len(df_clean)}")
    logger.info(f"Augmented records: {len(df_all_augmented)}")
    logger.info(f"Final records: {len(df_final)}")
    logger.info(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")
    
    logger.info("\n✅ Next step: Run scripts/02_build_vector_store.py")


if __name__ == "__main__":
    main()
