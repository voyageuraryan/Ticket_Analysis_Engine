"""
Train/validation/test splitting with stratification.
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from loguru import logger


class DataSplitter:
    """Split data into train/val/test sets."""
    
    def __init__(self, train_size: float = 0.7, val_size: float = 0.15, test_size: float = 0.15, random_seed: int = 42):
        """
        Initialize splitter.
        
        Args:
            train_size: Proportion for training
            val_size: Proportion for validation
            test_size: Proportion for testing
            random_seed: Random seed for reproducibility
        """
        assert abs(train_size + val_size + test_size - 1.0) < 1e-6, "Sizes must sum to 1.0"
        
        self.train_size = train_size
        self.val_size = val_size
        self.test_size = test_size
        self.random_seed = random_seed
    
    def split(self, df: pd.DataFrame, stratify_column: str = 'Module') -> tuple:
        """
        Split dataset with stratification.
        
        Args:
            df: DataFrame to split
            stratify_column: Column to stratify on
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        logger.info(f"Splitting {len(df)} records")
        logger.info(f"Train: {self.train_size:.1%}, Val: {self.val_size:.1%}, Test: {self.test_size:.1%}")
        
        # First split: train vs (val + test)
        train_df, temp_df = train_test_split(
            df,
            train_size=self.train_size,
            stratify=df[stratify_column],
            random_state=self.random_seed
        )
        
        # Second split: val vs test
        val_ratio = self.val_size / (self.val_size + self.test_size)
        val_df, test_df = train_test_split(
            temp_df,
            train_size=val_ratio,
            stratify=temp_df[stratify_column],
            random_state=self.random_seed
        )
        
        logger.info(f"Train: {len(train_df)} records")
        logger.info(f"Val: {len(val_df)} records")
        logger.info(f"Test: {len(test_df)} records")
        
        # Verify stratification
        logger.info("\nClass distribution:")
        for split_name, split_df in [("Train", train_df), ("Val", val_df), ("Test", test_df)]:
            dist = split_df[stratify_column].value_counts(normalize=True)
            logger.info(f"\n{split_name}:")
            for module, pct in dist.items():
                logger.info(f"  {module}: {pct:.1%}")
        
        return train_df, val_df, test_df
