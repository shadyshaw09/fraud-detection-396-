"""
Data Loading and Filtering Module

Handles loading fraud detection dataset, filtering transaction types,
and preparing data for feature engineering.
"""

import pandas as pd
import sys
from pathlib import Path


class DataLoader:
    """Load and filter PaySim fraud detection dataset."""
    
    def __init__(self, csv_path: str, n_rows: int = 50000):
        """
        Initialize DataLoader.
        
        Args:
            csv_path: Path to paysim.csv file
            n_rows: Number of rows to load (default 50,000)
        """
        self.csv_path = csv_path
        self.n_rows = n_rows
        self.df = None
        self.filtered_df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load first n_rows from CSV file.
        
        Returns:
            DataFrame with raw data
            
        Raises:
            FileNotFoundError: If CSV file doesn't exist
        """
        print(f"📥 Loading {self.n_rows:,} rows from {self.csv_path}...")
        
        if not Path(self.csv_path).exists():
            raise FileNotFoundError(
                f"Dataset not found at {self.csv_path}\n"
                f"Please place paysim.csv in the data/ directory."
            )
        
        self.df = pd.read_csv(self.csv_path, nrows=self.n_rows)
        print(f"✓ Loaded {len(self.df):,} transactions")
        print(f"  Columns: {', '.join(self.df.columns.tolist())}")
        
        return self.df
    
    def filter_transaction_types(self, types: list = None) -> pd.DataFrame:
        """
        Filter to only specified transaction types.
        
        Args:
            types: List of transaction types to keep (default: TRANSFER, CASH_OUT)
        
        Returns:
            Filtered DataFrame
        """
        if types is None:
            types = ["TRANSFER", "CASH_OUT"]
        
        print(f"🔍 Filtering to transaction types: {', '.join(types)}...")
        
        if 'type' not in self.df.columns:
            raise ValueError("Column 'type' not found in dataset")
        
        self.filtered_df = self.df[self.df['type'].isin(types)].copy()
        
        print(f"✓ Filtered to {len(self.filtered_df):,} transactions ({len(self.filtered_df)/len(self.df)*100:.1f}%)")
        
        return self.filtered_df
    
    def get_processed_data(self) -> pd.DataFrame:
        """
        Get the final processed dataset (filtered and ready for feature engineering).
        
        Returns:
            Processed DataFrame
        """
        if self.filtered_df is None:
            raise RuntimeError("Data not yet processed. Call load_data() and filter_transaction_types() first.")
        
        return self.filtered_df
    
    def load_and_filter(self, types: list = None) -> pd.DataFrame:
        """
        Complete pipeline: load and filter in one call.
        
        Args:
            types: List of transaction types to keep
        
        Returns:
            Filtered DataFrame
        """
        self.load_data()
        self.filter_transaction_types(types)
        
        return self.get_processed_data()
