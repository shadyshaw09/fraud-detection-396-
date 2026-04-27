"""
Feature Engineering Module

Generates engineered features for fraud detection model:
  - balance_discrepancy: (oldbalanceOrg - amount) - newbalanceOrig
  - zero_balance_flag: 1 if newbalanceOrig == 0 else 0
  - amount_log: log1p(amount)
"""

import numpy as np
import pandas as pd


class FeatureEngineer:
    """Engineer domain-specific fraud detection features."""
    
    def __init__(self, df: pd.DataFrame):
        """
        Initialize FeatureEngineer.
        
        Args:
            df: DataFrame with raw transaction data
        """
        self.df = df.copy()
        self.features = None
        self.feature_names = [
            'balance_discrepancy',
            'zero_balance_flag', 
            'amount_log'
        ]
    
    def engineer_features(self) -> pd.DataFrame:
        """
        Engineer 3 fraud detection features.
        
        Features:
        1. balance_discrepancy: (oldbalanceOrg - amount) - newbalanceOrig
           Measures inconsistency in balance accounting
           
        2. zero_balance_flag: 1 if newbalanceOrig == 0 else 0
           Flags transactions that empty the origin account
           
        3. amount_log: log1p(amount)
           Log-transformed transaction amount (handles skewness)
        
        Returns:
            DataFrame with engineered features (n_samples × 3)
        """
        print("⚙️  Engineering features...")
        
        # Validate required columns
        required_cols = ['amount', 'oldbalanceOrg', 'newbalanceOrig']
        missing = [col for col in required_cols if col not in self.df.columns]
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        
        # Feature 1: Balance Discrepancy
        # Expected: oldbalanceOrg - amount == newbalanceOrig
        # Fraud often shows: (oldbalanceOrg - amount) != newbalanceOrig
        self.df['balance_discrepancy'] = (
            (self.df['oldbalanceOrg'] - self.df['amount']) - self.df['newbalanceOrig']
        )
        
        # Feature 2: Zero Balance Flag
        # Suspicious: account balance goes to exactly zero
        self.df['zero_balance_flag'] = (self.df['newbalanceOrig'] == 0).astype(int)
        
        # Feature 3: Amount Log
        # Log transformation to handle skewed amount distributions
        self.df['amount_log'] = np.log1p(self.df['amount'])
        
        # Create feature matrix (3 features only)
        self.features = self.df[self.feature_names].values
        
        print(f"✓ Engineered 3 features: {', '.join(self.feature_names)}")
        print(f"  Feature matrix shape: {self.features.shape}")
        print(f"  Stats (balance_discrepancy): mean={self.features[:, 0].mean():.2f}, "
              f"std={self.features[:, 0].std():.2f}")
        print(f"  Stats (zero_balance_flag): {np.sum(self.features[:, 1])} flags set")
        print(f"  Stats (amount_log): min={self.features[:, 2].min():.2f}, "
              f"max={self.features[:, 2].max():.2f}")
        
        return self.df[self.feature_names]
    
    def get_feature_matrix(self) -> np.ndarray:
        """
        Get the engineered feature matrix.
        
        Returns:
            numpy array of shape (n_samples, 3)
        """
        if self.features is None:
            raise RuntimeError("Features not yet engineered. Call engineer_features() first.")
        
        return self.features
    
    def get_feature_dataframe(self) -> pd.DataFrame:
        """
        Get engineered features as a DataFrame.
        
        Returns:
            DataFrame with feature columns
        """
        if self.features is None:
            raise RuntimeError("Features not yet engineered. Call engineer_features() first.")
        
        return self.df[self.feature_names].copy()
    
    def get_full_dataframe(self) -> pd.DataFrame:
        """
        Get full dataframe with original columns + engineered features.
        
        Returns:
            DataFrame with all columns
        """
        if self.features is None:
            raise RuntimeError("Features not yet engineered. Call engineer_features() first.")
        
        return self.df.copy()
