"""
Data Preprocessing Module for Fraud Detection

Handles data loading, cleaning, missing value treatment, encoding, and normalization.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from pathlib import Path
from typing import Tuple, Dict
import warnings

warnings.filterwarnings('ignore')


class DataPreprocessor:
    """
    Preprocessing pipeline for transaction data.
    
    This class handles:
    - Loading CSV data
    - Handling missing values
    - Encoding categorical features
    - Normalizing numerical features
    - Train/test data separation (if labeled)
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the preprocessor.
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.scaler = StandardScaler()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.categorical_features = []
        self.numerical_features = []
        self.feature_names = []
        self.is_fitted = False
        
    def load_data(self, filepath: str) -> pd.DataFrame:
        """
        Load transaction data from CSV file.
        
        Args:
            filepath: Path to CSV file
            
        Returns:
            DataFrame with loaded data
        """
        print(f"[PREPROCESSING] Loading data from {filepath}...")
        df = pd.read_csv(filepath)
        print(f"[PREPROCESSING] Loaded {len(df)} transactions with {len(df.columns)} columns")
        return df
    
    def handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle missing values in the dataset.
        
        Strategy:
        - Numerical features: fill with median
        - Categorical features: fill with mode
        - Drop rows if >30% missing
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with missing values handled
        """
        print("[PREPROCESSING] Handling missing values...")
        
        # Check for missing values
        missing_pct = (df.isnull().sum() / len(df)) * 100
        if missing_pct.any():
            print(f"[PREPROCESSING] Found missing values:\n{missing_pct[missing_pct > 0]}")
            
            # Drop columns with >30% missing
            cols_to_drop = missing_pct[missing_pct > 30].index.tolist()
            if cols_to_drop:
                print(f"[PREPROCESSING] Dropping columns with >30% missing: {cols_to_drop}")
                df = df.drop(columns=cols_to_drop)
            
            # Fill remaining missing values
            for col in df.columns:
                if df[col].isnull().any():
                    if df[col].dtype in ['int64', 'float64']:
                        df[col].fillna(df[col].median(), inplace=True)
                    else:
                        df[col].fillna(df[col].mode()[0] if not df[col].mode().empty else 'UNKNOWN', inplace=True)
        
        print("[PREPROCESSING] Missing values handled")
        return df
    
    def identify_feature_types(self, df: pd.DataFrame) -> Tuple[list, list]:
        """
        Identify categorical and numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            Tuple of (categorical_features, numerical_features)
        """
        # Exclude special columns
        exclude_cols = ['isFraud', 'Unnamed: 0', 'index']
        df_analysis = df.drop(columns=exclude_cols, errors='ignore')
        
        categorical = df_analysis.select_dtypes(include=['object']).columns.tolist()
        numerical = df_analysis.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        self.categorical_features = categorical
        self.numerical_features = numerical
        
        print(f"[PREPROCESSING] Identified {len(categorical)} categorical and {len(numerical)} numerical features")
        return categorical, numerical
    
    def encode_categorical_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.
        
        Args:
            df: Input DataFrame
            fit: If True, fit encoders; if False, use existing encoders
            
        Returns:
            DataFrame with encoded categorical features
        """
        df_encoded = df.copy()
        
        print(f"[PREPROCESSING] Encoding {len(self.categorical_features)} categorical features...")
        
        for col in self.categorical_features:
            if col not in df_encoded.columns:
                continue
                
            if fit:
                self.label_encoders[col] = LabelEncoder()
                df_encoded[col] = self.label_encoders[col].fit_transform(df_encoded[col].astype(str))
            else:
                df_encoded[col] = self.label_encoders[col].transform(df_encoded[col].astype(str))
        
        return df_encoded
    
    def normalize_features(self, df: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Normalize numerical features using StandardScaler.
        
        Args:
            df: Input DataFrame (with categorical features already encoded)
            fit: If True, fit scaler; if False, use existing scaler
            
        Returns:
            DataFrame with normalized numerical features
        """
        df_normalized = df.copy()
        
        print(f"[PREPROCESSING] Normalizing {len(self.numerical_features)} numerical features...")
        
        if fit:
            df_normalized[self.numerical_features] = self.scaler.fit_transform(
                df_normalized[self.numerical_features]
            )
            self.is_fitted = True
        else:
            df_normalized[self.numerical_features] = self.scaler.transform(
                df_normalized[self.numerical_features]
            )
        
        return df_normalized
    
    def fit_and_preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fit preprocessor and transform data (for training set).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = self.handle_missing_values(df)
        self.identify_feature_types(df)
        df = self.encode_categorical_features(df, fit=True)
        df = self.normalize_features(df, fit=True)
        
        self.feature_names = df.drop(columns=['isFraud'], errors='ignore').columns.tolist()
        print(f"[PREPROCESSING] Preprocessing complete. Features: {len(self.feature_names)}")
        return df
    
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using fitted preprocessor (for test/new data).
        
        Args:
            df: Input DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        if not self.is_fitted:
            raise ValueError("Preprocessor not fitted. Call fit_and_preprocess first.")
        
        df = self.handle_missing_values(df)
        df = self.encode_categorical_features(df, fit=False)
        df = self.normalize_features(df, fit=False)
        
        return df
    
    def get_feature_names(self) -> list:
        """Get names of all features (excluding isFraud and index)."""
        return self.feature_names


def create_synthetic_paysim_data(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic PaySim-like dataset if real data is unavailable.
    
    Features:
    - step: Time step (1-744, representing days)
    - type: Transaction type (CASH_OUT, PAYMENT, CASH_IN, TRANSFER, DEBIT)
    - amount: Transaction amount
    - nameOrig: Origin account name
    - oldbalanceOrg: Old balance of origin
    - newbalanceOrig: New balance of origin
    - nameDest: Destination account name
    - oldbalanceDest: Old balance of destination
    - newbalanceDest: New balance of destination
    - isFraud: Fraud label (used only for evaluation)
    
    Args:
        n_samples: Number of transactions to generate
        random_state: Random seed
        
    Returns:
        DataFrame with synthetic transaction data
    """
    print(f"[PREPROCESSING] Generating synthetic PaySim data with {n_samples} transactions...")
    
    np.random.seed(random_state)
    
    # Generate basic features
    data = {
        'step': np.random.randint(1, 745, n_samples),
        'type': np.random.choice(['CASH_OUT', 'PAYMENT', 'CASH_IN', 'TRANSFER', 'DEBIT'], n_samples),
        'amount': np.random.exponential(scale=5000, size=n_samples).clip(0, 1000000),
        'nameOrig': [f'Customer_{i % 5000}' for i in range(n_samples)],
        'oldbalanceOrg': np.random.exponential(scale=10000, size=n_samples).clip(0),
        'newbalanceOrig': np.zeros(n_samples),
        'nameDest': [f'Merchant_{i % 2000}' for i in range(n_samples)],
        'oldbalanceDest': np.random.exponential(scale=5000, size=n_samples).clip(0),
        'newbalanceDest': np.zeros(n_samples),
    }
    
    # Calculate new balances
    data['newbalanceOrig'] = data['oldbalanceOrg'] - data['amount']
    data['newbalanceDest'] = data['oldbalanceDest'] + data['amount']
    
    # Clip negative balances to 0 (realistic behavior)
    data['newbalanceOrig'] = np.maximum(data['newbalanceOrig'], 0)
    
    # Generate fraud labels (1-2% fraud rate, realistic)
    n_fraud = max(1, int(n_samples * 0.015))
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    data['isFraud'] = 0
    data['isFraud'][fraud_indices] = 1
    
    # Inject anomalies in fraudulent transactions
    for idx in fraud_indices:
        anomaly_type = np.random.choice(['high_amount', 'zero_balance', 'inconsistent_balance'])
        
        if anomaly_type == 'high_amount':
            data['amount'][idx] *= np.random.uniform(5, 15)  # Unusually high amount
        elif anomaly_type == 'zero_balance':
            data['newbalanceOrig'][idx] = 0  # Account drained
        else:
            data['newbalanceDest'][idx] -= data['amount'][idx] * 2  # Inconsistent accounting
    
    df = pd.DataFrame(data)
    print(f"[PREPROCESSING] Generated {len(df)} transactions ({(df['isFraud'].sum() / len(df) * 100):.2f}% fraud)")
    
    return df
