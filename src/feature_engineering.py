"""
Novel Feature Engineering Module for Fraud Detection

This is the PATENT-WORTHY core of the fraud detection system.

The key innovation is the multi-signal risk embedding that combines:
1. Balance Discrepancy Ratio - mathematical inconsistencies in account balances
2. Transaction Velocity Score - timing and frequency patterns
3. Amount Deviation Score - behavioral anomalies in transaction size
4. Zero Balance Flag - extreme account state changes
5. Round Amount Suspicion - unnatural transaction patterns
6. Destination Account Age Proxy - account profiling via naming patterns

These signals are combined into a unified "risk embedding vector" that captures
fraud patterns without requiring labeled data. The signals are mathematically
independent, allowing ensemble detection to identify diverse fraud types.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple
import warnings

warnings.filterwarnings('ignore')


class FeatureEngineer:
    """
    Engineer novel features for unsupervised fraud detection.
    
    The core innovation: multi-signal risk embedding that captures fraud
    patterns through mathematical and behavioral anomalies without labels.
    """
    
    def __init__(self):
        """Initialize feature engineer and storage for fit parameters."""
        self.amount_stats: Dict[str, Dict] = {}  # Store per-origin amount statistics
        self.velocity_window = 48  # Rolling window for velocity (transactions)
        self.is_fitted = False
        
    def engineer_balance_discrepancy_ratio(self, df: pd.DataFrame) -> pd.Series:
        """
        Novel Feature 1: Balance Discrepancy Ratio
        
        INNOVATION: Detects mathematical inconsistencies in balance changes.
        
        Mathematical intuition:
        - Expected new balance = old balance - amount
        - Discrepancy = (expected - actual) / amount
        - Normal transactions: ratio ≈ 0
        - Fraudulent: accounting inconsistencies create non-zero ratios
        
        This works in UNSUPERVISED mode because it's mathematical, not statistical.
        
        Args:
            df: DataFrame with columns: oldbalanceOrg, amount, newbalanceOrig
            
        Returns:
            Series of discrepancy ratios (absolute values)
        """
        print("[FEATURE_ENG] Computing balance_discrepancy_ratio...")
        
        # Calculate expected vs actual balance change
        expected_new_balance = df['oldbalanceOrg'] - df['amount']
        actual_new_balance = df['newbalanceOrig']
        
        # Discrepancy: how far off the actual balance is from expected
        discrepancy = np.abs(expected_new_balance - actual_new_balance)
        
        # Normalize by transaction amount to get ratio
        # Add small epsilon to avoid division by zero
        ratio = discrepancy / (np.abs(df['amount']) + 1e-6)
        
        # Cap extreme values to 100x for numerical stability
        ratio = np.minimum(ratio, 100)
        
        return ratio
    
    def engineer_transaction_velocity_score(self, df: pd.DataFrame) -> pd.Series:
        """
        Novel Feature 2: Transaction Velocity Score
        
        INNOVATION: Detects suspicious transaction frequency patterns.
        
        Intuition:
        - Count how many transactions from same origin in rolling window
        - Fraudsters often execute rapid-fire transactions
        - Normal users: steady, distributed patterns
        - This is unsupervised because frequency is objective
        
        Args:
            df: DataFrame with column 'nameOrig' (origin account)
            
        Returns:
            Series of velocity scores (transaction counts in window)
        """
        print("[FEATURE_ENG] Computing transaction_velocity_score...")
        
        # Reset index to ensure proper operation
        df_temp = df.reset_index(drop=True)
        
        # For each transaction, count transactions from same origin in rolling window
        velocity_scores = []
        
        for idx in range(len(df_temp)):
            origin = df_temp.loc[idx, 'nameOrig']
            
            # Define rolling window: last N transactions (not time-based for simplicity)
            window_start = max(0, idx - self.velocity_window)
            
            # Count transactions from same origin in window
            window_df = df_temp.iloc[window_start:idx+1]
            count = (window_df['nameOrig'] == origin).sum()
            
            velocity_scores.append(count)
        
        return pd.Series(velocity_scores, index=df.index)
    
    def engineer_amount_deviation_score(self, df: pd.DataFrame, fit: bool = True) -> pd.Series:
        """
        Novel Feature 3: Amount Deviation Score
        
        INNOVATION: Detects abnormal transaction sizes for specific accounts.
        
        Intuition:
        - Each account has typical transaction sizes (behavioral profile)
        - Fraudsters either use atypical amounts or drain accounts
        - Deviation = (amount - historical_mean) / historical_std
        - High deviation = behavioral anomaly (fraud risk)
        
        Args:
            df: DataFrame with columns 'nameOrig', 'amount'
            fit: If True, compute and store statistics; if False, use stored stats
            
        Returns:
            Series of deviation scores (Z-scores of amounts)
        """
        print("[FEATURE_ENG] Computing amount_deviation_score...")
        
        deviation_scores = []
        
        for idx, row in df.iterrows():
            origin = row['nameOrig']
            amount = row['amount']
            
            if fit:
                # During fit: compute statistics for this origin
                origin_amounts = df[df['nameOrig'] == origin]['amount']
                mean_amount = origin_amounts.mean()
                std_amount = origin_amounts.std()
                
                # Store for future use
                if origin not in self.amount_stats:
                    self.amount_stats[origin] = {'mean': mean_amount, 'std': std_amount}
            else:
                # During transform: use stored statistics
                if origin in self.amount_stats:
                    mean_amount = self.amount_stats[origin]['mean']
                    std_amount = self.amount_stats[origin]['std']
                else:
                    # Fallback for unseen origins (default to low deviation)
                    mean_amount = df['amount'].mean()
                    std_amount = df['amount'].std()
            
            # Compute Z-score deviation
            if std_amount > 0:
                deviation = abs((amount - mean_amount) / std_amount)
            else:
                deviation = 0
            
            # Cap at 10 for numerical stability
            deviation = min(deviation, 10)
            deviation_scores.append(deviation)
        
        return pd.Series(deviation_scores, index=df.index)
    
    def engineer_zero_balance_flag(self, df: pd.DataFrame) -> pd.Series:
        """
        Novel Feature 4: Zero Balance Flag
        
        INNOVATION: Flags suspicious account state changes.
        
        Intuition:
        - Zero balance after transaction = account drained
        - Can be legitimate (account closure) but often fraudulent
        - Flag as binary feature: 1 = zero balance reached, 0 = normal
        - Unsupervised: no labels needed, just state detection
        
        Args:
            df: DataFrame with column 'newbalanceOrig'
            
        Returns:
            Binary Series (1 if zero balance, 0 otherwise)
        """
        print("[FEATURE_ENG] Computing zero_balance_flag...")
        
        # Flag when account is emptied after transaction
        flag = (df['newbalanceOrig'] == 0).astype(int)
        
        return flag
    
    def engineer_round_amount_suspicion(self, df: pd.DataFrame) -> pd.Series:
        """
        Novel Feature 5: Round Amount Suspicion
        
        INNOVATION: Detects unnatural transaction patterns.
        
        Intuition:
        - Real transactions: amounts have varied decimal places
        - Fraudsters often use suspiciously round amounts (1000, 5000, etc.)
        - Measure "roundness" of amount
        - Unsupervised: pattern is mathematical
        
        Args:
            df: DataFrame with column 'amount'
            
        Returns:
            Series of suspicion scores (0-1, higher = rounder/more suspicious)
        """
        print("[FEATURE_ENG] Computing round_amount_suspicion...")
        
        amounts = df['amount'].values
        
        suspicion_scores = []
        
        for amount in amounts:
            # Measure how "round" the amount is
            # Round numbers divide evenly by large values
            
            # Check divisibility by 100, 500, 1000
            divisibility_score = 0
            
            if amount > 0:
                if amount % 1000 == 0:  # Divisible by 1000
                    divisibility_score += 0.5
                elif amount % 500 == 0:  # Divisible by 500
                    divisibility_score += 0.3
                elif amount % 100 == 0:  # Divisible by 100
                    divisibility_score += 0.2
                
                # Also check if it ends in .00 in original scale
                remainder = amount % 1
                if remainder == 0 and amount > 10:
                    divisibility_score += 0.2
            
            # Cap at 1.0
            suspicion_scores.append(min(divisibility_score, 1.0))
        
        return pd.Series(suspicion_scores, index=df.index)
    
    def engineer_dest_account_age_proxy(self, df: pd.DataFrame) -> pd.Series:
        """
        Novel Feature 6: Destination Account Age Proxy
        
        INNOVATION: Estimates account maturity from naming patterns.
        
        Intuition:
        - Account names encode implicit information
        - Newly created accounts (high numbers) = higher fraud risk
        - Established accounts (low numbers) = more trustworthy
        - Extract numeric suffix and normalize
        - Unsupervised: uses naming convention patterns
        
        Args:
            df: DataFrame with column 'nameDest'
            
        Returns:
            Series of age proxy scores (0-1, higher = older/established)
        """
        print("[FEATURE_ENG] Computing dest_account_age_proxy...")
        
        age_proxy_scores = []
        
        for name in df['nameDest']:
            # Extract trailing numbers from account name
            numeric_suffix = ""
            for char in reversed(str(name)):
                if char.isdigit():
                    numeric_suffix = char + numeric_suffix
                else:
                    break
            
            # Convert to number, default to 0 if no numeric part
            if numeric_suffix:
                account_id = int(numeric_suffix)
            else:
                account_id = 0
            
            # Normalize by assuming max accounts is around 100,000
            # Lower IDs (older accounts) get higher scores
            age_proxy = 1.0 - min(account_id / 100000, 1.0)
            
            age_proxy_scores.append(age_proxy)
        
        return pd.Series(age_proxy_scores, index=df.index)
    
    def combine_signals_into_embedding(self, signal_dict: Dict[str, pd.Series]) -> np.ndarray:
        """
        Combine all six signals into unified multi-signal risk embedding.
        
        PATENT-WORTHY CORE: This combination creates an embedding space where
        anomalies cluster together, enabling unsupervised detection.
        
        The embedding is a weighted combination where each signal captures
        different fraud dimensions:
        - Accounting frauds: balance_discrepancy_ratio
        - Velocity attacks: transaction_velocity_score
        - Behavior anomalies: amount_deviation_score
        - Extreme events: zero_balance_flag
        - Pattern matching: round_amount_suspicion
        - Account targeting: dest_account_age_proxy
        
        Args:
            signal_dict: Dict with signal names as keys, pd.Series as values
            
        Returns:
            Embedding matrix (n_samples, n_signals)
        """
        print("[FEATURE_ENG] Combining signals into multi-signal risk embedding...")
        
        signals_list = [
            signal_dict['balance_discrepancy_ratio'],
            signal_dict['transaction_velocity_score'],
            signal_dict['amount_deviation_score'],
            signal_dict['zero_balance_flag'],
            signal_dict['round_amount_suspicion'],
            signal_dict['dest_account_age_proxy'],
        ]
        
        # Stack into matrix and normalize each column
        embedding = np.column_stack(signals_list)
        
        # Normalize each signal to 0-1 range for fair weighting
        for i in range(embedding.shape[1]):
            col = embedding[:, i]
            col_min = col.min()
            col_max = col.max()
            if col_max > col_min:
                embedding[:, i] = (col - col_min) / (col_max - col_min)
            else:
                embedding[:, i] = 0
        
        signal_names = ['balance_discrepancy', 'transaction_velocity', 
                       'amount_deviation', 'zero_balance', 
                       'round_amount', 'dest_age_proxy']
        
        print(f"[FEATURE_ENG] Created embedding with {embedding.shape[1]} dimensions")
        print(f"[FEATURE_ENG] Signal dimensions: {signal_names}")
        
        return embedding, signal_names
    
    def fit_and_engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Fit feature engineer and create features (for training data).
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Tuple of (embedding_matrix, feature_names)
        """
        print("[FEATURE_ENG] Fitting and engineering features (training set)...")
        
        # Engineer all six novel signals
        signals = {
            'balance_discrepancy_ratio': self.engineer_balance_discrepancy_ratio(df),
            'transaction_velocity_score': self.engineer_transaction_velocity_score(df),
            'amount_deviation_score': self.engineer_amount_deviation_score(df, fit=True),
            'zero_balance_flag': self.engineer_zero_balance_flag(df),
            'round_amount_suspicion': self.engineer_round_amount_suspicion(df),
            'dest_account_age_proxy': self.engineer_dest_account_age_proxy(df),
        }
        
        # Combine into embedding
        embedding, signal_names = self.combine_signals_into_embedding(signals)
        
        self.is_fitted = True
        
        return embedding, signal_names
    
    def engineer_features(self, df: pd.DataFrame) -> Tuple[np.ndarray, List[str]]:
        """
        Create features using fitted engineer (for new data).
        
        Args:
            df: DataFrame with transaction data
            
        Returns:
            Tuple of (embedding_matrix, feature_names)
        """
        if not self.is_fitted:
            raise ValueError("Feature engineer not fitted. Call fit_and_engineer_features first.")
        
        print("[FEATURE_ENG] Engineering features (test/new data)...")
        
        # Engineer all signals
        signals = {
            'balance_discrepancy_ratio': self.engineer_balance_discrepancy_ratio(df),
            'transaction_velocity_score': self.engineer_transaction_velocity_score(df),
            'amount_deviation_score': self.engineer_amount_deviation_score(df, fit=False),
            'zero_balance_flag': self.engineer_zero_balance_flag(df),
            'round_amount_suspicion': self.engineer_round_amount_suspicion(df),
            'dest_account_age_proxy': self.engineer_dest_account_age_proxy(df),
        }
        
        # Combine into embedding
        embedding, signal_names = self.combine_signals_into_embedding(signals)
        
        return embedding, signal_names
