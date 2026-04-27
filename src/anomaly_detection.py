"""
Anomaly Detection Module

Runs Isolation Forest on suspicious cluster to identify confirmed fraud cases.
Flags top 2% of suspicious transactions as confirmed fraud.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler


class IsolationForestAnalyzer:
    """Detect anomalies using Isolation Forest on suspicious cluster."""
    
    def __init__(self, features: np.ndarray, suspicious_mask: np.ndarray, random_state: int = 42):
        """
        Initialize IsolationForestAnalyzer.
        
        Args:
            features: Full feature matrix (n_samples, n_features)
            suspicious_mask: Boolean mask indicating suspicious cluster members
            random_state: Random seed for reproducibility
        """
        self.features = features
        self.suspicious_mask = suspicious_mask
        self.random_state = random_state
        self.n_total = len(features)
        
        # Standardize features
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(features)
        
        self.iso_forest = None
        self.anomaly_scores_full = None  # Scores for all samples
        self.fraud_flags = None  # Binary fraud flags for all samples
        self.fraud_indices = None  # Indices of flagged fraud transactions
        
    def run_isolation_forest(self, contamination: float = 0.02):
        """
        Run Isolation Forest on suspicious cluster to detect anomalies.
        
        Isolation Forest identifies anomalies by isolating them in random forests.
        More isolated points are more anomalous.
        
        Args:
            contamination: Expected proportion of outliers in suspicious cluster (default 2%)
        
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        # Extract features for suspicious cluster only
        suspicious_features = self.features_scaled[self.suspicious_mask]
        n_suspicious = len(suspicious_features)
        
        print(f"\n🔍 Running Isolation Forest on {n_suspicious:,} suspicious transactions...")
        
        # Fit Isolation Forest
        self.iso_forest = IsolationForest(
            contamination=contamination,
            random_state=self.random_state,
            n_estimators=100
        )
        
        # Get predictions and anomaly scores
        predictions = self.iso_forest.fit_predict(suspicious_features)
        
        # Get raw anomaly scores (-1 = anomaly, 1 = normal)
        # Higher scores = more anomalous
        raw_scores = self.iso_forest.score_samples(suspicious_features)
        
        # Convert to positive anomaly scores (higher = more anomalous)
        anomaly_scores_suspicious = -raw_scores  # Flip sign so higher = more anomalous
        
        # Create full-size arrays
        self.anomaly_scores_full = np.zeros(self.n_total)
        self.anomaly_scores_full[self.suspicious_mask] = anomaly_scores_suspicious
        
        # Create fraud flags (top 2% of suspicious cluster)
        self.fraud_flags = np.zeros(self.n_total, dtype=bool)
        
        # Flag top 2% of suspicious cluster as fraud
        n_fraud = max(1, int(n_suspicious * contamination))
        fraud_threshold_idx = np.argsort(anomaly_scores_suspicious)[-n_fraud:]
        
        # Convert back to full indices
        suspicious_indices = np.where(self.suspicious_mask)[0]
        fraud_full_indices = suspicious_indices[fraud_threshold_idx]
        
        self.fraud_flags[fraud_full_indices] = True
        self.fraud_indices = fraud_full_indices
        
        print(f"✓ Isolation Forest fitted on suspicious cluster")
        print(f"  Anomaly scores - min: {anomaly_scores_suspicious.min():.4f}, "
              f"max: {anomaly_scores_suspicious.max():.4f}, "
              f"mean: {anomaly_scores_suspicious.mean():.4f}")
        print(f"  Flagged {len(self.fraud_indices)} transactions (top 2%) as confirmed fraud")
        print(f"  Mean anomaly score of flagged: {self.anomaly_scores_full[self.fraud_flags].mean():.4f}")
        
        return self.anomaly_scores_full
    
    def get_anomaly_scores(self) -> np.ndarray:
        """
        Get anomaly scores for all samples.
        
        Returns:
            Array of anomaly scores (0 for non-suspicious cluster, scores for suspicious cluster)
        """
        if self.anomaly_scores_full is None:
            raise RuntimeError("Isolation Forest not yet fitted. Call run_isolation_forest() first.")
        
        return self.anomaly_scores_full
    
    def get_fraud_flags(self) -> np.ndarray:
        """
        Get binary fraud flags for all samples.
        
        Returns:
            Boolean array (True = confirmed fraud, False = normal)
        """
        if self.fraud_flags is None:
            raise RuntimeError("Isolation Forest not yet fitted. Call run_isolation_forest() first.")
        
        return self.fraud_flags
    
    def get_fraud_indices(self) -> np.ndarray:
        """
        Get indices of flagged fraud transactions.
        
        Returns:
            Array of indices
        """
        if self.fraud_indices is None:
            raise RuntimeError("Isolation Forest not yet fitted. Call run_isolation_forest() first.")
        
        return self.fraud_indices
    
    def get_fraud_count(self) -> int:
        """Get count of flagged fraud transactions."""
        if self.fraud_flags is None:
            raise RuntimeError("Isolation Forest not yet fitted. Call run_isolation_forest() first.")
        
        return np.sum(self.fraud_flags)
    
    def get_fraud_percentage(self) -> float:
        """Get percentage of total transactions flagged as fraud."""
        if self.fraud_flags is None:
            raise RuntimeError("Isolation Forest not yet fitted. Call run_isolation_forest() first.")
        
        return (np.sum(self.fraud_flags) / self.n_total) * 100
    
    def get_flagged_anomaly_scores_mean(self) -> float:
        """Get mean anomaly score of flagged fraud transactions."""
        if self.fraud_flags is None or self.anomaly_scores_full is None:
            raise RuntimeError("Isolation Forest not yet fitted. Call run_isolation_forest() first.")
        
        if not np.any(self.fraud_flags):
            return 0.0
        
        return self.anomaly_scores_full[self.fraud_flags].mean()
