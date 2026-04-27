"""
Isolation Forest Model for Unsupervised Anomaly Detection

Uses the Isolation Forest algorithm which is highly effective for anomaly detection
because it doesn't require distance or density calculations - it simply isolates
anomalies by random feature selection and splitting.

Theory:
- Normal points require many splits to be isolated
- Anomalies require few splits (they're easier to isolate)
- Anomaly score is based on path length to leaf node
"""

import numpy as np
from sklearn.ensemble import IsolationForest
import joblib
from typing import Optional, Tuple
import warnings

warnings.filterwarnings('ignore')


class IsolationForestModel:
    """
    Isolation Forest-based anomaly detection model.
    
    Provides unsupervised anomaly detection using Random Forest-inspired
    isolation approach. Particularly effective for high-dimensional data
    like our engineered fraud features.
    """
    
    def __init__(
        self, 
        contamination: float = 0.1,
        n_estimators: int = 100,
        random_state: int = 42,
        n_jobs: int = -1
    ):
        """
        Initialize Isolation Forest model.
        
        Args:
            contamination: Expected proportion of outliers (0.0-1.0).
                          Default 0.1 = 10% expected anomalies.
            n_estimators: Number of isolation trees to build
            random_state: Random seed for reproducibility
            n_jobs: Number of parallel jobs (-1 = use all cores)
        """
        self.contamination = contamination
        self.n_estimators = n_estimators
        self.random_state = random_state
        self.n_jobs = n_jobs
        
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=random_state,
            n_jobs=n_jobs
        )
        
        self.is_fitted = False
        self.n_features = None
        
    def fit(self, X: np.ndarray) -> None:
        """
        Fit Isolation Forest on training data (UNSUPERVISED).
        
        No labels needed - the model learns what 'normal' looks like
        by assuming the majority of data are normal transactions.
        
        Args:
            X: Training feature matrix (n_samples, n_features)
               Typically the engineered multi-signal risk embedding
        """
        print(f"[ISO_FOREST] Fitting Isolation Forest on {X.shape[0]} samples with {X.shape[1]} features...")
        print(f"[ISO_FOREST] Contamination parameter: {self.contamination}")
        
        self.model.fit(X)
        self.n_features = X.shape[1]
        self.is_fitted = True
        
        print(f"[ISO_FOREST] Model fitted successfully")
    
    def predict_anomalies(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies in new data.
        
        Returns binary predictions: 1 for anomalies, -1 for normal.
        Uses the contamination parameter to determine threshold.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of predictions (1 for anomaly, -1 for normal)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}"
            )
        
        predictions = self.model.predict(X)
        return predictions
    
    def get_anomaly_scores(self, X: np.ndarray) -> np.ndarray:
        """
        Get raw anomaly scores for new data.
        
        Higher scores indicate more anomalous behavior.
        Scores typically range from 0 to 1 after normalization.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Array of anomaly scores (higher = more anomalous)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if X.shape[1] != self.n_features:
            raise ValueError(
                f"Expected {self.n_features} features, got {X.shape[1]}"
            )
        
        # Get decision scores (negative values = normal, positive = anomalies)
        decision_scores = self.model.decision_function(X)
        
        # Convert to 0-1 range for easier interpretation
        # Lower decision scores (more negative) = more normal
        # Higher decision scores (less negative/positive) = more anomalous
        scores = 1 / (1 + np.exp(-decision_scores))  # Sigmoid normalization
        
        return scores
    
    def predict_and_score(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get both binary predictions and continuous anomaly scores.
        
        Args:
            X: Feature matrix (n_samples, n_features)
            
        Returns:
            Tuple of (predictions, scores)
        """
        predictions = self.predict_anomalies(X)
        scores = self.get_anomaly_scores(X)
        
        return predictions, scores
    
    def save_model(self, filepath: str) -> None:
        """
        Save trained model to disk.
        
        Args:
            filepath: Path to save model (.pkl or .joblib)
        """
        if not self.is_fitted:
            raise ValueError("Model not fitted. Cannot save unfitted model.")
        
        joblib.dump(self.model, filepath)
        print(f"[ISO_FOREST] Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load trained model from disk.
        
        Args:
            filepath: Path to saved model
        """
        self.model = joblib.load(filepath)
        self.is_fitted = True
        self.n_features = self.model.n_features_in_
        print(f"[ISO_FOREST] Model loaded from {filepath}")
    
    def get_model_info(self) -> dict:
        """
        Get information about the fitted model.
        
        Returns:
            Dictionary with model parameters and statistics
        """
        if not self.is_fitted:
            return {"status": "Not fitted"}
        
        return {
            "status": "Fitted",
            "algorithm": "Isolation Forest",
            "n_features": self.n_features,
            "n_estimators": self.n_estimators,
            "contamination": self.contamination,
            "random_state": self.random_state,
        }
