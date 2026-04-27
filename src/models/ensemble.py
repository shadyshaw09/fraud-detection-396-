"""
Ensemble Anomaly Scorer

Combines predictions from multiple anomaly detection models:
1. Isolation Forest - tree-based method
2. Autoencoder - neural network reconstruction method

INNOVATION: Different models capture different fraud patterns.
- Isolation Forest excels at finding outliers in high-dimensional space
- Autoencoder excels at finding reconstruction anomalies
- Ensemble combines strengths through weighted voting

This creates a more robust final anomaly score.
"""

import numpy as np
from typing import Tuple, Optional, Dict
import warnings

warnings.filterwarnings('ignore')


class EnsembleAnomalyScorer:
    """
    Ensemble scorer combining multiple anomaly detection models.
    
    Provides weighted combination of scores from different algorithms,
    creating a more robust final anomaly detection score.
    """
    
    def __init__(
        self,
        iso_forest_weight: float = 0.5,
        autoencoder_weight: float = 0.5,
        combination_method: str = 'weighted_mean'
    ):
        """
        Initialize ensemble scorer.
        
        Args:
            iso_forest_weight: Weight for Isolation Forest scores (0-1)
            autoencoder_weight: Weight for Autoencoder scores (0-1)
            combination_method: 'weighted_mean', 'max', or 'voting'
        """
        # Normalize weights to sum to 1
        total_weight = iso_forest_weight + autoencoder_weight
        self.iso_forest_weight = iso_forest_weight / total_weight
        self.autoencoder_weight = autoencoder_weight / total_weight
        self.combination_method = combination_method
        
        print(f"[ENSEMBLE] Initialized with method: {combination_method}")
        print(f"[ENSEMBLE] Isolation Forest weight: {self.iso_forest_weight:.2f}")
        print(f"[ENSEMBLE] Autoencoder weight: {self.autoencoder_weight:.2f}")
    
    def compute_weighted_mean(
        self,
        iso_forest_scores: np.ndarray,
        autoencoder_scores: np.ndarray
    ) -> np.ndarray:
        """
        Combine scores using weighted mean.
        
        THEORY: Assumes both scores are on 0-1 scale.
        Weighted average respects the confidence of each model.
        
        Args:
            iso_forest_scores: Array of IF anomaly scores (0-1)
            autoencoder_scores: Array of AE anomaly scores (0-1)
            
        Returns:
            Array of combined scores
        """
        combined_scores = (
            self.iso_forest_weight * iso_forest_scores +
            self.autoencoder_weight * autoencoder_scores
        )
        return combined_scores
    
    def compute_max_score(
        self,
        iso_forest_scores: np.ndarray,
        autoencoder_scores: np.ndarray
    ) -> np.ndarray:
        """
        Combine scores using maximum (pessimistic/conservative).
        
        THEORY: If either model flags an anomaly, treat as anomalous.
        More likely to catch fraud but may have higher false positives.
        
        Args:
            iso_forest_scores: Array of IF anomaly scores (0-1)
            autoencoder_scores: Array of AE anomaly scores (0-1)
            
        Returns:
            Array of combined scores
        """
        combined_scores = np.maximum(iso_forest_scores, autoencoder_scores)
        return combined_scores
    
    def compute_voting_score(
        self,
        iso_forest_predictions: np.ndarray,
        autoencoder_predictions: np.ndarray,
        iso_forest_scores: np.ndarray,
        autoencoder_scores: np.ndarray
    ) -> np.ndarray:
        """
        Combine using weighted voting.
        
        THEORY: Each model votes (1 for anomaly, -1 for normal).
        Voting score is weighted by model confidences (scores).
        Requires both predictions and scores.
        
        Args:
            iso_forest_predictions: Binary predictions (1 or -1)
            autoencoder_predictions: Binary predictions (1 or -1)
            iso_forest_scores: Continuous scores (0-1)
            autoencoder_scores: Continuous scores (0-1)
            
        Returns:
            Array of combined scores
        """
        # Convert predictions to 0 or 1 (1 = anomaly, 0 = normal)
        iso_forest_votes = (iso_forest_predictions == 1).astype(float)
        autoencoder_votes = (autoencoder_predictions == 1).astype(float)
        
        # Weight votes by confidence
        weighted_votes = (
            self.iso_forest_weight * iso_forest_votes * iso_forest_scores +
            self.autoencoder_weight * autoencoder_votes * autoencoder_scores
        )
        
        return weighted_votes
    
    def combine_scores(
        self,
        iso_forest_scores: np.ndarray,
        autoencoder_scores: np.ndarray,
        iso_forest_predictions: Optional[np.ndarray] = None,
        autoencoder_predictions: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Combine scores from both models into final anomaly score.
        
        Args:
            iso_forest_scores: Array of IF scores (0-1)
            autoencoder_scores: Array of AE scores (0-1)
            iso_forest_predictions: Optional IF predictions (for voting)
            autoencoder_predictions: Optional AE predictions (for voting)
            
        Returns:
            Tuple of (final_scores, scores_dict)
        """
        print(f"[ENSEMBLE] Combining scores using {self.combination_method} method...")
        
        if self.combination_method == 'weighted_mean':
            final_scores = self.compute_weighted_mean(iso_forest_scores, autoencoder_scores)
        
        elif self.combination_method == 'max':
            final_scores = self.compute_max_score(iso_forest_scores, autoencoder_scores)
        
        elif self.combination_method == 'voting':
            if iso_forest_predictions is None or autoencoder_predictions is None:
                raise ValueError("Predictions required for voting method")
            final_scores = self.compute_voting_score(
                iso_forest_predictions,
                autoencoder_predictions,
                iso_forest_scores,
                autoencoder_scores
            )
        
        else:
            raise ValueError(f"Unknown combination method: {self.combination_method}")
        
        # Normalize final scores to 0-1 range
        final_scores = np.clip(final_scores, 0, 1)
        
        # Create detailed scores dictionary
        scores_dict = {
            'isolation_forest': iso_forest_scores,
            'autoencoder': autoencoder_scores,
            'ensemble': final_scores,
        }
        
        print(f"[ENSEMBLE] Ensemble scores computed")
        print(f"[ENSEMBLE] Mean ensemble score: {final_scores.mean():.6f}")
        print(f"[ENSEMBLE] Std ensemble score: {final_scores.std():.6f}")
        
        return final_scores, scores_dict
    
    def get_predictions_at_threshold(
        self,
        ensemble_scores: np.ndarray,
        threshold: float = 0.5
    ) -> np.ndarray:
        """
        Get binary predictions from continuous scores at threshold.
        
        Args:
            ensemble_scores: Array of ensemble anomaly scores (0-1)
            threshold: Decision threshold (default 0.5)
            
        Returns:
            Binary predictions (1 for anomaly, -1 for normal)
        """
        predictions = np.where(ensemble_scores > threshold, 1, -1)
        return predictions
    
    def compute_anomaly_percentile(
        self,
        ensemble_scores: np.ndarray,
        percentile: float = 95
    ) -> Tuple[float, np.ndarray]:
        """
        Find threshold at given percentile and get predictions.
        
        Useful for unsupervised threshold selection.
        
        Args:
            ensemble_scores: Array of ensemble anomaly scores (0-1)
            percentile: Percentile threshold (default 95th = top 5% flagged)
            
        Returns:
            Tuple of (threshold, predictions)
        """
        threshold = np.percentile(ensemble_scores, percentile)
        predictions = self.get_predictions_at_threshold(ensemble_scores, threshold)
        
        n_anomalies = (predictions == 1).sum()
        pct_flagged = (n_anomalies / len(predictions)) * 100
        
        print(f"[ENSEMBLE] {percentile}th percentile threshold: {threshold:.6f}")
        print(f"[ENSEMBLE] Flagged {n_anomalies} anomalies ({pct_flagged:.2f}%)")
        
        return threshold, predictions
    
    def get_detailed_report(
        self,
        ensemble_scores: np.ndarray,
        threshold: Optional[float] = None,
        percentile: Optional[float] = None
    ) -> Dict:
        """
        Get comprehensive analysis report.
        
        Args:
            ensemble_scores: Array of ensemble anomaly scores
            threshold: Optional fixed threshold
            percentile: Optional percentile threshold
            
        Returns:
            Dictionary with detailed statistics
        """
        report = {
            'n_samples': len(ensemble_scores),
            'mean_score': float(ensemble_scores.mean()),
            'std_score': float(ensemble_scores.std()),
            'min_score': float(ensemble_scores.min()),
            'max_score': float(ensemble_scores.max()),
            'median_score': float(np.median(ensemble_scores)),
            'q25_score': float(np.percentile(ensemble_scores, 25)),
            'q75_score': float(np.percentile(ensemble_scores, 75)),
        }
        
        if threshold is not None:
            predictions = self.get_predictions_at_threshold(ensemble_scores, threshold)
            n_anomalies = (predictions == 1).sum()
            report['threshold_used'] = threshold
            report['n_anomalies_at_threshold'] = int(n_anomalies)
            report['pct_anomalies_at_threshold'] = float((n_anomalies / len(ensemble_scores)) * 100)
        
        if percentile is not None:
            threshold = np.percentile(ensemble_scores, percentile)
            predictions = self.get_predictions_at_threshold(ensemble_scores, threshold)
            n_anomalies = (predictions == 1).sum()
            report[f'threshold_at_{percentile}th_percentile'] = threshold
            report[f'n_anomalies_at_{percentile}th_percentile'] = int(n_anomalies)
            report[f'pct_anomalies_at_{percentile}th_percentile'] = float((n_anomalies / len(ensemble_scores)) * 100)
        
        return report
    
    def get_ensemble_config(self) -> Dict:
        """Get configuration of ensemble."""
        return {
            'iso_forest_weight': self.iso_forest_weight,
            'autoencoder_weight': self.autoencoder_weight,
            'combination_method': self.combination_method,
        }
