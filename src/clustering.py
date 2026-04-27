"""
K-Means Clustering Module

Implements clustering analysis with:
  - Elbow Method (k=2 to k=8)
  - K-Means with k=3
  - Silhouette Score calculation
  - WCC (Within-Cluster Cohesion) calculation
  - Suspicious cluster identification
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform


class KMeansAnalyzer:
    """Perform K-Means clustering with comprehensive evaluation metrics."""
    
    def __init__(self, features: np.ndarray, balance_discrepancy: np.ndarray, k_final: int = 3):
        """
        Initialize KMeansAnalyzer.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            balance_discrepancy: Balance discrepancy values for each sample
            k_final: Final number of clusters (default 3)
        """
        self.features = features
        self.balance_discrepancy = balance_discrepancy
        self.k_final = k_final
        
        # Standardize features for clustering
        self.scaler = StandardScaler()
        self.features_scaled = self.scaler.fit_transform(features)
        
        self.kmeans_models = {}  # Store models for k=2 to k=8
        self.inertia_values = []
        self.k_range = range(2, 9)
        
        self.final_model = None
        self.cluster_labels = None
        self.silhouette_score_val = None
        self.wcc_values = None
        self.suspicious_cluster_id = None
    
    def run_elbow_analysis(self):
        """
        Run K-Means for k=2 to k=8 and calculate inertia values.
        
        Inertia measures sum of squared distances from each point to its cluster center.
        The "elbow" indicates optimal k.
        """
        print("\n📊 Elbow Method: Running K-Means for k=2 to k=8...")
        
        for k in self.k_range:
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(self.features_scaled)
            self.kmeans_models[k] = km
            self.inertia_values.append(km.inertia_)
            print(f"  k={k}: inertia={km.inertia_:.2f}")
        
        print(f"✓ Elbow Method complete: inertia={self.inertia_values}")
        
        return self.inertia_values
    
    def fit_final_model(self):
        """
        Fit final K-Means model with k_final clusters.
        """
        print(f"\n🎯 Fitting final K-Means with k={self.k_final}...")
        
        self.final_model = KMeans(n_clusters=self.k_final, random_state=42, n_init=10)
        self.cluster_labels = self.final_model.fit_predict(self.features_scaled)
        
        print(f"✓ K-Means fitted with {self.k_final} clusters")
        print(f"  Cluster distribution: {np.bincount(self.cluster_labels)}")
        
        return self.cluster_labels
    
    def calculate_silhouette_score(self) -> float:
        """
        Calculate Silhouette Score for clustering quality.
        
        Silhouette Score measures how similar a point is to its own cluster
        vs other clusters. Range: [-1, 1], higher is better.
        
        Returns:
            Silhouette score
        """
        if self.cluster_labels is None:
            raise RuntimeError("Final model not yet fitted. Call fit_final_model() first.")
        
        print("\n📈 Calculating Silhouette Score...")
        
        self.silhouette_score_val = silhouette_score(
            self.features_scaled,
            self.cluster_labels,
            metric='euclidean'
        )
        
        print(f"✓ Silhouette Score: {self.silhouette_score_val:.4f}")
        
        return self.silhouette_score_val
    
    def calculate_wcc(self) -> list:
        """
        Calculate Within-Cluster Cohesion (WCC).
        
        WCC is the average intra-cluster distance for each cluster.
        Lower is better (tighter clusters). Returns a value per cluster.
        
        Returns:
            List of WCC values, one per cluster
        """
        if self.cluster_labels is None:
            raise RuntimeError("Final model not yet fitted. Call fit_final_model() first.")
        
        print("\n🔗 Calculating Within-Cluster Cohesion (WCC)...")
        
        self.wcc_values = []
        
        for cluster_id in range(self.k_final):
            # Get all points in this cluster
            cluster_mask = self.cluster_labels == cluster_id
            cluster_points = self.features_scaled[cluster_mask]
            
            if len(cluster_points) <= 1:
                self.wcc_values.append(0.0)
                continue
            
            # Calculate pairwise distances within cluster
            if len(cluster_points) > 0:
                distances = pdist(cluster_points, metric='euclidean')
                if len(distances) > 0:
                    wcc = np.mean(distances)
                else:
                    wcc = 0.0
            else:
                wcc = 0.0
            
            self.wcc_values.append(wcc)
        
        print(f"✓ WCC per cluster: {[f'{w:.4f}' for w in self.wcc_values]}")
        
        return self.wcc_values
    
    def identify_suspicious_cluster(self) -> int:
        """
        Identify suspicious cluster as the one with HIGHEST mean balance_discrepancy.
        
        High balance discrepancy suggests unusual account balance changes,
        often associated with fraudulent activity.
        
        Returns:
            Index of suspicious cluster
        """
        if self.cluster_labels is None:
            raise RuntimeError("Final model not yet fitted. Call fit_final_model() first.")
        
        print("\n🚨 Identifying Suspicious Cluster...")
        
        cluster_balance_discrepancies = []
        
        for cluster_id in range(self.k_final):
            cluster_mask = self.cluster_labels == cluster_id
            mean_discrepancy = self.balance_discrepancy[cluster_mask].mean()
            cluster_balance_discrepancies.append(mean_discrepancy)
            print(f"  Cluster {cluster_id}: mean balance_discrepancy = {mean_discrepancy:.2f}")
        
        self.suspicious_cluster_id = np.argmax(cluster_balance_discrepancies)
        suspicious_value = cluster_balance_discrepancies[self.suspicious_cluster_id]
        
        print(f"✓ Suspicious Cluster: {self.suspicious_cluster_id} "
              f"(mean balance_discrepancy: {suspicious_value:.2f})")
        
        return self.suspicious_cluster_id
    
    def get_suspicious_cluster_mask(self) -> np.ndarray:
        """
        Get boolean mask for transactions in suspicious cluster.
        
        Returns:
            Boolean array indicating membership in suspicious cluster
        """
        if self.suspicious_cluster_id is None:
            raise RuntimeError("Suspicious cluster not identified. Call identify_suspicious_cluster() first.")
        
        return self.cluster_labels == self.suspicious_cluster_id
    
    def get_suspicious_indices(self) -> np.ndarray:
        """
        Get indices of transactions in suspicious cluster.
        
        Returns:
            Array of indices
        """
        mask = self.get_suspicious_cluster_mask()
        return np.where(mask)[0]
    
    def run_complete_analysis(self):
        """
        Run complete clustering analysis pipeline.
        
        Returns:
            Tuple of (cluster_labels, silhouette_score, wcc_values, 
                     suspicious_cluster_id, inertia_values)
        """
        self.run_elbow_analysis()
        self.fit_final_model()
        self.calculate_silhouette_score()
        self.calculate_wcc()
        self.identify_suspicious_cluster()
        
        return (
            self.cluster_labels,
            self.silhouette_score_val,
            self.wcc_values,
            self.suspicious_cluster_id,
            self.inertia_values
        )
