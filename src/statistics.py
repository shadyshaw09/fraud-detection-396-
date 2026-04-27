"""
Statistics Calculation Module

Computes and displays all statistical outcomes from fraud detection pipeline.
"""

import numpy as np
import pandas as pd


class StatisticsCalculator:
    """Calculate and report fraud detection statistics."""
    
    def __init__(
        self,
        n_total: int,
        n_clusters: int,
        silhouette_score: float,
        wcc_values: list,
        fraud_count: int,
        fraud_percentage: float,
        mean_anomaly_score_flagged: float,
        balance_discrepancy: np.ndarray,
        fraud_flags: np.ndarray
    ):
        """
        Initialize StatisticsCalculator.
        
        Args:
            n_total: Total transactions scanned
            n_clusters: Number of clusters formed
            silhouette_score: Silhouette Score for clustering
            wcc_values: WCC values per cluster
            fraud_count: Number of transactions flagged as fraud
            fraud_percentage: Percentage of total flagged as fraud
            mean_anomaly_score_flagged: Mean anomaly score of flagged transactions
            balance_discrepancy: Balance discrepancy values for all transactions
            fraud_flags: Boolean array of fraud flags
        """
        self.n_total = n_total
        self.n_clusters = n_clusters
        self.silhouette_score = silhouette_score
        self.wcc_values = wcc_values
        self.fraud_count = fraud_count
        self.fraud_percentage = fraud_percentage
        self.mean_anomaly_score_flagged = mean_anomaly_score_flagged
        self.balance_discrepancy = balance_discrepancy
        self.fraud_flags = fraud_flags
        
        self.stats_dict = {}
    
    def calculate_balance_discrepancy_stats(self) -> dict:
        """
        Calculate mean balance_discrepancy for flagged vs normal transactions.
        
        Returns:
            Dictionary with flagged and normal mean values
        """
        if np.any(self.fraud_flags):
            mean_discrepancy_flagged = self.balance_discrepancy[self.fraud_flags].mean()
        else:
            mean_discrepancy_flagged = 0.0
        
        if np.any(~self.fraud_flags):
            mean_discrepancy_normal = self.balance_discrepancy[~self.fraud_flags].mean()
        else:
            mean_discrepancy_normal = 0.0
        
        return {
            'flagged': mean_discrepancy_flagged,
            'normal': mean_discrepancy_normal
        }
    
    def calculate_all_stats(self) -> dict:
        """
        Calculate all fraud detection statistics.
        
        Returns:
            Dictionary with all computed statistics
        """
        balance_stats = self.calculate_balance_discrepancy_stats()
        
        self.stats_dict = {
            'total_transactions': self.n_total,
            'num_clusters': self.n_clusters,
            'silhouette_score': self.silhouette_score,
            'wcc_per_cluster': self.wcc_values,
            'fraud_count': self.fraud_count,
            'fraud_percentage': self.fraud_percentage,
            'mean_anomaly_score_flagged': self.mean_anomaly_score_flagged,
            'mean_balance_discrepancy_flagged': balance_stats['flagged'],
            'mean_balance_discrepancy_normal': balance_stats['normal']
        }
        
        return self.stats_dict
    
    def print_statistics(self):
        """Print all statistics to console with clear formatting."""
        if not self.stats_dict:
            self.calculate_all_stats()
        
        print("\n" + "="*70)
        print("📊 FRAUD DETECTION STATISTICAL OUTCOMES")
        print("="*70)
        
        print(f"\n📈 Data Overview:")
        print(f"  Total transactions scanned:      {self.stats_dict['total_transactions']:,}")
        print(f"  Number of clusters:              {self.stats_dict['num_clusters']}")
        
        print(f"\n🎯 Clustering Quality Metrics:")
        print(f"  Silhouette Score:                {self.stats_dict['silhouette_score']:.4f}")
        
        wcc_str = ", ".join([f"{wcc:.4f}" for wcc in self.stats_dict['wcc_per_cluster']])
        print(f"  WCC per cluster:                 [{wcc_str}]")
        
        print(f"\n🚨 Fraud Detection Results:")
        print(f"  Fraud flagged count:             {self.stats_dict['fraud_count']:,}")
        print(f"  Detection rate (% of total):     {self.stats_dict['fraud_percentage']:.2f}%")
        print(f"  Mean anomaly score (flagged):    {self.stats_dict['mean_anomaly_score_flagged']:.4f}")
        
        print(f"\n💰 Balance Discrepancy Analysis:")
        print(f"  Mean balance_discrepancy (flagged):  {self.stats_dict['mean_balance_discrepancy_flagged']:.2f}")
        print(f"  Mean balance_discrepancy (normal):   {self.stats_dict['mean_balance_discrepancy_normal']:.2f}")
        
        print("\n" + "="*70)
    
    def get_stats_dict(self) -> dict:
        """
        Get statistics dictionary (for dashboard).
        
        Returns:
            Dictionary with all statistics
        """
        if not self.stats_dict:
            self.calculate_all_stats()
        
        return self.stats_dict
    
    def get_stats_dataframe(self) -> pd.DataFrame:
        """
        Get statistics as a pandas DataFrame (for dashboard table).
        
        Returns:
            DataFrame with statistics
        """
        if not self.stats_dict:
            self.calculate_all_stats()
        
        # Format for display
        stats_display = {
            'Metric': [
                'Total Scanned',
                'Clusters',
                'Silhouette Score',
                'WCC Cluster 0',
                'WCC Cluster 1',
                'WCC Cluster 2',
                'Fraud Count',
                'Detection Rate (%)',
                'Mean Anomaly Score (Flagged)',
                'Mean Balance Discrepancy (Flagged)',
                'Mean Balance Discrepancy (Normal)'
            ],
            'Value': [
                f"{self.stats_dict['total_transactions']:,}",
                f"{self.stats_dict['num_clusters']}",
                f"{self.stats_dict['silhouette_score']:.4f}",
                f"{self.stats_dict['wcc_per_cluster'][0]:.4f}",
                f"{self.stats_dict['wcc_per_cluster'][1]:.4f}",
                f"{self.stats_dict['wcc_per_cluster'][2]:.4f}",
                f"{self.stats_dict['fraud_count']:,}",
                f"{self.stats_dict['fraud_percentage']:.2f}%",
                f"{self.stats_dict['mean_anomaly_score_flagged']:.4f}",
                f"{self.stats_dict['mean_balance_discrepancy_flagged']:.2f}",
                f"{self.stats_dict['mean_balance_discrepancy_normal']:.2f}"
            ]
        }
        
        return pd.DataFrame(stats_display)
