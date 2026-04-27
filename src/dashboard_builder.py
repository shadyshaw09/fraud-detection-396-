"""
Dashboard Builder Module

Creates a comprehensive 6-panel plotly dashboard with:
  - Elbow curve
  - UMAP cluster visualization
  - UMAP fraud detection visualization
  - Anomaly score distribution
  - Statistics table
  - Top 10 suspicious transactions table
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from umap import UMAP


class DashboardBuilder:
    """Build interactive plotly dashboard for fraud detection results."""
    
    def __init__(
        self,
        features: np.ndarray,
        cluster_labels: np.ndarray,
        inertia_values: list,
        anomaly_scores: np.ndarray,
        fraud_flags: np.ndarray,
        balance_discrepancy: np.ndarray,
        amounts: np.ndarray,
        stats_dict: dict,
        k_range: list = None
    ):
        """
        Initialize DashboardBuilder.
        
        Args:
            features: Feature matrix (n_samples, n_features)
            cluster_labels: Cluster assignment for each sample
            inertia_values: Inertia values for k=2 to k=8
            anomaly_scores: Anomaly scores for each sample
            fraud_flags: Boolean fraud flags
            balance_discrepancy: Balance discrepancy values
            amounts: Transaction amounts
            stats_dict: Dictionary with all statistics
            k_range: Range of k values (default 2-8)
        """
        self.features = features
        self.cluster_labels = cluster_labels
        self.inertia_values = inertia_values
        self.anomaly_scores = anomaly_scores
        self.fraud_flags = fraud_flags
        self.balance_discrepancy = balance_discrepancy
        self.amounts = amounts
        self.stats_dict = stats_dict
        self.k_range = k_range if k_range else list(range(2, 9))
        
        self.umap_embedding = None
        self.fig = None
    
    def compute_umap_embedding(self, n_components: int = 2, n_neighbors: int = 15, min_dist: float = 0.1):
        """
        Compute UMAP embedding for 2D visualization.
        
        Args:
            n_components: Number of dimensions (default 2)
            n_neighbors: Number of neighbors for UMAP (default 15)
            min_dist: Minimum distance for UMAP (default 0.1)
        """
        print("  Computing UMAP embedding...")
        
        umap = UMAP(
            n_components=n_components,
            n_neighbors=n_neighbors,
            min_dist=min_dist,
            random_state=42
        )
        
        self.umap_embedding = umap.fit_transform(self.features)
        
        print(f"  ✓ UMAP embedding computed: {self.umap_embedding.shape}")
    
    def build_dashboard(self, output_path: str = 'outputs/dashboard.html'):
        """
        Build complete 6-panel dashboard.
        
        Args:
            output_path: Path to save HTML file
        """
        print("\n📊 Building interactive dashboard...")
        
        # Compute UMAP if not already done
        if self.umap_embedding is None:
            self.compute_umap_embedding()
        
        # Create subplots (2 rows x 3 columns)
        self.fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=(
                '①: Elbow Curve (K-Means)',
                '②: UMAP by Cluster',
                '③: UMAP by Fraud Status',
                '④: Anomaly Score Distribution',
                '⑤: Statistical Outcomes',
                '⑥: Top 10 Suspicious Transactions'
            ),
            specs=[
                [{'type': 'scatter'}, {'type': 'scatter'}, {'type': 'scatter'}],
                [{'type': 'histogram'}, {'type': 'table'}, {'type': 'table'}]
            ],
            vertical_spacing=0.15,
            horizontal_spacing=0.12
        )
        
        # Panel 1: Elbow Curve
        self._add_elbow_curve()
        
        # Panel 2: UMAP by Cluster
        self._add_umap_clusters()
        
        # Panel 3: UMAP by Fraud
        self._add_umap_fraud()
        
        # Panel 4: Anomaly Score Histogram
        self._add_anomaly_histogram()
        
        # Panel 5: Statistics Table
        self._add_statistics_table()
        
        # Panel 6: Top 10 Suspicious
        self._add_top_suspicious_table()
        
        # Add header stats bar
        self._add_header_stats()
        
        # Update layout
        self.fig.update_layout(
            title_text="<b>🚨 Fraud Detection Dashboard</b>",
            title_font_size=24,
            showlegend=True,
            height=1000,
            template='plotly_dark',
            hovermode='closest',
            font=dict(size=11)
        )
        
        # Save to file
        self.fig.write_html(output_path)
        print(f"✓ Dashboard saved to {output_path}")
        
        return self.fig
    
    def _add_elbow_curve(self):
        """Add Panel 1: Elbow curve (k vs inertia)."""
        self.fig.add_trace(
            go.Scatter(
                x=list(self.k_range),
                y=self.inertia_values,
                mode='lines+markers',
                name='Inertia',
                line=dict(color='#FF6B6B', width=3),
                marker=dict(size=8),
                hovertemplate='<b>k=%{x}</b><br>Inertia: %{y:.0f}<extra></extra>'
            ),
            row=1, col=1
        )
        
        self.fig.update_xaxes(title_text='k (Number of Clusters)', row=1, col=1)
        self.fig.update_yaxes(title_text='Inertia', row=1, col=1)
    
    def _add_umap_clusters(self):
        """Add Panel 2: UMAP scatter colored by K-Means cluster."""
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']  # Blue, Orange, Green
        cluster_names = ['Cluster 0', 'Cluster 1', 'Cluster 2']
        
        for cluster_id in range(3):
            mask = self.cluster_labels == cluster_id
            self.fig.add_trace(
                go.Scatter(
                    x=self.umap_embedding[mask, 0],
                    y=self.umap_embedding[mask, 1],
                    mode='markers',
                    name=cluster_names[cluster_id],
                    marker=dict(size=5, color=colors[cluster_id], opacity=0.7),
                    hovertemplate='<b>%{customdata}</b><br>UMAP-1: %{x:.2f}<br>UMAP-2: %{y:.2f}<extra></extra>',
                    customdata=[cluster_names[cluster_id]] * np.sum(mask)
                ),
                row=1, col=2
            )
        
        self.fig.update_xaxes(title_text='UMAP Component 1', row=1, col=2)
        self.fig.update_yaxes(title_text='UMAP Component 2', row=1, col=2)
    
    def _add_umap_fraud(self):
        """Add Panel 3: UMAP scatter (blue=normal, red=fraud)."""
        # Normal transactions (blue)
        normal_mask = ~self.fraud_flags
        self.fig.add_trace(
            go.Scatter(
                x=self.umap_embedding[normal_mask, 0],
                y=self.umap_embedding[normal_mask, 1],
                mode='markers',
                name='Normal',
                marker=dict(size=5, color='#3498db', opacity=0.5),
                hovertemplate='<b>Normal</b><br>UMAP-1: %{x:.2f}<br>UMAP-2: %{y:.2f}<extra></extra>'
            ),
            row=1, col=3
        )
        
        # Fraud transactions (red)
        self.fig.add_trace(
            go.Scatter(
                x=self.umap_embedding[self.fraud_flags, 0],
                y=self.umap_embedding[self.fraud_flags, 1],
                mode='markers',
                name='Fraud',
                marker=dict(size=8, color='#e74c3c', opacity=0.9, symbol='star'),
                hovertemplate='<b>Fraud</b><br>UMAP-1: %{x:.2f}<br>UMAP-2: %{y:.2f}<extra></extra>'
            ),
            row=1, col=3
        )
        
        self.fig.update_xaxes(title_text='UMAP Component 1', row=1, col=3)
        self.fig.update_yaxes(title_text='UMAP Component 2', row=1, col=3)
    
    def _add_anomaly_histogram(self):
        """Add Panel 4: Anomaly score distribution histogram."""
        self.fig.add_trace(
            go.Histogram(
                x=self.anomaly_scores[self.anomaly_scores > 0],  # Only non-zero scores
                nbinsx=30,
                name='Anomaly Scores',
                marker_color='#9b59b6',
                hovertemplate='Score Range: %{x}<br>Count: %{y}<extra></extra>'
            ),
            row=2, col=1
        )
        
        self.fig.update_xaxes(title_text='Anomaly Score', row=2, col=1)
        self.fig.update_yaxes(title_text='Frequency', row=2, col=1)
    
    def _add_statistics_table(self):
        """Add Panel 5: Statistics table."""
        # Prepare table data
        metrics = [
            'Total Scanned',
            'Clusters',
            'Silhouette Score',
            'WCC Cluster 0',
            'WCC Cluster 1',
            'WCC Cluster 2',
            'Fraud Count',
            'Detection Rate (%)',
            'Mean Anomaly Score',
            'Mean Balance Discrepancy (Flagged)',
            'Mean Balance Discrepancy (Normal)'
        ]
        
        values = [
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
        
        self.fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Metric</b>', '<b>Value</b>'],
                    fill_color='#2c3e50',
                    align='left',
                    font=dict(color='white', size=12)
                ),
                cells=dict(
                    values=[metrics, values],
                    fill_color='#34495e',
                    align='left',
                    font=dict(color='white', size=11),
                    height=25
                )
            ),
            row=2, col=2
        )
    
    def _add_top_suspicious_table(self):
        """Add Panel 6: Top 10 suspicious transactions table."""
        # Get top 10 fraud transactions by anomaly score
        fraud_indices = np.where(self.fraud_flags)[0]
        fraud_scores = self.anomaly_scores[fraud_indices]
        top_10_idx = fraud_indices[np.argsort(fraud_scores)[-10:]][::-1]
        
        # Prepare table data
        rank = list(range(1, min(11, len(top_10_idx) + 1)))
        amounts = [f"${self.amounts[i]:,.2f}" for i in top_10_idx]
        discrepancies = [f"{self.balance_discrepancy[i]:.2f}" for i in top_10_idx]
        scores = [f"{self.anomaly_scores[i]:.4f}" for i in top_10_idx]
        indices = [f"#{i}" for i in top_10_idx]
        
        self.fig.add_trace(
            go.Table(
                header=dict(
                    values=['<b>Rank</b>', '<b>Tx ID</b>', '<b>Amount</b>', '<b>Balance Disc.</b>', '<b>Anomaly Score</b>'],
                    fill_color='#e74c3c',
                    align='left',
                    font=dict(color='white', size=11)
                ),
                cells=dict(
                    values=[rank, indices, amounts, discrepancies, scores],
                    fill_color='#c0392b',
                    align='left',
                    font=dict(color='white', size=10),
                    height=25
                )
            ),
            row=2, col=3
        )
    
    def _add_header_stats(self):
        """Add header stats bar with 5 key metrics."""
        total_scanned = self.stats_dict['total_transactions']
        n_clusters = self.stats_dict['num_clusters']
        silhouette = self.stats_dict['silhouette_score']
        fraud_count = self.stats_dict['fraud_count']
        detection_rate = self.stats_dict['fraud_percentage']
        
        header_text = (
            f"<b>Total Scanned:</b> {total_scanned:,} | "
            f"<b>Clusters:</b> {n_clusters} | "
            f"<b>Silhouette Score:</b> {silhouette:.4f} | "
            f"<b>Fraud Detected:</b> {fraud_count:,} | "
            f"<b>Detection Rate:</b> {detection_rate:.2f}%"
        )
        
        self.fig.add_annotation(
            text=header_text,
            xref='paper',
            yref='paper',
            x=0.5,
            y=1.08,
            showarrow=False,
            font=dict(size=13, color='white'),
            bgcolor='#2c3e50',
            bordercolor='#e74c3c',
            borderwidth=2,
            borderpad=10
        )
