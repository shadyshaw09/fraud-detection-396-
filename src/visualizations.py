"""
Visualization Module for Fraud Detection Results

Creates comprehensive visualizations:
- UMAP 2D embeddings with anomaly coloring
- Anomaly score distributions
- Precision-Recall curves
- Confusion matrices
- Feature importance heatmaps
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import precision_recall_curve, roc_auc_score, confusion_matrix, roc_curve, auc
from sklearn.metrics import classification_report
from pathlib import Path
from typing import Optional, Tuple, Dict
import warnings

warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 10


class FraudVisualizationEngine:
    """
    Comprehensive visualization suite for fraud detection results.
    """
    
    def __init__(self, output_dir: str = 'outputs'):
        """
        Initialize visualization engine.
        
        Args:
            output_dir: Directory to save plots
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        print(f"[VIZ] Output directory: {self.output_dir}")
    
    def plot_anomaly_score_distribution(
        self,
        scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        title: str = "Anomaly Score Distribution",
        filename: str = "01_anomaly_score_distribution.png"
    ) -> None:
        """
        Plot distribution of anomaly scores.
        
        Args:
            scores: Array of anomaly scores
            labels: Optional true labels (1=fraud, 0=normal)
            title: Plot title
            filename: Output filename
        """
        print(f"[VIZ] Creating anomaly score distribution plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Histogram
        axes[0].hist(scores, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        axes[0].set_xlabel('Anomaly Score')
        axes[0].set_ylabel('Frequency')
        axes[0].set_title('Anomaly Score Distribution')
        axes[0].grid(True, alpha=0.3)
        
        # Separated by class if labels provided
        if labels is not None:
            axes[1].hist(scores[labels == 0], bins=50, alpha=0.6, label='Normal', color='green')
            axes[1].hist(scores[labels == 1], bins=50, alpha=0.6, label='Fraud', color='red')
            axes[1].set_xlabel('Anomaly Score')
            axes[1].set_ylabel('Frequency')
            axes[1].set_title('Score Distribution by Class')
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Saved: {filepath}")
        plt.close()
    
    def plot_precision_recall_curve(
        self,
        scores: np.ndarray,
        labels: np.ndarray,
        filename: str = "02_precision_recall_curve.png"
    ) -> Tuple[float, float]:
        """
        Plot Precision-Recall curve and compute AUROC.
        
        Args:
            scores: Array of anomaly scores
            labels: True labels (1=fraud, 0=normal)
            filename: Output filename
            
        Returns:
            Tuple of (auroc, ap_score)
        """
        print(f"[VIZ] Creating Precision-Recall curve...")
        
        # Convert to binary labels if needed
        y_true = (labels == 1).astype(int)
        
        precision, recall, thresholds = precision_recall_curve(y_true, scores)
        auroc = roc_auc_score(y_true, scores)
        
        # Compute AP score
        ap_score = np.mean([precision[i] for i in range(len(precision)-1) 
                           if recall[i] != recall[i+1]])
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Precision-Recall curve
        axes[0].plot(recall, precision, linewidth=2, label=f'AP: {ap_score:.3f}')
        axes[0].set_xlabel('Recall')
        axes[0].set_ylabel('Precision')
        axes[0].set_title('Precision-Recall Curve')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        axes[0].set_xlim([0, 1])
        axes[0].set_ylim([0, 1])
        
        # ROC curve
        fpr, tpr, _ = roc_curve(y_true, scores)
        roc_auc = auc(fpr, tpr)
        axes[1].plot(fpr, tpr, linewidth=2, label=f'ROC AUC: {roc_auc:.3f}')
        axes[1].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
        axes[1].set_xlabel('False Positive Rate')
        axes[1].set_ylabel('True Positive Rate')
        axes[1].set_title('ROC Curve')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        axes[1].set_xlim([0, 1])
        axes[1].set_ylim([0, 1])
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Saved: {filepath}")
        print(f"[VIZ] AUROC: {auroc:.4f}, AP: {ap_score:.4f}")
        plt.close()
        
        return auroc, ap_score
    
    def plot_confusion_matrix(
        self,
        predictions: np.ndarray,
        labels: np.ndarray,
        threshold: float,
        filename: str = "03_confusion_matrix.png"
    ) -> Dict[str, float]:
        """
        Plot confusion matrix and compute metrics.
        
        Args:
            predictions: Binary predictions (-1 or 1)
            labels: True labels (1=fraud, 0 or -1=normal)
            threshold: Threshold used for predictions
            filename: Output filename
            
        Returns:
            Dictionary with performance metrics
        """
        print(f"[VIZ] Creating confusion matrix...")
        
        # Convert to binary (1 = fraud, 0 = normal)
        y_pred = (predictions == 1).astype(int)
        y_true = (labels == 1).astype(int)
        
        # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        
        # Compute metrics
        tn, fp, fn, tp = cm.ravel()
        fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
        fnr = fn / (fn + tp) if (fn + tp) > 0 else 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        metrics = {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'fpr': float(fpr),
            'fnr': float(fnr),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'threshold': float(threshold),
        }
        
        # Plot confusion matrix
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0],
                   xticklabels=['Normal', 'Fraud'],
                   yticklabels=['Normal', 'Fraud'])
        axes[0].set_ylabel('True Label')
        axes[0].set_xlabel('Predicted Label')
        axes[0].set_title(f'Confusion Matrix (threshold={threshold:.3f})')
        
        # Metrics table
        axes[1].axis('tight')
        axes[1].axis('off')
        
        metrics_text = f"""
        PERFORMANCE METRICS
        ════════════════════════════════
        
        True Positives (TP):      {tp:>10}
        True Negatives (TN):      {tn:>10}
        False Positives (FP):     {fp:>10}
        False Negatives (FN):     {fn:>10}
        
        ════════════════════════════════
        Precision:                {precision:>10.4f}
        Recall (Sensitivity):     {recall:>10.4f}
        False Positive Rate:      {fpr:>10.4f}
        False Negative Rate:      {fnr:>10.4f}
        F1 Score:                 {f1:>10.4f}
        ════════════════════════════════
        """
        
        axes[1].text(0.1, 0.5, metrics_text, fontsize=11, family='monospace',
                    verticalalignment='center', transform=axes[1].transAxes)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Saved: {filepath}")
        print(f"[VIZ] Precision: {precision:.4f}, Recall: {recall:.4f}, F1: {f1:.4f}")
        plt.close()
        
        return metrics
    
    def plot_model_comparison(
        self,
        iso_forest_scores: np.ndarray,
        autoencoder_scores: np.ndarray,
        ensemble_scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        filename: str = "04_model_comparison.png"
    ) -> None:
        """
        Compare anomaly scores from different models.
        
        Args:
            iso_forest_scores: IF anomaly scores
            autoencoder_scores: AE anomaly scores
            ensemble_scores: Ensemble anomaly scores
            labels: Optional true labels
            filename: Output filename
        """
        print(f"[VIZ] Creating model comparison plot...")
        
        fig, axes = plt.subplots(1, 3, figsize=(16, 5))
        
        models = [
            ('Isolation Forest', iso_forest_scores),
            ('Autoencoder', autoencoder_scores),
            ('Ensemble', ensemble_scores)
        ]
        
        for idx, (model_name, scores) in enumerate(models):
            if labels is not None:
                axes[idx].hist(scores[labels == 0], bins=40, alpha=0.6, label='Normal', color='green')
                axes[idx].hist(scores[labels == 1], bins=40, alpha=0.6, label='Fraud', color='red')
            else:
                axes[idx].hist(scores, bins=50, alpha=0.7, color='steelblue')
            
            axes[idx].set_xlabel('Anomaly Score')
            axes[idx].set_ylabel('Frequency')
            axes[idx].set_title(f'{model_name} Score Distribution')
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Saved: {filepath}")
        plt.close()
    
    def plot_score_comparison_scatter(
        self,
        iso_forest_scores: np.ndarray,
        autoencoder_scores: np.ndarray,
        ensemble_scores: np.ndarray,
        labels: Optional[np.ndarray] = None,
        filename: str = "05_score_comparison_scatter.png"
    ) -> None:
        """
        Scatter plot comparing different model scores.
        
        Args:
            iso_forest_scores: IF anomaly scores
            autoencoder_scores: AE anomaly scores
            ensemble_scores: Ensemble anomaly scores
            labels: Optional true labels for coloring
            filename: Output filename
        """
        print(f"[VIZ] Creating score comparison scatter plot...")
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        if labels is not None:
            colors = ['green' if l == 0 else 'red' for l in labels]
        else:
            colors = ensemble_scores
        
        # IF vs AE
        axes[0].scatter(iso_forest_scores, autoencoder_scores, c=colors, alpha=0.5, s=20)
        axes[0].set_xlabel('Isolation Forest Score')
        axes[0].set_ylabel('Autoencoder Score')
        axes[0].set_title('Model Score Comparison')
        axes[0].grid(True, alpha=0.3)
        
        # IF vs Ensemble
        axes[1].scatter(iso_forest_scores, ensemble_scores, c=colors, alpha=0.5, s=20)
        axes[1].set_xlabel('Isolation Forest Score')
        axes[1].set_ylabel('Ensemble Score')
        axes[1].set_title('IF Score vs Ensemble Score')
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Saved: {filepath}")
        plt.close()
    
    def plot_feature_importance(
        self,
        feature_names: list,
        anomaly_samples: pd.DataFrame,
        normal_samples: pd.DataFrame,
        filename: str = "06_feature_importance.png"
    ) -> None:
        """
        Plot feature importance based on anomaly vs normal distribution.
        
        Args:
            feature_names: Names of features (signal names)
            anomaly_samples: DataFrame/array of anomalous transaction features
            normal_samples: DataFrame/array of normal transaction features
            filename: Output filename
        """
        print(f"[VIZ] Creating feature importance heatmap...")
        
        # Compute mean values
        anomaly_mean = anomaly_samples.mean() if isinstance(anomaly_samples, pd.DataFrame) else anomaly_samples.mean(axis=0)
        normal_mean = normal_samples.mean() if isinstance(normal_samples, pd.DataFrame) else normal_samples.mean(axis=0)
        
        # Create comparison
        comparison = pd.DataFrame({
            'Normal': normal_mean,
            'Anomaly': anomaly_mean
        })
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Bar plot
        comparison.plot(kind='bar', ax=axes[0], color=['green', 'red'], alpha=0.7)
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Mean Value')
        axes[0].set_title('Feature Values: Normal vs Anomaly')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        # Heatmap
        heatmap_data = np.array([normal_mean, anomaly_mean])
        sns.heatmap(heatmap_data, annot=True, fmt='.3f', cmap='RdYlGn_r', ax=axes[1],
                   xticklabels=feature_names, yticklabels=['Normal', 'Anomaly'])
        axes[1].set_title('Feature Importance Heatmap')
        
        plt.tight_layout()
        filepath = self.output_dir / filename
        plt.savefig(filepath, dpi=300, bbox_inches='tight')
        print(f"[VIZ] Saved: {filepath}")
        plt.close()
    
    def create_summary_report(
        self,
        results: Dict,
        filename: str = "07_summary_report.txt"
    ) -> None:
        """
        Create text summary report of results.
        
        Args:
            results: Dictionary with all results
            filename: Output filename
        """
        print(f"[VIZ] Creating summary report...")
        
        filepath = self.output_dir / filename
        
        with open(filepath, 'w') as f:
            f.write("=" * 80 + "\n")
            f.write("FRAUD DETECTION - COMPREHENSIVE ANALYSIS REPORT\n")
            f.write("=" * 80 + "\n\n")
            
            f.write("PROJECT SUMMARY\n")
            f.write("-" * 80 + "\n")
            f.write("System: Unsupervised Multi-Signal Anomaly Detection for Fraud\n")
            f.write("Novel Approach: Multi-signal risk embedding combining 6 engineered features\n")
            f.write("Models: Isolation Forest + Autoencoder Ensemble\n\n")
            
            f.write("DATASET STATISTICS\n")
            f.write("-" * 80 + "\n")
            if 'dataset_stats' in results:
                for key, value in results['dataset_stats'].items():
                    f.write(f"{key:.<40} {value}\n")
            f.write("\n")
            
            f.write("ENGINEERED FEATURES (NOVEL)\n")
            f.write("-" * 80 + "\n")
            features_desc = [
                "1. Balance Discrepancy Ratio - Mathematical inconsistencies in balance changes",
                "2. Transaction Velocity Score - Frequency of transactions from same origin",
                "3. Amount Deviation Score - Behavioral anomalies in transaction size",
                "4. Zero Balance Flag - Accounts drained after transaction",
                "5. Round Amount Suspicion - Unnatural suspiciously round amounts",
                "6. Destination Account Age Proxy - Account maturity estimation"
            ]
            for desc in features_desc:
                f.write(f"{desc}\n")
            f.write("\n")
            
            f.write("MODEL PERFORMANCE\n")
            f.write("-" * 80 + "\n")
            if 'metrics' in results:
                for key, value in results['metrics'].items():
                    if isinstance(value, float):
                        f.write(f"{key:.<40} {value:.4f}\n")
                    else:
                        f.write(f"{key:.<40} {value}\n")
            f.write("\n")
            
            f.write("ENSEMBLE CONFIGURATION\n")
            f.write("-" * 80 + "\n")
            if 'ensemble_config' in results:
                for key, value in results['ensemble_config'].items():
                    f.write(f"{key:.<40} {value}\n")
            f.write("\n")
            
            f.write("CONCLUSIONS\n")
            f.write("-" * 80 + "\n")
            f.write("✓ Unsupervised system requires NO fraud labels for training\n")
            f.write("✓ Multi-signal approach captures diverse fraud patterns\n")
            f.write("✓ Ensemble combines strengths of different algorithms\n")
            f.write("✓ Results validated against ground truth labels (evaluation only)\n")
            f.write("\n")
        
        print(f"[VIZ] Saved: {filepath}")
