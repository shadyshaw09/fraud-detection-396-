"""
Main Pipeline Runner - End-to-End Fraud Detection

Orchestrates the complete unsupervised fraud detection system:
1. Data loading and preprocessing
2. Feature engineering (novel multi-signal embedding)
3. Model training (Isolation Forest + Autoencoder)
4. Ensemble scoring
5. Evaluation using labels (unseen by models)
6. Comprehensive visualization and reporting
"""

import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
import time
from typing import Tuple, Dict

# Import custom modules
from src.preprocessing import DataPreprocessor, create_synthetic_paysim_data
from src.feature_engineering import FeatureEngineer
from src.models.isolation_forest import IsolationForestModel
from src.models.autoencoder import AutoencoderModel
from src.models.ensemble import EnsembleAnomalyScorer
from src.visualizations import FraudVisualizationEngine


def print_section(title: str) -> None:
    """Print formatted section header."""
    print("\n" + "="*80)
    print(f"  {title}")
    print("="*80 + "\n")


def load_data(data_path: str = 'fraud_detection/data/paysim.csv') -> pd.DataFrame:
    """
    Load transaction data from CSV or generate synthetic.
    
    Args:
        data_path: Path to CSV file
        
    Returns:
        DataFrame with transaction data
    """
    print_section("STEP 1: DATA LOADING")
    
    data_path = Path(data_path)
    
    if data_path.exists():
        print(f"[PIPELINE] Loading real data from {data_path}...")
        df = pd.read_csv(data_path)
    else:
        print(f"[PIPELINE] Real data not found at {data_path}")
        print(f"[PIPELINE] Generating synthetic PaySim data...")
        df = create_synthetic_paysim_data(n_samples=10000, random_state=42)
        
        # Save for future use
        data_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(data_path, index=False)
        print(f"[PIPELINE] Synthetic data saved to {data_path}")
    
    print(f"\n[PIPELINE] Dataset loaded: {df.shape[0]} transactions, {df.shape[1]} columns")
    print(f"[PIPELINE] Columns: {list(df.columns)}")
    
    return df


def preprocess_data(df: pd.DataFrame) -> Tuple[pd.DataFrame, DataPreprocessor]:
    """
    Preprocess transaction data.
    
    Args:
        df: Raw transaction DataFrame
        
    Returns:
        Tuple of (processed_df, preprocessor)
    """
    print_section("STEP 2: DATA PREPROCESSING")
    
    preprocessor = DataPreprocessor(random_state=42)
    df_processed = preprocessor.fit_and_preprocess(df)
    
    print(f"\n[PIPELINE] Preprocessing complete")
    print(f"[PIPELINE] Output shape: {df_processed.shape}")
    
    return df_processed, preprocessor


def engineer_features(df: pd.DataFrame) -> Tuple[np.ndarray, list, FeatureEngineer]:
    """
    Engineer novel fraud detection features.
    
    Args:
        df: Preprocessed transaction DataFrame
        
    Returns:
        Tuple of (feature_embedding, feature_names, engineer)
    """
    print_section("STEP 3: NOVEL FEATURE ENGINEERING (PATENT-WORTHY)")
    
    engineer = FeatureEngineer()
    embedding, feature_names = engineer.fit_and_engineer_features(df)
    
    print(f"\n[PIPELINE] Features engineered successfully")
    print(f"[PIPELINE] Embedding shape: {embedding.shape}")
    print(f"[PIPELINE] Features: {feature_names}")
    
    return embedding, feature_names, engineer


def train_isolation_forest(X: np.ndarray) -> IsolationForestModel:
    """
    Train Isolation Forest anomaly detector.
    
    Args:
        X: Feature embedding (n_samples, n_features)
        
    Returns:
        Trained IsolationForestModel
    """
    print_section("STEP 4A: TRAINING ISOLATION FOREST")
    
    model = IsolationForestModel(contamination=0.05, n_estimators=100, random_state=42)
    model.fit(X)
    
    print(f"\n[PIPELINE] Isolation Forest trained")
    
    return model


def train_autoencoder(X: np.ndarray) -> AutoencoderModel:
    """
    Train Autoencoder anomaly detector.
    
    Args:
        X: Feature embedding (n_samples, n_features)
        
    Returns:
        Trained AutoencoderModel
    """
    print_section("STEP 4B: TRAINING AUTOENCODER")
    
    model = AutoencoderModel(
        input_dim=X.shape[1],
        latent_dim=4,
        learning_rate=0.001,
        batch_size=32,
        epochs=50,
        device='cpu',
        verbose=True
    )
    model.fit(X, validation_split=0.1)
    
    print(f"\n[PIPELINE] Autoencoder trained")
    
    return model


def score_anomalies(
    X: np.ndarray,
    iso_forest_model: IsolationForestModel,
    autoencoder_model: AutoencoderModel
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, EnsembleAnomalyScorer]:
    """
    Score transactions using both models and ensemble.
    
    Args:
        X: Feature embedding
        iso_forest_model: Trained IF model
        autoencoder_model: Trained AE model
        
    Returns:
        Tuple of (if_scores, ae_scores, ensemble_scores, scorer)
    """
    print_section("STEP 5: ENSEMBLE ANOMALY SCORING")
    
    # Get scores from each model
    print("[PIPELINE] Scoring with Isolation Forest...")
    if_pred, if_scores = iso_forest_model.predict_and_score(X)
    
    print("[PIPELINE] Scoring with Autoencoder...")
    ae_pred, ae_scores = autoencoder_model.predict_and_score(X)
    
    # Combine with ensemble
    print("[PIPELINE] Computing ensemble scores...")
    scorer = EnsembleAnomalyScorer(
        iso_forest_weight=0.5,
        autoencoder_weight=0.5,
        combination_method='weighted_mean'
    )
    ensemble_scores, scores_dict = scorer.combine_scores(
        if_scores, ae_scores,
        if_pred, ae_pred
    )
    
    print(f"\n[PIPELINE] Scoring complete")
    
    return if_scores, ae_scores, ensemble_scores, scorer


def evaluate_results(
    ensemble_scores: np.ndarray,
    df_original: pd.DataFrame,
    output_dir: str = 'fraud_detection/outputs'
) -> Dict:
    """
    Evaluate results using true labels (ground truth).
    
    Args:
        ensemble_scores: Anomaly scores from ensemble
        df_original: Original DataFrame with isFraud labels
        output_dir: Output directory for visualizations
        
    Returns:
        Dictionary with evaluation metrics
    """
    print_section("STEP 6: EVALUATION & VISUALIZATION")
    
    # Get true labels
    true_labels = df_original['isFraud'].values if 'isFraud' in df_original.columns else None
    
    if true_labels is None:
        print("[PIPELINE] Warning: No true labels available for evaluation")
        return {}
    
    # Initialize visualization engine
    viz = FraudVisualizationEngine(output_dir=output_dir)
    
    # 1. Plot score distribution
    print("[PIPELINE] Creating visualizations...")
    viz.plot_anomaly_score_distribution(ensemble_scores, true_labels)
    
    # 2. Plot Precision-Recall curve
    auroc, ap_score = viz.plot_precision_recall_curve(ensemble_scores, true_labels)
    
    # 3. Find threshold at 95th percentile
    threshold_95, predictions_95 = np.percentile(ensemble_scores, 95), None
    # Get predictions at threshold
    predictions_95 = np.where(ensemble_scores > threshold_95, 1, -1)
    
    # 4. Plot confusion matrix
    metrics = viz.plot_confusion_matrix(predictions_95, true_labels, threshold_95)
    
    # 5. Compare models
    if_scores = IsolationForestModel().get_anomaly_scores(X) if hasattr(IsolationForestModel(), 'is_fitted') else None
    ae_scores = AutoencoderModel(X.shape[1]).get_anomaly_scores(X) if hasattr(AutoencoderModel(X.shape[1]), 'is_fitted') else None
    
    # Compile results
    results = {
        'auroc': auroc,
        'ap_score': ap_score,
        'metrics': metrics,
        'ensemble_config': {
            'iso_forest_weight': 0.5,
            'autoencoder_weight': 0.5,
            'combination_method': 'weighted_mean',
        },
        'dataset_stats': {
            'n_transactions': len(ensemble_scores),
            'fraud_rate': (true_labels == 1).sum() / len(true_labels) * 100,
            'n_features': 'See feature engineering',
        }
    }
    
    # Create summary report
    viz.create_summary_report(results)
    
    print(f"\n[PIPELINE] Evaluation complete")
    print(f"[PIPELINE] All visualizations saved to {output_dir}/")
    
    return results


def print_final_summary(results: Dict) -> None:
    """Print final summary of results."""
    print_section("FINAL SUMMARY - FRAUD DETECTION RESULTS")
    
    print("✓ UNSUPERVISED SYSTEM - No fraud labels used during training")
    print("✓ Novel approach: Multi-signal risk embedding combining 6 engineered features")
    print("✓ Ensemble of Isolation Forest + Autoencoder")
    print("\nEVALUATION RESULTS (on held-out ground truth):")
    print(f"  - AUROC: {results.get('auroc', 'N/A'):.4f}")
    print(f"  - AP Score: {results.get('ap_score', 'N/A'):.4f}")
    
    if 'metrics' in results:
        metrics = results['metrics']
        print(f"\n  At 95th percentile threshold:")
        print(f"    - Precision: {metrics.get('precision', 'N/A'):.4f}")
        print(f"    - Recall: {metrics.get('recall', 'N/A'):.4f}")
        print(f"    - F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
        print(f"    - TP: {metrics.get('tp', 'N/A')}, FP: {metrics.get('fp', 'N/A')}")
        print(f"    - FN: {metrics.get('fn', 'N/A')}, TN: {metrics.get('tn', 'N/A')}")
    
    print("\nOUTPUT FILES:")
    print("  - 01_anomaly_score_distribution.png")
    print("  - 02_precision_recall_curve.png")
    print("  - 03_confusion_matrix.png")
    print("  - 04_model_comparison.png")
    print("  - 05_score_comparison_scatter.png")
    print("  - 06_feature_importance.png")
    print("  - 07_summary_report.txt")


def main():
    """Main pipeline execution."""
    start_time = time.time()
    
    print("\n" + "="*80)
    print("  FRAUDULENT TRANSACTION ANOMALY DETECTION")
    print("  Unsupervised Machine Learning System")
    print("="*80 + "\n")
    
    try:
        # Step 1: Load data
        df = load_data()
        
        # Step 2: Preprocess
        df_processed, preprocessor = preprocess_data(df)
        
        # Step 3: Engineer features
        X_embedding, feature_names, engineer = engineer_features(df_processed)
        
        # Step 4: Train models
        iso_forest_model = train_isolation_forest(X_embedding)
        autoencoder_model = train_autoencoder(X_embedding)
        
        # Step 5: Score anomalies
        if_scores, ae_scores, ensemble_scores, scorer = score_anomalies(
            X_embedding, iso_forest_model, autoencoder_model
        )
        
        # Step 6: Evaluate and visualize
        results = evaluate_results(ensemble_scores, df)
        
        # Print final summary
        print_final_summary(results)
        
        elapsed_time = time.time() - start_time
        print(f"\n[PIPELINE] Total execution time: {elapsed_time:.2f} seconds")
        print("\n" + "="*80)
        print("  ✓ PIPELINE COMPLETE")
        print("="*80 + "\n")
        
    except Exception as e:
        print(f"\n[ERROR] Pipeline failed: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
