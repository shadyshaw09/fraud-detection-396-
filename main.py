#!/usr/bin/env python3
"""
K-Means + Isolation Forest Fraud Detection POC

Complete pipeline orchestrator:
  1. Load 50K transactions and filter to TRANSFER/CASH_OUT
  2. Engineer 3 features (balance_discrepancy, zero_balance_flag, amount_log)
  3. Run K-Means clustering with k=3 (Elbow Method k=2-8)
  4. Identify suspicious cluster (highest mean balance_discrepancy)
  5. Run Isolation Forest on suspicious cluster only
  6. Generate comprehensive 6-panel plotly dashboard
  7. Print all statistics

Usage:
    cd fraud_detection
    python main.py
"""

import sys
import os
import webbrowser
from pathlib import Path

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# Import modules
from src.data_loader import DataLoader
from src.feature_engineer import FeatureEngineer
from src.clustering import KMeansAnalyzer
from src.anomaly_detection import IsolationForestAnalyzer
from src.statistics import StatisticsCalculator
from src.dashboard_builder import DashboardBuilder


def run_fraud_detection_pipeline():
    """
    Run complete fraud detection pipeline with all 7 phases.
    """
    print("\n" + "="*70)
    print("🚨 K-MEANS + ISOLATION FOREST FRAUD DETECTION POC")
    print("="*70)
    
    try:
        # ===== PHASE 1: DATA LOADING =====
        print("\n[Phase 1/7] 📥 Loading and Filtering Data")
        print("-" * 70)
        
        csv_path = os.path.join(project_root, 'data', 'paysim.csv')
        data_loader = DataLoader(csv_path, n_rows=50000)
        df = data_loader.load_and_filter(types=['TRANSFER', 'CASH_OUT'])
        
        print(f"✓ Data ready: {len(df):,} transactions loaded and filtered")
        
        # ===== PHASE 2: FEATURE ENGINEERING =====
        print("\n[Phase 2/7] ⚙️  Feature Engineering")
        print("-" * 70)
        
        feature_engineer = FeatureEngineer(df)
        feature_df = feature_engineer.engineer_features()
        feature_matrix = feature_engineer.get_feature_matrix()
        balance_discrepancy = feature_df['balance_discrepancy'].values
        
        print(f"✓ Features engineered: {feature_matrix.shape}")
        
        # ===== PHASE 3: K-MEANS CLUSTERING =====
        print("\n[Phase 3/7] 🎯 K-Means Clustering Analysis")
        print("-" * 70)
        
        kmeans_analyzer = KMeansAnalyzer(
            features=feature_matrix,
            balance_discrepancy=balance_discrepancy,
            k_final=3
        )
        
        cluster_labels, silhouette_score_val, wcc_values, suspicious_cluster_id, inertia_values = \
            kmeans_analyzer.run_complete_analysis()
        
        print(f"✓ Clustering complete: 3 clusters formed")
        
        # ===== PHASE 4: ANOMALY DETECTION =====
        print("\n[Phase 4/7] 🔍 Isolation Forest Anomaly Detection")
        print("-" * 70)
        
        suspicious_mask = kmeans_analyzer.get_suspicious_cluster_mask()
        
        iso_forest_analyzer = IsolationForestAnalyzer(
            features=feature_matrix,
            suspicious_mask=suspicious_mask
        )
        
        anomaly_scores = iso_forest_analyzer.run_isolation_forest(contamination=0.02)
        fraud_flags = iso_forest_analyzer.get_fraud_flags()
        fraud_count = iso_forest_analyzer.get_fraud_count()
        fraud_percentage = iso_forest_analyzer.get_fraud_percentage()
        mean_anomaly_score_flagged = iso_forest_analyzer.get_flagged_anomaly_scores_mean()
        
        print(f"✓ Anomaly detection complete: {fraud_count:,} frauds flagged")
        
        # ===== PHASE 5: STATISTICS CALCULATION =====
        print("\n[Phase 5/7] 📊 Computing Statistics")
        print("-" * 70)
        
        stats_calculator = StatisticsCalculator(
            n_total=len(df),
            n_clusters=3,
            silhouette_score=silhouette_score_val,
            wcc_values=wcc_values,
            fraud_count=fraud_count,
            fraud_percentage=fraud_percentage,
            mean_anomaly_score_flagged=mean_anomaly_score_flagged,
            balance_discrepancy=balance_discrepancy,
            fraud_flags=fraud_flags
        )
        
        stats_dict = stats_calculator.calculate_all_stats()
        stats_calculator.print_statistics()
        
        print(f"✓ Statistics calculated and displayed")
        
        # ===== PHASE 6: DASHBOARD GENERATION =====
        print("\n[Phase 6/7] 📈 Building Dashboard")
        print("-" * 70)
        
        # Ensure outputs directory exists
        outputs_dir = os.path.join(project_root, 'outputs')
        Path(outputs_dir).mkdir(exist_ok=True)
        
        dashboard_builder = DashboardBuilder(
            features=feature_matrix,
            cluster_labels=cluster_labels,
            inertia_values=inertia_values,
            anomaly_scores=anomaly_scores,
            fraud_flags=fraud_flags,
            balance_discrepancy=balance_discrepancy,
            amounts=df['amount'].values,
            stats_dict=stats_dict,
            k_range=list(kmeans_analyzer.k_range)
        )
        
        dashboard_path = os.path.join(outputs_dir, 'dashboard.html')
        dashboard_builder.build_dashboard(output_path=dashboard_path)
        
        print(f"✓ Dashboard created: {dashboard_path}")
        
        # ===== PHASE 7: BROWSER LAUNCH =====
        print("\n[Phase 7/7] 🌐 Opening Dashboard in Browser")
        print("-" * 70)
        
        dashboard_url = f"file:///{os.path.abspath(dashboard_path).replace(chr(92), '/')}"
        webbrowser.open(dashboard_url)
        
        print(f"✓ Dashboard opened in browser")
        print(f"  URL: {dashboard_url}")
        
        # ===== COMPLETION =====
        print("\n" + "="*70)
        print("✅ FRAUD DETECTION PIPELINE COMPLETE")
        print("="*70)
        print(f"\n📊 Summary:")
        print(f"  Transactions Scanned: {stats_dict['total_transactions']:,}")
        print(f"  Clusters Formed: {stats_dict['num_clusters']}")
        print(f"  Fraud Detected: {stats_dict['fraud_count']:,} ({stats_dict['fraud_percentage']:.2f}%)")
        print(f"  Dashboard: {dashboard_path}")
        print("\n")
        
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    run_fraud_detection_pipeline()
