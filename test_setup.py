#!/usr/bin/env python3
"""Test script to diagnose the pipeline"""

import os
import sys

# Add to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

print("="*70)
print("Testing Fraud Detection Pipeline Setup")
print("="*70)

# Test 1: Import all modules
print("\n[Test 1] Importing modules...")
try:
    from src.data_loader import DataLoader
    print("  ✓ DataLoader")
    from src.feature_engineer import FeatureEngineer
    print("  ✓ FeatureEngineer")
    from src.clustering import KMeansAnalyzer
    print("  ✓ KMeansAnalyzer")
    from src.anomaly_detection import IsolationForestAnalyzer
    print("  ✓ IsolationForestAnalyzer")
    from src.statistics import StatisticsCalculator
    print("  ✓ StatisticsCalculator")
    from src.dashboard_builder import DashboardBuilder
    print("  ✓ DashboardBuilder")
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import error: {e}")
    sys.exit(1)

# Test 2: Check data file
print("\n[Test 2] Checking data file...")
csv_path = os.path.join(project_root, 'data', 'paysim.csv')
print(f"  Looking for: {csv_path}")
if os.path.exists(csv_path):
    print(f"  ✓ File exists")
    import pandas as pd
    df = pd.read_csv(csv_path, nrows=10)
    print(f"  ✓ Can read file, columns: {list(df.columns)}")
else:
    print(f"  ✗ File not found")
    sys.exit(1)

# Test 3: Load and filter data
print("\n[Test 3] Loading and filtering data...")
try:
    data_loader = DataLoader(csv_path, n_rows=1000)  # Use 1000 for quick test
    df = data_loader.load_and_filter(types=['TRANSFER', 'CASH_OUT'])
    print(f"  ✓ Loaded {len(df)} rows")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

# Test 4: Engineer features
print("\n[Test 4] Engineering features...")
try:
    feature_engineer = FeatureEngineer(df)
    features = feature_engineer.engineer_features()
    print(f"  ✓ Engineered features shape: {features.shape}")
except Exception as e:
    print(f"  ✗ Error: {e}")
    sys.exit(1)

print("\n" + "="*70)
print("✓ All tests passed - pipeline is ready!")
print("="*70)
print("\nYou can now run the full pipeline with:")
print("  python main.py")
