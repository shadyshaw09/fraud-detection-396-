#!/usr/bin/env python3
"""
Setup Verification Script

Checks that all dependencies are installed and project structure is correct.
"""

import sys
from pathlib import Path

def check_structure():
    """Check that project structure is correct."""
    print("\n📁 Checking project structure...")
    
    required_dirs = [
        'data',
        'notebooks',
        'outputs',
        'src/models',
    ]
    
    required_files = [
        'main.py',
        'requirements.txt',
        'README.md',
        'QUICKSTART.md',
        'src/__init__.py',
        'src/preprocessing.py',
        'src/feature_engineering.py',
        'src/pipeline.py',
        'src/visualizations.py',
        'src/models/__init__.py',
        'src/models/isolation_forest.py',
        'src/models/autoencoder.py',
        'src/models/ensemble.py',
        'notebooks/eda_analysis.ipynb',
    ]
    
    project_root = Path(__file__).parent
    
    all_good = True
    
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✓ {dir_path}/")
        else:
            print(f"  ✗ {dir_path}/ (MISSING)")
            all_good = False
    
    for file_path in required_files:
        full_path = project_root / file_path
        if full_path.exists():
            print(f"  ✓ {file_path}")
        else:
            print(f"  ✗ {file_path} (MISSING)")
            all_good = False
    
    return all_good


def check_dependencies():
    """Check that required packages are installed."""
    print("\n📦 Checking dependencies...")
    
    required_packages = {
        'pandas': 'Data manipulation',
        'numpy': 'Numerical computing',
        'sklearn': 'Scikit-learn - Machine learning',
        'torch': 'PyTorch - Deep learning',
        'umap': 'UMAP - Dimensionality reduction',
        'matplotlib': 'Plotting',
        'seaborn': 'Statistical visualization',
        'plotly': 'Interactive plots',
    }
    
    all_good = True
    
    for package, description in required_packages.items():
        try:
            __import__(package)
            print(f"  ✓ {package:20} - {description}")
        except ImportError:
            print(f"  ✗ {package:20} - {description} (NOT INSTALLED)")
            all_good = False
    
    return all_good


def main():
    """Run all checks."""
    print("="*70)
    print("  FRAUD DETECTION - SETUP VERIFICATION")
    print("="*70)
    
    structure_ok = check_structure()
    deps_ok = check_dependencies()
    
    print("\n" + "="*70)
    
    if structure_ok and deps_ok:
        print("\n✓ ALL CHECKS PASSED!")
        print("\nYou're ready to run:")
        print("  python main.py")
        return 0
    else:
        print("\n✗ SOME CHECKS FAILED")
        if not deps_ok:
            print("\nTo install dependencies:")
            print("  pip install -r requirements.txt")
        return 1


if __name__ == '__main__':
    sys.exit(main())
