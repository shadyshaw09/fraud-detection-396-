# Fraudulent Transaction Anomaly Detection
## Unsupervised Machine Learning System for Digital Payment Fraud Detection

### 🎯 Project Goal
Detect fraudulent transactions **WITHOUT any labeled training data** using unsupervised anomaly detection techniques with patent-worthy novelty.

### 🔬 Novel Innovation: Multi-Signal Risk Embedding
The core innovation is engineering 6 novel fraud signals and combining them into a unified risk embedding:

1. **Balance Discrepancy Ratio** - Detects mathematical inconsistencies in account balance changes
2. **Transaction Velocity Score** - Flags suspicious transaction frequency patterns
3. **Amount Deviation Score** - Identifies behavioral anomalies in transaction sizes
4. **Zero Balance Flag** - Detects accounts that are drained after transactions
5. **Round Amount Suspicion** - Identifies unnatural suspiciously round transaction amounts
6. **Destination Account Age Proxy** - Estimates account maturity and trustworthiness

These signals are combined into a **multi-signal risk embedding vector** that captures diverse fraud patterns without requiring labeled data.

### 🏗️ Architecture

```
fraud_detection/
├── data/                   # Raw and processed data
│   └── paysim.csv         # PaySim synthetic dataset
├── notebooks/              # EDA and experimentation
│   └── eda_analysis.ipynb
├── src/
│   ├── preprocessing.py    # Data cleaning and normalization
│   ├── feature_engineering.py  # NOVEL: Multi-signal risk embedding
│   ├── models/
│   │   ├── isolation_forest.py    # Tree-based anomaly detection
│   │   ├── autoencoder.py         # Neural network reconstruction
│   │   └── ensemble.py             # Weighted ensemble scoring
│   ├── visualizations.py   # Comprehensive result visualizations
│   └── pipeline.py         # End-to-end orchestration
├── outputs/                # Results, plots, reports
├── main.py                 # Single-command entry point
├── requirements.txt        # Dependencies
└── README.md
```

### 🚀 Quick Start

#### 1. Install Dependencies
```bash
cd fraud_detection
pip install -r requirements.txt
```

#### 2. Run Complete Pipeline
```bash
python main.py
```

This executes the entire workflow:
- ✓ Loads/generates PaySim dataset
- ✓ Preprocesses and normalizes data
- ✓ Engineers novel fraud signals
- ✓ Trains Isolation Forest (unsupervised)
- ✓ Trains Autoencoder (unsupervised)
- ✓ Computes ensemble anomaly scores
- ✓ Evaluates using ground truth labels (held-out)
- ✓ Creates comprehensive visualizations
- ✓ Generates summary report

### 📊 Key Features

#### Unsupervised Learning
- ✓ **No fraud labels required during training**
- System learns "normality" from data distribution
- Anomalies are detected as deviations from learned patterns

#### Dual-Model Ensemble
- **Isolation Forest**: Excels at high-dimensional outlier detection
- **Autoencoder**: Captures reconstruction anomalies
- **Ensemble**: Combines strengths through weighted voting

#### Comprehensive Evaluation
Despite being unsupervised, system is evaluated on held-out labels:
- Precision-Recall curves
- ROC-AUC curves
- Confusion matrices
- Model comparison analysis

### 📈 Outputs

The pipeline generates comprehensive visualizations in `outputs/`:

1. **01_anomaly_score_distribution.png** - Score distributions by class
2. **02_precision_recall_curve.png** - PR and ROC curves
3. **03_confusion_matrix.png** - Performance metrics visualization
4. **04_model_comparison.png** - Individual model performance
5. **05_score_comparison_scatter.png** - Model correlation analysis
6. **06_feature_importance.png** - Signal contribution heatmap
7. **07_summary_report.txt** - Detailed text report

### 🧪 Experimentation

For interactive exploration, use the Jupyter notebook:

```bash
cd notebooks
jupyter notebook eda_analysis.ipynb
```

### 🔧 Technical Stack

- **Python 3.13**
- **scikit-learn** - Isolation Forest, preprocessing, metrics
- **PyTorch** - Autoencoder neural network
- **UMAP** - Dimensionality reduction
- **pandas/numpy** - Data manipulation
- **matplotlib/seaborn/plotly** - Visualization

### 📝 Patent-Worthy Aspects

1. **Multi-Signal Risk Embedding** - Combining 6 engineered signals into unified embedding
2. **Unsupervised Ensemble** - Weighted combination of diverse anomaly detection algorithms
3. **Domain-Specific Feature Engineering** - Transaction-specific signals capturing fraud patterns
4. **Mathematical Consistency Checking** - Balance discrepancy ratio captures accounting frauds

### 🎓 Academic Notes

This system demonstrates:
- Unsupervised learning for security applications
- Feature engineering for domain-specific ML
- Ensemble methods for robust anomaly detection
- Evaluation on real data despite no labeled training

### 💾 Dataset

The system uses **PaySim synthetic dataset** from Kaggle:
- Realistic transaction patterns
- Multiple transaction types
- Natural fraud distribution
- 10,000+ transactions for training

If dataset unavailable, system auto-generates synthetic PaySim-like data.

### 🤝 Contributing

This is a university ML project designed to demonstrate:
- Advanced unsupervised learning techniques
- Industrial-grade anomaly detection
- Production-ready code structure
- Comprehensive documentation

### 📄 License

University project for educational purposes.

---

**Created:** 2026
**Status:** Production-Ready
**Last Updated:** April 2026
