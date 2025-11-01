# Credit Card Fraud Detection - Production ML Pipeline

A comprehensive, production-quality machine learning projet for detecting fraudulent credit card transactions using the Kaggle Credit Card Fraud Detection dataset.

## 🎯 Project Overview

**Problem**: Binary classification to identify fraudulent credit card transactions in a highly imbalanced dataset.

**Business Impact:**
- Reduce financial losses from fraudulent transactions
- Minimize false positives (legitimate transactions incorrectly flagged)
- Enable real-time fraud detection
- Support fraud investigation with interpretable models

**Key Challenges:**
- Extreme class imbalance (~0.17% fraud rate)
- Time-series data requiring temporal validation
- Anonymous features (V1-V28 PCA-transformed)
- Real-time inference requirements
- High cost of both false positives and false negatives

## 📊 Dataset
**Source**: [Kaggle Credit Card Fraud Detection] (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

**Description:**
- 284,807 transactions over 2 days (September 2013)
- 492 frauds (~0.172% of all transactions)
- 30 features: Time, V1-V28 (PCA-component), Amount, Class

**Features:**
- `Time`: Seconds elapsed between first transaction and current transaction
- `V1-V28`: PCA-transformed features (confidential original features)
- `Amount`: Transaction amount
- `Class`: Target variable (0 = legitimate, 1 = fraud)

**Important Notes:**
- Feature V1-V28 are principal components from PCA for privacy
- No missing values in the dataset
- Highly imbalanced: 492 frauds vs 284.315 legitimate transactions
- Temporal nature: transactions occur over 2 days

## 📁 Repository Structure
```
fraud_detection/
├── configs/
├── data
│   ├── external           # External reference data
│   ├── processed          # Cleaned and transformed data
│   └── raw                # Original immutable dataset
├── models
│   └── mlruns             # MLflow experiment tracking
├── notebooks              # Jupyter notebooks for each project phase
|   └── 01_data_audit.ipynb
|   └── 02_eda_exploration.ipynb
|   └── 03_data_cleaning.ipynb
|   └── 04_feature_engineering.ipynb
|   └── 05_validation_strategy.ipynb
|   └── 06_modeling.ipynb
│   └── utils.py           # Shared utility functions
├── .gitignore
├── README.md
├── requirements.txt       # Pinned dependencies
└── src                    # Production code (future)
```

## 🚀 Getting Started

### Prerequisites
- Python 3.9+
- pip or conda

### Installation

1. **Clone the repository** (or create the structure):
```bash
git clone <repository-url>
cd fraud_detection
```

2. **Create virtual environment**:
```bash
python -m venv .venv
source .venv/bin/activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Download dataset**:
- Visit [Kaggle Dataset] (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- Download `creditcard.csv`
- Place in `data/raw/` directory

5. **Launch Jupyter**:
```bash
jupyter-notebook
```

6. **Start MLflow UI** (Optional, for tracking):
```bash
mlflow-ui --backend-store-uri file:///path/to/fraud_detection/models/mlruns
```

## 🔬 Project Phases

### Phase 0: Kickoff and Skeleton ✅
- [x] Repository structure setup
- [x] Environment configuration
- [x] Utility module creation
- [x] MLflow initialization

### Phase 1: Data Audit & Contracts
- [ ] Schema validation
- [ ] Data quality assessment
- [ ] Great Expectations suite
- [ ] Data dictionary reation

### Phase 2: High-Impact EDA
- [ ] DuckDB-powered analysis
- [ ] Target-feature relationships
- [ ] Correlation analysis
- [ ] Temporal patterns
- [ ] outlier detection

### Phase 3: Data Cleaning & Normalization
- [ ] Outlier treatment strategy
- [ ] Feature scaling decisions
- [ ]  Temporal split preparation
- [ ] Data validation pipeline

### Phase 4: Feature Engineering
- [ ] Time-based features
- [ ] Interaction features
- [ ] Aggregation features
- [ ] Feature selection

### Phase 5: Validation Stragety
- [ ] Time-based cross-validation
- [ ] Metric selection (PR-AUC focus)
- [ ] Stratification strategy
- [ ] Calibration assessment

### Phase 6: Baseline Models
- [ ] Dummy classifier baseline
- [ ] Logistic Regression + class weights
- [ ] Random Forest
- [ ] Initial preformance comparison

### Phase 7: Advanced Modeling
- [ ] XGBoost with hyperparameter tuning
- [ ] LightGBM optimization
- [ ] Threshold optimization
- [ ] Model calibration
- [ ] Ensemble methods

### Phase 8: Model Evaluation & Selection
- [ ] Comprehensive metric comparison
- [ ] Confusion metrix analysis
- [ ] Cost-benefit analysis
- [ ] Threshold tuning for business objectives
- [ ] Final model selection

### Phase 9: Explainability & Robustness
- [ ] Feature importance analysis
- [ ] SHAP values for interpretability
- [ ] Drift simulation
- [ ] Robustness testing
- [ ] Fairness checks

### Phase 10: Deployment Preparation
- [ ] Inference pipeline export
- [ ] FastAPI endpoint
- [ ] Docker containerization
- [ ] Unit/integration tests
- [ ] Performance benchmarking

### Phase 11: Documentation & Handover
- [ ] Model card
- [ ] Training runbook
- [ ] Inference documentation
- [ ] Monitoring recommendations
- [ ] Future improvements

## 📈 Evaluation Metrics

**Primary Metrics:**
- **PR-AUC (Precision-Recall AUC)**: Primary metric due to extreme imbalance
- **ROC-AUC**: Secondary metric for overall performance
- **Recall @ 90% Precision**: Business-oriented threshold metric


**Secondary Metrics:**
- F1-Score (at optimal threshold)
- Matthews Correlation Coefficient (MCC)
- Cost-weighted metric (if business costs available)

**Why PR-AUC?**: With 0.17% fraud rate, accuray is meaningless (99.83% by predicting all legitimate). PR-AUC focuses on minority class performance, which is critical for fraud detection.

## 🛠️ Technology Stack

**Core ML:**
- pandas: Data manipulation
- numpy: Numerical operations
- DuckDB: Memory-efficient queries

**Validation & Quality:**
- Great Expectations: Data contracts
- Pytest: Testing framework

**Tracking & Deployment:**
- MLflow: Experiment tracking
- FastAPI: REST API (future)
- Docker: Containerization (future)

**Visualization:**
- matplotlib, seaborn: Static plots
- plotly: Interactive visualization

## 🎯 Success Criteriy

**Model Performance:**
- PR-AUC > 0.75
- ROC-AUC > 0.95
- Recall @ 90% Precision > 0.70

**Production Readiness:**
- Inference latency < 100ms (p95)
- Reproducible pipeline with seeds
- Comprehensive test coverage
- Clear documentation

**Business Value:**
- Demonstrate cost-benefit analysis
- Explainable predictions for investigation
- Robust to temporal drift

## 📚 References

- [Kaggle Dataset] (https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- [Original Paper] (https://www.researchgate.net/publication/319867396_Credit_Card_Fraud_Detection_A_Realistic_Modeling_and_a_Novel_Learning_Strategy)
- [Imbalanced Learning Techniques] (https://imbalanced-learn.org/)

## 👤 Author

[Nowraj Farhan]
Aspiring Data Scientist | Machine Learning Engineer
[Portfolio] (https://github.com/nfarhan) | [LinkedIn] (https://linkedin.com/in/nowrajfarhan)

## 📝 License

This project is for educational and portfilio purposes.

---