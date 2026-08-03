# Phishing Website Detection using Machine Learning

> **Summer Internship Project — NIT Jalandhar**

A machine learning pipeline to classify URLs as **phishing** or **legitimate** using 46 carefully validated features extracted from URL structure, HTML content, and page metadata.

## Highlights

- **235,795 URLs** from the PhiUSIIL benchmark dataset (UCI ML Repository)
- **Rigorous data leakage investigation** — identified and removed 5 leaked/confounded features
- **5 models trained & compared**: Logistic Regression, Naive Bayes, Random Forest, XGBoost, MLP
- **Best model: XGBoost** — F1 = 0.9999, only 1 missed phishing page out of 20,104 test samples

---

## Dataset

| Property | Value |
|----------|-------|
| **Name** | PhiUSIIL Phishing URL Dataset |
| **Source** | [UCI ML Repository (id=967)](https://archive.ics.uci.edu/dataset/967) / [Kaggle](https://www.kaggle.com/datasets/ndarvind/phiusiil-phishing-url-dataset) |
| **Samples** | 235,795 URLs |
| **Features** | 55 columns (51 numeric + 4 text/categorical) |
| **Target** | `label` — 1 = Legitimate (57.3%), 0 = Phishing (42.7%) |

---

## Project Structure

```
phishing-detector/
├── data/
│   ├── PhiUSIIL_Phishing_URL_Dataset.csv   # Main dataset (Kaggle download)
│   ├── phiusiil_raw.csv                     # UCI ML Repository copy
│   ├── X_train_raw.csv, X_test_raw.csv      # Phase 3 outputs (raw features)
│   └── y_train.npy, y_test.npy              # Phase 3 outputs (labels)
├── models/
│   ├── best_model.pkl                       # Saved XGBoost classifier
│   ├── scaler_phase4.pkl                    # StandardScaler (fitted on train)
│   ├── feature_columns.json                 # Feature schema for deployment
│   ├── tld_encoder.pkl                      # TLD label encoder
│   ├── confusion_matrices.png               # Evaluation plots
│   ├── roc_curves.png
│   ├── feature_importance.png
│   └── leakage_distributions.png
├── notebooks/
│   ├── 01_data_acquisition.ipynb            # Phase 1: Setup & data download
│   ├── 02_feature_engineering.ipynb         # Phase 2: Custom feature extractor
│   ├── 03_preprocessing.ipynb               # Phase 3: EDA, encoding, split
│   └── 04_model_training.ipynb              # Phase 4: Leakage analysis & training
├── app/                                      # (Phase 5: Deployment — future)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Pipeline Overview

### Phase 1 — Data Acquisition
- Fetched dataset from UCI ML Repository and Kaggle
- Saved local CSV copies for reproducibility
- Validated dataset shape and column types

### Phase 2 — Feature Engineering
- Built a **custom web scraper** that can extract features from any live URL:
  - **URL-string features**: length, domain, TLD, subdomain count, character ratios, HTTPS, special characters
  - **HTML/content features**: lines of code, images, CSS/JS files, forms, iframes, social links, copyright
  - **WHOIS features**: domain age
- Tested end-to-end on live URLs (google.com, wikipedia.org)

### Phase 3 — Preprocessing & EDA
- Checked for missing values (none) and duplicates (425 URLs removed)
- Analyzed class distribution (57/43% — slight imbalance, not enough to require SMOTE)
- Visualized feature distributions and 51×51 correlation heatmap
- Dropped text columns (`URL`, `Domain`, `Title`) and encoded `TLD` with LabelEncoder
- 80/20 stratified train/test split
- StandardScaler fitted on train, applied to test

### Phase 4 — Model Training & Leakage Investigation

#### Data Leakage Found
The initial training (all 51 features) gave **99.9–100% accuracy** on all models including Naive Bayes — a clear leakage signal.

**Dropped features (5):**

| Feature | Reason | Correlation |
|---------|--------|:-----------:|
| `URLSimilarityIndex` | Computed using the true label (per dataset paper) | 0.86 |
| `HasSocialNet` | Scraping artifact — 99.5% zero for phishing | 0.78 |
| `HasCopyrightInfo` | Scraping artifact — 94.3% zero for phishing | 0.74 |
| `HasDescription` | Scraping artifact — 95.6% zero for phishing | 0.69 |
| `IsResponsive` | Scraping artifact — 68.3% zero for phishing | 0.55 |

**Kept features** (`NoOfExternalRef`, `LineOfCode`, `NoOfSelfRef`, `NoOfImage`, `NoOfJS`) — validated as genuine behavioral signal through quantile analysis (continuous distributions, 0% dead-page rate, moderate correlations 0.26–0.35).

#### Final Model Comparison (46 clean features)

| Model | Accuracy | Precision | Recall | F1 | AUC | FN |
|-------|:--------:|:---------:|:------:|:---:|:---:|:--:|
| **XGBoost** | **0.9999** | **0.9999** | **1.0000** | **0.9999** | **1.0000** | **1** |
| Random Forest | 0.9999 | 0.9999 | 0.9999 | 0.9999 | 1.0000 | 4 |
| MLP | 0.9995 | 0.9997 | 0.9994 | 0.9996 | 0.9999 | 7 |
| Logistic Regression | 0.9991 | 0.9987 | 0.9998 | 0.9993 | 0.9999 | 34 |
| Naive Bayes | 0.9317 | 0.8941 | 0.9992 | 0.9437 | 0.9515 | 3192 |

**Selected: XGBoost** — highest F1, perfect recall, only 1 false negative. In phishing detection, missed threats are more dangerous than false alarms.

---

## How to Reproduce

### 1. Setup
```bash
git clone <repo-url>
cd phishing-detector
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### 2. Run Notebooks (in order)
```bash
jupyter notebook notebooks/
```
1. `01_data_acquisition.ipynb` — downloads dataset
2. `02_feature_engineering.ipynb` — builds feature extractor (requires internet for live URL tests)
3. `03_preprocessing.ipynb` — EDA + train/test split
4. `04_model_training.ipynb` — leakage investigation + model training

### 3. Use the Trained Model
```python
import pickle, json
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Load artifacts
with open('models/best_model.pkl', 'rb') as f:
    model = pickle.load(f)
with open('models/scaler_phase4.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open('models/feature_columns.json', 'r') as f:
    schema = json.load(f)

# Predict on new data
features = schema['feature_columns']  # 46 features in exact order
# ... extract features for a URL ...
# X = scaler.transform(df[features])
# prediction = model.predict(X)  # 0=phishing, 1=legitimate
```

---

## Technologies

- **Python 3.13** — pandas, numpy, scikit-learn, XGBoost, matplotlib, seaborn
- **BeautifulSoup** + requests — web scraping for feature extraction
- **python-whois** — domain age lookup
- **Jupyter Notebooks** — interactive development and documentation

---

## Key Takeaways

1. **High accuracy alone doesn't mean a good model** — the original 100% accuracy was caused by data leakage
2. **Always investigate suspiciously high-performing features** — `URLSimilarityIndex` was computed using the label itself
3. **Scraping artifacts can masquerade as behavioral features** — site-completeness features reflected dead pages, not phishing behavior
4. **Genuine behavioral signal exists** — phishing pages really are simpler clones with fewer resources, less code, and fewer links
5. **Tree-based models (XGBoost, RF) excel** on tabular feature engineering tasks — even simple Logistic Regression achieves 99.9% F1 on clean features
