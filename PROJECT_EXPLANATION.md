# Project Explanation — PRCP-1010 InsClaimPred

This document explains the purpose, scope, and contents of the **Insurance Claim Prediction** project in simple terms. It is meant for anyone reviewing the repository (mentor, reviewer, or teammate) without needing to open every file first.

---

## 1. What is this project?

**InsClaimPred** is an intern-level data science / machine learning project in the **finance / insurance** domain.

An insurance company wants to understand which customers are more likely to be associated with claim-related outcomes, so the marketing and product teams can:

1. Target the right customers more effectively
2. Design better product offers
3. Reduce wasted outreach on low-value / poorly matched leads

The original problem code is **PRCP-1010-InsClaimPred**.

---

## 2. Business problem (from the brief)

### Task 1 — Predictive model
Create a model that helps the insurance marketing team identify which customers are more likely to buy / engage with the product based on customer characteristics.

### Task 2 — Marketing suggestions
Give practical recommendations the marketing team can use to improve product uptake (not only model scores).

### Additional deliverables
- **Model Comparison Report:** compare multiple models and recommend one for production use
- **Challenges Report:** document data/modeling difficulties and how they were handled
- Submit the work in a **single Jupyter notebook**

---

## 3. Dataset overview

| Item | Detail |
|---|---|
| Source | Capstone zip from DataTrained / CDS project host |
| File used | `train.csv` |
| Rows | 595,212 |
| Columns | 59 (`id`, `target`, + 57 anonymized features) |
| Target | Binary (`0` = majority / no event, `1` = rare positive ~3.6%) |
| Privacy | Feature names are anonymized, so deep name-based EDA is limited |

### Important data quirks
- Missing values are often encoded as **`-1`** (not standard `NaN`)
- Strong **class imbalance** makes accuracy a misleading metric
- Dataset is large, so stratified sampling is used for faster intern iteration

Download command:

```bash
python scripts/download_data.py
```

---

## 4. What each file / folder does

```text
InsClaimPred/
├── PRCP-1010-InsClaimPred (5).docx   # Original project brief
├── PROJECT_EXPLANATION.md            # This explanation document
├── README.md                         # Setup + quick-start guide
├── requirements.txt                  # Python dependencies
├── notebooks/
│   └── PRCP_1010_InsClaimPred.ipynb  # Main end-to-end submission notebook
├── scripts/
│   ├── download_data.py              # Downloads and extracts train.csv
│   └── run_pipeline.py               # CLI pipeline: train, compare, export reports
├── reports/                          # Generated charts + metrics
│   ├── model_comparison.csv
│   ├── summary.json
│   ├── roc_curves.png
│   ├── confusion_gradient_boosting.png
│   └── feature_importance.png
└── data/                             # Local dataset (not committed; download required)
```

### Why both a notebook and scripts?
- The **notebook** matches the official submission format from the brief
- The **scripts** make the same workflow reproducible from the command line

---

## 5. Modeling workflow (high level)

1. **Load data** from `data/train.csv`
2. **Light checks** on shape, target balance, and `-1` missing counts
3. **Preprocess**
   - replace `-1` with `NaN`
   - drop `id`
   - impute missing values (median)
   - scale features for logistic regression
4. **Train multiple models**
   - Logistic Regression
   - Random Forest
   - Gradient Boosting
   - XGBoost (if installed)
5. **Compare metrics**
   - primary: ROC-AUC, Normalized Gini
   - supporting: Average Precision, Precision, Recall, F1
6. **Recommend best model** for production ranking use
7. **Write marketing suggestions** and challenges

### Why not use accuracy as the main score?
Because about **96%** of customers are class `0`. A model that always predicts `0` looks accurate but is useless for finding rare positives. Ranking metrics are more meaningful for marketing prioritization.

---

## 6. Sample results (from committed reports)

On a stratified development sample, **Gradient Boosting** was the strongest model by ROC-AUC in the CLI pipeline run.

Typical interpretation for this project:
- Use the model to **rank** customers by probability
- Do **not** rely only on a hard `0.5` threshold
- Review top-scoring segments first for campaigns / offers

Exact numbers are stored in:
- `reports/model_comparison.csv`
- `reports/summary.json`

---

## 7. Marketing recommendations (Task 2 summary)

1. Score all customers and prioritize the **top decile** for outreach
2. Tune decision thresholds to fit campaign budget and tolerance for false positives
3. Offer safer product tiers to higher-risk scored customers; conversion offers to safer segments
4. Run A/B tests (model-targeted vs random) to prove marketing lift
5. Retrain periodically because claim behavior and customer mix drift over time
6. Keep business rules / compliance filters on top of ML scores

---

## 8. Challenges faced and techniques used

| Challenge | Impact | Technique |
|---|---|---|
| Severe class imbalance (~3.6% positives) | Accuracy looks good while missing rare events | Class weights / scale_pos_weight; rank by ROC-AUC/Gini |
| Missing values stored as `-1` | Trees/linear models may treat missing as real numbers | Convert to `NaN` + median imputation |
| Anonymized feature names | Hard to tell a business story from column names alone | Focus on model quality + feature-importance proxies |
| Large dataset size | Slow experiment loops for an intern setup | Stratified sampling for development; full-data option available |

---

## 9. How to run the project

```bash
# Install
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# Get data
python scripts/download_data.py

# Option A: notebook (official-style submission)
jupyter notebook notebooks/PRCP_1010_InsClaimPred.ipynb

# Option B: CLI pipeline
python scripts/run_pipeline.py                # sample run
python scripts/run_pipeline.py --sample-size 0  # full dataset
```

---

## 10. Recommended reading order

If you are new to this repository, read files in this order:

1. `PROJECT_EXPLANATION.md` (this file)
2. `README.md` (setup)
3. `notebooks/PRCP_1010_InsClaimPred.ipynb` (full analysis)
4. `reports/model_comparison.csv` and charts (results)
5. `scripts/run_pipeline.py` (reproducible implementation)

---

## 11. Scope and limitations

This is an **intern-level baseline**, not a full production ML platform.

Included:
- clean project structure
- multiple classic model baselines
- evaluation suitable for imbalanced data
- marketing + challenges write-up

Not included yet (possible stretch goals):
- hyperparameter tuning / cross-validation grids
- model calibration and threshold optimization UI
- real-time scoring API / dashboard
- fairness / bias audit by segment

---

## 12. Final takeaway

The project turns the PRCP-1010 insurance brief into a complete, runnable repository:
- one submission notebook
- reproducible scripts
- clear reports
- practical marketing guidance

Best use of the model in practice is as a **customer ranking engine** that helps marketing decide whom to contact first and what kind of offer to present.
