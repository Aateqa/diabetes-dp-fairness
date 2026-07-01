# Causal and Fair Machine Learning on Diabetes Health Indicators

A research project applying fairness-aware training, differential privacy, causal inference, counterfactual analysis, and transfer learning to diabetes risk prediction. The goal is to build trustworthy healthcare models that do not systematically disadvantage demographic subgroups.

---

## Research Questions

1. **Fairness** - Do standard ML models exhibit measurable demographic bias (by sex, age, income, education) in diabetes prediction, even when sensitive attributes are excluded from training?
2. **Fairness-aware training** - Can Fairlearn's ExponentiatedGradient or a Variational Fair Autoencoder (VFAE) reduce demographic parity difference and FNR gap without unacceptable utility loss?
3. **Differential privacy** - Does adding calibrated noise (diffprivlib DP-LR) further reduce demographic parity differences? Does stricter privacy trade off against fairness, or do the two align?
4. **Causal analysis** - What is the causal effect of obesity on diabetes risk after adjusting for confounders (age, income, physical activity) using a DAG-based backdoor estimator (DoWhy)?
5. **Counterfactual fairness** - What fraction of individuals would receive a different prediction if their sensitive attribute were changed (formal counterfactual fairness ratio, Kusner et al., 2017)?
6. **Transfer and subgroup generalisation** - Does a model trained on high-income patients transfer to low-income patients, and does importance-weighted covariate shift correction (Shimodaira, 2000) close the gap?

---

## Dataset

**BRFSS 2015 Diabetes Health Indicators** (available on Kaggle)

| File | Rows | Class balance |
|---|---|---|
| diabetes_binary_health_indicators_BRFSS2015.csv | 253,680 | ~14% diabetes |
| diabetes_binary_5050split_health_indicators_BRFSS2015.csv | 70,692 | 50/50 |

**Protected attributes evaluated:** Sex, Age (13 categories), Income (8 levels), Education (6 levels), Sex x Age intersection.

---

## Project Structure

```
data/                          Raw CSV files
models/
    __init__.py                Re-exports all model factories and get_models()
    logistic.py                Logistic Regression, Fairlearn ExponentiatedGradient
    tree_ensemble.py           Random Forest, XGBoost, LightGBM, CatBoost
    neural.py                  MLP
    dp_model.py                DP Logistic Regression (diffprivlib), non-private baseline
    vfae.py                    VFAE architecture and VFAEClassifier sklearn wrapper
results/                       Generated CSVs (created on first run)
graphs/                        Generated plots (created on first run)

config.py                      Paths, random seed, CV folds, protected attribute config
data_loader_diabetes.py        Load, deduplicate, preprocess, build feature sets
feature_engineering.py         BMI category, cardiometabolic risk, lifestyle score, etc.
metrics.py                     All evaluation metrics including group fairness and counterfactual fairness ratio

main.py                        Entry point: 5-fold CV on all sklearn, boosting, and VFAE models
dp_training.py                 DP-LR privacy-utility-fairness sweep over epsilon values
vfae_experiment.py             Full VFAE training with threshold tuning (standalone)
causal_analysis.py             DoWhy backdoor causal estimate: obesity to diabetes
counterfactual_diabetes.py     Counterfactual sensitivity and formal fairness ratio
shap_analysis.py               SHAP feature importance across feature sets
transfer_experiment.py         Subgroup generalisation and importance-weighted transfer

cross_validation.py            Stratified K-fold CV loop shared by main.py
compare_all_results.py         Unified comparison table and plots across all methods
plot_tradeoffs.py              Fairness-utility tradeoff plots from CV results
plot_calibration_roc.py        Calibration curves and ROC plots
requirements.txt
```

---

## Setup

```bash
python3.11 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## How to Run

Run `main.py` first since `compare_all_results.py` and `plot_tradeoffs.py` depend on its outputs.

```bash
source venv/bin/activate

python main.py                  # 5-fold CV on all models, raw and balanced datasets
python dp_training.py           # Differential privacy sweep over epsilon values
python vfae_experiment.py       # Full VFAE training with threshold tuning
python causal_analysis.py       # DoWhy backdoor causal estimate
python counterfactual_diabetes.py   # Counterfactual fairness analysis
python shap_analysis.py         # SHAP feature importance
python transfer_experiment.py   # Transfer learning and subgroup generalisation

python compare_all_results.py   # Unified comparison across all methods
python plot_tradeoffs.py        # Fairness-utility tradeoff plots
python plot_calibration_roc.py  # Calibration and ROC curves
```

---

## Methods

### Feature Sets

| Name | Description |
|---|---|
| Original Features | All cleaned and engineered features |
| Without Sensitive Attributes | Drops Sex, Age, and age_bmi_risk |
| Proxy-Reduced | Also drops Income and Education to reduce indirect discrimination |

### Engineered Features

`bmi_category`, `cardiometabolic_risk`, `lifestyle_score`, `bp_chol_combined`, `obese_inactive`, `poor_health_no_access`, `age_bmi_risk`, `any_mental_health_days`, `high_mental_burden`, `any_physical_health_days`, `high_physical_burden`

### Models

| Model | Category |
|---|---|
| Logistic Regression | Baseline |
| Random Forest | Ensemble |
| XGBoost, LightGBM, CatBoost | Gradient Boosting |
| MLP | Neural Network |
| Fairlearn ExponentiatedGradient | Fairness-Constrained (Demographic Parity) |
| VFAE | Fair Deep Representation (adversarial + VAE) |
| DP Logistic Regression (epsilon in 0.1, 0.5, 1, 5, inf) | Differential Privacy |

### Fairness Metrics

| Metric | Description |
|---|---|
| Demographic Parity Difference | Gap in positive prediction rates across groups |
| FNR Gap | Gap in false negative rates across groups; critical in healthcare |
| Worst-Group Sensitivity | TPR of the worst-performing demographic group |
| Macro-Averaged FNR | Average miss rate weighted equally across groups |
| Equalized Odds Difference | Max of TPR gap and FPR gap |
| Counterfactual Fairness Ratio | Fraction of individuals whose prediction changes when sex or age is intervened upon (Kusner et al., 2017) |

### Causal Analysis

A DAG is declared encoding `Sex → Age → BMI_obese → Diabetes`, `Income → PhysActivity → Diabetes`, with direct edges from Age and Income to Diabetes. DoWhy's backdoor estimator adjusts for all declared confounders.

### Transfer Learning

Source domain: high-income group. Target domain: low-income group.

Three conditions are compared:

1. **Oracle** - train on target, test on target (upper bound)
2. **Naive transfer** - train on source, test on target
3. **IPS-weighted transfer** - source reweighted by density ratio (Shimodaira, 2000)

The gap between Naive and Oracle quantifies generalisation failure. The remaining gap after IPS correction motivates causality-based invariant learning.

---

## Key Findings (Preliminary)

| Finding | Result |
|---|---|
| Stricter DP noise reduces demographic parity difference | epsilon=0.1: dp_diff=0.0002 vs non-private: dp_diff=0.034 |
| VFAE with threshold=0.15 | recall=0.80, FNR=0.20, worst-group sensitivity=0.79, FNR gap=0.023 |
| Counterfactual fairness ratio (sex flip) | 4.6% of predictions change when sex is flipped |
| Causal (DoWhy backdoor) obesity to diabetes effect | 0.1375 after adjustment vs naive 0.1398 - confounding is modest but present |

Full 5-fold results are saved to `results/` after running `python main.py`.

---

## Connection to Research Areas

| Internship Topic | This Project |
|---|---|
| Fair ML algorithms | Fairlearn EG, VFAE, fairness metrics across 5 protected attributes |
| Causality for de-biasing | DoWhy backdoor estimation, counterfactual fairness ratio, counterfactual sensitivity analysis |
| Transfer learning and causality | Subgroup generalisation experiment, IPS baseline, gap motivates causal invariant learning |
| Differential privacy | DP-LR sweep, privacy-fairness interaction, grounded in DP-ERM theory |

---

## References

1. Kusner et al. (2017). *Counterfactual Fairness.* NeurIPS.
2. Louizos et al. (2016). *The Variational Fair Autoencoder.* ICLR.
3. Shimodaira (2000). *Improving predictive inference under covariate shift.*
4. Schölkopf et al. (2012). *On causal and anticausal learning.*
5. Wang et al. (2019). *Differentially Private Empirical Risk Minimization with Non-Convex Loss Functions.* ICML.
6. Wang and Xu (2019). *On Sparse Linear Regression in the Local Differential Privacy Model.* ICML.
