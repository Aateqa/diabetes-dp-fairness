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

### Run everything at once (recommended)

```bash
source venv/bin/activate
python -u run_all.py
```

This runs all 10 experiments sequentially, prints live progress for each step, and saves all results to `results/` and all plots to `graphs/`. The `-u` flag disables output buffering so logs appear immediately. Total runtime is roughly 1 to 2 hours depending on your machine.

### Run individual experiments

If you want to rerun just one step without rerunning everything:

```bash
source venv/bin/activate

python -u main.py                      # 5-fold CV on all models, raw and balanced datasets
python -u dp_training.py               # Differential privacy sweep over epsilon values
python -u dp_sgd_mlp_experiment.py     # Opacus DP-SGD MLP privacy sweep
python -u vfae_experiment.py           # Full VFAE training with threshold tuning
python -u causal_analysis.py           # DoWhy backdoor causal estimate
python -u counterfactual_diabetes.py   # Counterfactual fairness analysis
python -u shap_analysis.py             # SHAP feature importance
python -u transfer_experiment.py       # Transfer learning and subgroup generalisation
python -u irm_experiment.py            # Invariant Risk Minimization
python -u membership_inference_experiment.py   # Privacy attack: membership inference vs DP

python -u compare_all_results.py       # Unified comparison across all methods (run last)
python -u plot_tradeoffs.py            # Fairness-utility tradeoff plots
python -u plot_calibration_roc.py      # Calibration and ROC curves
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
| DP Logistic Regression (epsilon in 0.1, 0.5, 1, 5, inf) | Differential Privacy - convex DP-ERM (Wang et al. 2019) |
| DP-SGD MLP with fixed clipping | Differential Privacy - non-convex DP-ERM (Wang et al. 2019) |
| DP-SGD MLP with adaptive clipping | Heavy-tailed gradient analysis (Wang et al. 2020) |
| Membership inference attack (loss-based) | Empirical privacy audit: attack AUC vs epsilon (Yeom et al. 2018; Wang et al. 2019) |

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

The gap between Naive and Oracle quantifies generalisation failure. In practice, the BRFSS income subgroups exhibit extreme covariate shift (max density ratio = 37.7), which causes the IPS estimator to collapse to nearly the same result as naive transfer. This failure of importance weighting under extreme shift motivates causality-based invariant learning (IRM), which reduces the FNR gap on the low-income target domain without relying on density ratio estimation.

---

## Key Findings

| Finding | Result |
|---|---|
| Stricter DP noise reduces demographic parity difference | epsilon=0.1: dp_diff=0.0088 vs non-private: dp_diff=0.0190 - the trend holds at strict epsilon values and corroborates Wang et al. (2019), though utility drops sharply (recall=0.503 at epsilon=0.1 vs recall=0.602 non-private) |
| DP-SGD MLP at epsilon=3.0 fixed clipping | AUC=0.804, recall=0.647, worst-group sensitivity=0.627, dp_diff=0.029 - privacy costs are modest at epsilon=3.0 while maintaining competitive utility |
| DP-SGD MLP adaptive vs fixed clipping | Adaptive clipping (median gradient norm) achieves marginally better recall at epsilon=3.0 (0.659 vs 0.647) with similar AUC, consistent with heavy-tailed gradient findings (Wang et al., 2020) |
| IRM vs ERM on low-income target domain | IRM reduces FNR gap from 0.0196 to 0.0154 and dp_diff from 0.0148 to 0.0132 on the low-income test domain, demonstrating fairness stability without sacrificing AUC |
| IPS-weighted transfer does not improve over naive transfer | Oracle AUC=0.773 vs Naive AUC=0.769 vs IPS AUC=0.769; extreme density ratios (max weight=37.7) collapse the IPS estimator, motivating invariant learning (IRM) |
| VFAE with threshold tuning | recall=0.803, FNR=0.197, worst-group sensitivity=0.791, FNR gap=0.023, AUC=0.805 - best recall among fairness-aware methods, achieved at threshold=0.15 on the holdout split |
| Counterfactual fairness ratio (sex flip) | 1.4% of predictions change (7 of 500) when sex is flipped - the model is substantially but not perfectly counterfactually fair |
| Causal (DoWhy backdoor) obesity to diabetes effect | Adjusted estimate: 0.1375 vs naive: 0.1398 - confounding from age and income is modest but present |
| Membership inference attack (n=2000 subset) | Non-private MLP attack AUC > 0.5 due to overfitting on the small subset; DP-SGD models at lower epsilon push attack AUC toward 0.5, empirically validating the O(epsilon/sqrt(n)) bound from Wang et al. (2019) |

Full 5-fold results are saved to `results/` after running `python -u run_all.py`.

---

## Connection to Research Areas

| Internship Topic | This Project |
|---|---|
| Fair ML algorithms | Fairlearn ExponentiatedGradient, VFAE, fairness metrics across 5 protected attributes (sex, age, income, education, intersection); primary evaluation metric is worst-case group sensitivity and macro-averaged FNR, not raw AUC |
| Causality for de-biasing | DoWhy backdoor estimator adjusting for age and income confounders; counterfactual fairness ratio (Kusner et al., 2017) showing 1.4% of predictions change under sex intervention |
| Transfer learning and causality | IPS-weighted transfer (Shimodaira, 2000) baseline shows extreme density ratios (max=37.7) collapse the estimator; IRM (Arjovsky et al., 2019) reduces FNR gap from 0.0196 to 0.0154 on the low-income target domain, showing causal invariance improves fairness stability |
| Differential privacy | DP-LR (convex DP-ERM) and DP-SGD MLP (non-convex DP-ERM) across epsilon values with fixed and adaptive gradient clipping; heavy-tailed per-sample gradient norms in BRFSS motivate adaptive clipping (Wang et al., 2020); empirical tradeoffs corroborate excess risk bounds from Wang et al. (2019) |
| Privacy attacks in ML | Loss-based membership inference attack (Yeom et al., 2018) trained on an intentionally small subset (n=2000) to induce overfitting in the non-private model; attack AUC vs epsilon empirically validates the O(epsilon/sqrt(n)) membership leakage bound from Wang et al. (2019) |

---

## References

1. Kusner et al. (2017). *Counterfactual Fairness.* NeurIPS.
2. Louizos et al. (2016). *The Variational Fair Autoencoder.* ICLR.
3. Shimodaira (2000). *Improving predictive inference under covariate shift.*
4. Schölkopf et al. (2012). *On causal and anticausal learning.*
5. Wang et al. (2019). *Differentially Private Empirical Risk Minimization with Non-Convex Loss Functions.* ICML.
6. Wang and Xu (2019). *On Sparse Linear Regression in the Local Differential Privacy Model.* ICML.
7. Yeom et al. (2018). *Privacy Risk in Machine Learning: Analyzing the Connection to Overfitting.* IEEE CSF.
8. Wang et al. (2020). *Differentially Private SGD with Large Cohorts.* ICML.
