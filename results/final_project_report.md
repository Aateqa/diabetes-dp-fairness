# Diabetes DP-Fairness Final Project Report

This report is generated automatically from the saved experiment outputs.


## 1. Overall Summary

This project evaluates diabetes prediction under a combined fairness, privacy, clinical screening, transfer learning, causal analysis, counterfactual fairness, and interpretability setting. The main result is that DP-SGD MLP models can preserve strong clinical recall and worst-group sensitivity while sharply reducing empirical membership-inference leakage compared with a non-private MLP.


## 2. Best DP-SGD Clinical Screening Result

The best DP-SGD clinical screening configuration is **DP-SGD MLP ε=5.0 [fixed]**. It achieves AUC=0.8031, recall=0.8931, worst-group sensitivity=0.8898, FNR gap=0.0064, and dp_diff=0.0616.

The non-private MLP baseline achieves AUC=0.8045, recall=0.8881, worst-group sensitivity=0.8820, and FNR gap=0.0118.


## 3. Membership Inference Privacy Audit

The membership inference audit uses multiple attack signals: loss, confidence, entropy, margin, correctness, and a learned logistic-regression attack over these features. Attack AUC near 0.5 indicates random guessing; higher values indicate membership leakage.

- **MLP non-private**: attack AUC=0.6061, advantage=0.1061, strongest feature=neg_loss

- **DP-SGD MLP ε=0.5**: attack AUC=0.5028, advantage=0.0028, strongest feature=neg_loss

- **DP-SGD MLP ε=2.0**: attack AUC=0.5028, advantage=0.0028, strongest feature=neg_loss

- **DP-SGD MLP ε=10.0**: attack AUC=0.5029, advantage=0.0029, strongest feature=neg_loss

- **DP-SGD MLP ε=50.0**: attack AUC=0.5031, advantage=0.0031, strongest feature=neg_loss


## 4. Privacy-Utility Tradeoff

```text
                       model  epsilon clipping  classification_auc   recall  worst_group_sensitivity  fnr_gap  membership_attack_auc
    DP-SGD MLP ε=0.5 [fixed]      0.5    fixed            0.799671 0.876795                 0.869163 0.014701               0.502800
 DP-SGD MLP ε=0.5 [adaptive]      0.5 adaptive            0.800773 0.879644                 0.874141 0.010601               0.502800
    DP-SGD MLP ε=1.0 [fixed]      1.0    fixed            0.800321 0.875769                 0.870822 0.009529                    NaN
 DP-SGD MLP ε=1.0 [adaptive]      1.0 adaptive            0.800374 0.862776                 0.861816 0.001851                    NaN
    DP-SGD MLP ε=3.0 [fixed]      3.0    fixed            0.803654 0.880442                 0.874852 0.010768                    NaN
 DP-SGD MLP ε=3.0 [adaptive]      3.0 adaptive            0.802800 0.887623                 0.882674 0.009533                    NaN
    DP-SGD MLP ε=5.0 [fixed]      5.0    fixed            0.803071 0.893093                 0.889784 0.006374                    NaN
 DP-SGD MLP ε=5.0 [adaptive]      5.0 adaptive            0.801005 0.863004                 0.857549 0.010508                    NaN
   DP-SGD MLP ε=10.0 [fixed]     10.0    fixed            0.802694 0.884431                 0.879592 0.009321               0.502918
DP-SGD MLP ε=10.0 [adaptive]     10.0 adaptive            0.802383 0.878847                 0.875089 0.007238               0.502918
             MLP non-private      inf     none            0.804470 0.888078                 0.881963 0.011781               0.606092
```




## 5. Transfer Learning and Subgroup Generalisation

- **Oracle (train on target)**: AUC=0.7725, F1=0.4605, recall=0.9628, worst-group sensitivity=0.9553, FNR gap=0.0111.

- **Naive transfer**: AUC=0.7690, F1=0.4633, recall=0.9526, worst-group sensitivity=0.9418, FNR gap=0.0161.

- **IPS-weighted transfer**: AUC=0.7687, F1=0.4623, recall=0.9522, worst-group sensitivity=0.9432, FNR gap=0.0134.


## 6. Causal Analysis

The causal analysis estimates the adjusted effect of obesity on diabetes risk using age-adjusted/backdoor-style estimation.

```text
                                analysis                                                                       description  estimate
                        naive_difference Difference in diabetes rate between obese and non-obese groups without adjustment  0.139768
adjusted_logistic_probability_difference         Average predicted probability difference for BMI_obese, adjusting for Age  0.141263
        dowhy_backdoor_linear_regression                                        DoWhy backdoor estimate using declared DAG  0.137496
```




## 7. Counterfactual Fairness

```text
  counterfactual  n_samples  counterfactual_fairness_ratio  n_predictions_changed  percent_predictions_changed  mean_absolute_probability_change  mean_signed_probability_change  max_absolute_probability_change  original_accuracy  counterfactual_accuracy  original_auc  counterfactual_auc
        Sex flip        500                          0.014                      7                          1.4                          0.022447                        0.001695                         0.136589              0.862                     0.86      0.802855            0.806118
Age young→middle        500                          0.002                      1                          0.2                          0.010550                        0.010550                         0.248990              0.862                     0.86      0.802855            0.792722
```




## 8. SHAP Interpretability

SHAP analysis compares feature importance across the original, no-sensitive, and proxy-reduced feature spaces.

```text
               feature  mean_abs_shap                  feature_set
               GenHlth       0.486172            Original Features
  cardiometabolic_risk       0.411387            Original Features
                   Age       0.302949            Original Features
                   BMI       0.238730            Original Features
          age_bmi_risk       0.159353            Original Features
                HighBP       0.146524            Original Features
                   Sex       0.105201            Original Features
                Income       0.095614            Original Features
any_mental_health_days       0.071671            Original Features
     HvyAlcoholConsump       0.065384            Original Features
               GenHlth       0.481689 Without Sensitive Attributes
  cardiometabolic_risk       0.477351 Without Sensitive Attributes
                   BMI       0.329756 Without Sensitive Attributes
                HighBP       0.186214 Without Sensitive Attributes
any_mental_health_days       0.143514 Without Sensitive Attributes
```




## 9. Final Takeaway

The strongest project claim is that privacy-preserving deep learning can maintain clinically useful screening behaviour while reducing empirical membership leakage. The DP-SGD MLP preserves high recall and high worst-group sensitivity, while the membership inference audit shows the non-private MLP leaking membership signal and DP-SGD models remaining close to random guessing.
