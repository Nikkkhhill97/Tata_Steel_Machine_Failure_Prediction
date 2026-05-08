# 🏭 Tata Steel Machine Failure Prediction
Capstone Project — Machine Learning & GenAI with Microsoft Azure
Domain: Manufacturing | Type: Multi-label Classification | Tools: Pandas, Scikit-Learn, XGBoost, LightGBM, SHAP

## 📌 Business Context
In the manufacturing sector, unplanned machine downtime leads to massive production losses, high repair costs, and serious safety risks. Tata Steel, a global leader in steel manufacturing, relies on continuous, high-load machinery whose failure can cascade across the entire production line.
This project builds an end-to-end Predictive Maintenance System using machine learning — enabling Tata Steel to shift from a reactive "break-fix" approach to a proactive, data-driven maintenance strategy, potentially reducing unplanned downtime by 20–30%.

## 🎯 Problem Statement
Given real-time sensor data (temperatures, rotational speed, torque, tool wear), the model must:

Predict Binary Failure — Will this machine fail? (Yes / No)
Identify the Failure Mode — Which specific failure type is occurring?

Tool Wear Failure (TWF)
Heat Dissipation Failure (HDF)
Power Failure (PWF)
Overstrain Failure (OSF)
Random Failures (RNF)

Success = minimizing False Negatives (missed failures) while maintaining high Precision, so maintenance teams get actionable, reliable alerts.

## 📊 Dataset Overview
| Split | Records | Columns |
|---|---|---|
| Training Set | 1,36,429 | 14 |
| Test Set | 90,954 | 14 |

Air Temperature [K], Process Temperature [K]
Rotational Speed [rpm], Torque [Nm], Tool Wear [min]
Type — Product quality level (L / M / H)
Machine Failure — Binary target
TWF, HDF, PWF, OSF, RNF — Failure mode flags

Class Imbalance: Only 3.39% of records represent actual machine failures — a classic "needle in a haystack" problem.

## 🔬 Project Workflow
Raw Data → EDA → Feature Engineering → SMOTE Balancing → Model Training → Hyperparameter Tuning → SHAP Explainability → Business Insights
1. Exploratory Data Analysis (EDA)

Validated fundamental mechanical laws (inverse correlation between Torque & Rotational Speed confirmed)
Identified failure mode distribution and trigger conditions
Structured analysis using the UBM Rule: Univariate → Bivariate → Multivariate

2. Feature Engineering
   
| Feature | Formula | Business Purpose |
|---|---|---|
| `temp_diff` | Process Temp − Air Temp | Monitors cooling efficiency |
| `power_est` | Torque × Rotational Speed | Reflects total machine workload |
| `torque_per_speed` | Torque / Rotational Speed | Detects mechanical strain |

3. Handling Class Imbalance

Applied SMOTE (Synthetic Minority Over-sampling Technique) on training data
Used scale_pos_weight in XGBoost to bias learning toward rare failure events

4. Models Evaluated
   
| Model | Notes |
|---|---|
| Logistic Regression | Baseline / performance floor |
| Random Forest | Ensemble, captures non-linearity |
| XGBoost (Tuned) | **Champion Model** — best F1-macro |
| LightGBM | Efficient gradient boosting |

6. Model Explainability (SHAP)
Top 3 failure predictors identified:

Torque — Most critical sensor signal
Tool Wear — Mechanical fatigue indicator
Temperature Difference — Thermal inefficiency signal

## 🖼️ Key Visualizations

All charts saved in assets/ folder. See below for what each represents.

| Chart | Folder |
|---|---|
| Distribution of Machine Failure (0 = No Failure, 1 = Failure) | `assets/eda/` |
| Frequency of Specific Machine Failure Modes | `assets/eda/` |
| Machine Failure Distribution across Product Types (L, M, H) | `assets/eda/` |
| Correlation Heatmap of Sensor Features | `assets/eda/` |
| Torque vs. Rotational Speed (Sampled Data) | `assets/eda/` |
| Distribution of Features (Multi-subplot) | `assets/eda/` |
| Effect of Temperature Difference on Failure | `assets/eda/` |
| Tool Wear vs. Machine Failure | `assets/eda/` |
| Multivariate Scatter (Chart 9) | `assets/eda/` |
| Torque Distribution by Product Type and Failure | `assets/eda/` |
| Air Temp vs Failure & Process Temp vs Failure | `assets/eda/` |
| Rotational Speed Distribution by Product Type | `assets/eda/` |
| Count of Concurrent Failure Modes | `assets/eda/` |
| Complete Feature Correlation Matrix | `assets/eda/` |
| Multidimensional Sensor Relationships (Pair Plot) | `assets/eda/` |
| Evaluation Metric Comparison: Logistic Regression (Baseline vs Tuned) | `assets/models/` |

## 🧰 Tech Stack

| Library | Purpose |
|---|---|
| Pandas, NumPy | Data manipulation & numerical computation |
| Matplotlib, Seaborn | EDA visualizations |
| Scikit-Learn | ML pipelines, metrics, cross-validation |
| XGBoost | Champion gradient boosting model |
| LightGBM | Efficient boosting alternative |
| Imbalanced-Learn | SMOTE for class balancing |
| SHAP | Model explainability |
| SciPy | Statistical analysis |

## 💼 Business Impact

20–30% reduction in unplanned downtime by predicting failures before they occur
Maintenance can be scheduled in planned windows, not emergency shutdowns
Real-time monitoring of Torque, Tool Wear, and temp_diff thresholds provides a clear operational dashboard for maintenance teams
Reduces repair costs, improves worker safety, and ensures consistent production quality
