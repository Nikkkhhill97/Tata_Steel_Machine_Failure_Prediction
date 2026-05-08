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
SplitRecordsColumnsTraining Set1,36,42914Test Set90,95414
Key Features:

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
Three high-impact synthetic variables created:
FeatureFormulaBusiness Purposetemp_diffProcess Temp − Air TempMonitors cooling efficiencypower_estTorque × Rotational SpeedReflects total machine workloadtorque_per_speedTorque / Rotational SpeedDetects mechanical strain
3. Handling Class Imbalance

Applied SMOTE (Synthetic Minority Over-sampling Technique) on training data
Used scale_pos_weight in XGBoost to bias learning toward rare failure events

4. Models Evaluated
ModelNotesLogistic RegressionBaseline / performance floorRandom ForestEnsemble, captures non-linearityXGBoost (Tuned)Champion Model — best F1-macroLightGBMEfficient gradient boosting
Hyperparameter Tuning: RandomizedSearchCV with 3-fold Stratified Cross-Validation, optimizing for F1-macro score.
5. Model Explainability (SHAP)
Top 3 failure predictors identified:

Torque — Most critical sensor signal
Tool Wear — Mechanical fatigue indicator
Temperature Difference — Thermal inefficiency signal

## 🖼️ Key Visualizations

All charts saved in assets/ folder. See below for what each represents.

ChartFolderInsightClass Imbalance Distributioneda/Shows 3.39% failure rate — justifies SMOTEFailure Mode Breakdowneda/Which failure type dominatesTorque vs Speed Scattereda/Confirms inverse mechanical lawFeature Correlation Heatmapeda/Multicollinearity checkTemperature Distributioneda/Air vs Process temp spreadTool Wear vs Failure Boxploteda/Fatigue threshold identificationConfusion Matrix (XGBoost)models/True/False Positive-Negative breakdownROC Curve Comparisonmodels/All models vs baselineModel Performance Bar Chartmodels/F1 / Precision / Recall across all modelsSHAP Summary Plotexplainability/Feature impact distributionSHAP Feature Importance Barexplainability/Top predictors ranked

## 🧰 Tech Stack
LibraryPurposePandas, NumPyData manipulation & numerical computationMatplotlib, SeabornEDA visualizationsScikit-LearnML pipelines, metrics, cross-validationXGBoostChampion gradient boosting modelLightGBMEfficient boosting alternativeImbalanced-LearnSMOTE for class balancingSHAPModel explainabilitySciPyStatistical analysis

## 💼 Business Impact

20–30% reduction in unplanned downtime by predicting failures before they occur
Maintenance can be scheduled in planned windows, not emergency shutdowns
Real-time monitoring of Torque, Tool Wear, and temp_diff thresholds provides a clear operational dashboard for maintenance teams
Reduces repair costs, improves worker safety, and ensures consistent production quality
