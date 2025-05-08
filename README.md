# Predictive Analytics for Cardiovascular Disease Detection

**Leveraging machine learning and big‑data processing to enable early, scalable CVD risk assessment**

---

## Problem Statement

Cardiovascular diseases (CVDs) account for roughly 31% of global mortality. Traditional diagnostics rely on time‑intensive tests and specialist interpretation, delaying intervention. We aim to build an automated, accurate risk‑prediction system to flag high‑risk patients early, improving outcomes and optimizing resource allocation.

## Business Problem & Impact

- **Problem:** Hospitals and clinics lack an efficient, scalable way to triage asymptomatic or high‑risk patients before advanced symptoms emerge.  
- **Impact:** Early identification of CVD risk can:
  - Reduce downstream treatment costs  
  - Improve patient prognoses through timely interventions  
  - Enable payers to design data‑driven premium models  
  - Support medical device companies and telehealth platforms in delivering targeted monitoring  

## Research Questions

1. How can machine learning improve the early detection of CVD?  
2. How does age influence CVD risk across genders?  
3. Do chest pain type and exercise‑induced angina compound risk?  
4. What role does resting ECG play in risk stratification?  
5. How does ‘old peak’ ST‑depression correlate with disease presence?  

## Data Description

- **Source:** Kaggle “Heart Failure Prediction” dataset  
- **Records:** ~300 patient visits, no missing values after ingestion  
- **Features (11 total):**  
  - Age, Sex, ChestPainType, RestingBP, Cholesterol, FastingBS  
  - RestingECG, MaxHR, ExerciseAngina, Oldpeak, ST_Slope  
- **Target:** HeartDisease (0 = No, 1 = Yes)  

## Approach & Methodology

1. **Data Ingestion & Cleaning**  
   - Load CSV into Spark DataFrame  
   - Drop duplicates, validate ranges, correct obvious data errors (e.g., zero BP/cholesterol)  
   - One‑hot encode categorical features  

2. **Exploratory Data Analysis (EDA)**  
   - Summary statistics, boxplots and distributions  
   - Correlation heatmaps for numeric features  
   - Cross‑tabulation of categorical–numerical interactions  

3. **Feature Engineering**  
   - Binarize high‑risk thresholds (e.g., Cholesterol > 240 mg/dL)  
   - Create age‑group buckets for interaction analysis  

4. **Model Training**  
   - Random Forest classifier in Spark MLlib  
   - 5‑fold cross‑validation to tune `n_estimators`, `max_depth`, `min_samples_split`  
   - Track experiments with MLflow for reproducibility  

5. **Evaluation**  
   - Metrics: ROC AUC, accuracy, precision, recall  
   - Confusion matrix and ROC curves  

## Exploratory Data Analysis & Key Findings

- **Age & Gender:** Mean age 53.5; risk rises sharply after 50, similar trends in males and females.  
- **Chest Pain & Angina:** Asymptomatic chest pain (ASY) and exercise‑induced angina (Y) show the highest heart‑disease rates.  
- **Resting ECG & Oldpeak:** Flat/downward ST slopes and higher Oldpeak depression correlate strongly with positive cases.  
- **Outliers & Data Quality:** A few zero values in BP/cholesterol flagged for further validation.  

## Modeling & Evaluation

| Metric      | Training | Validation |
|-------------|----------|------------|
| Accuracy    | 98.2%    | 95.4%      |
| ROC AUC     | 0.99     | 0.98       |
| Precision   | 0.96     | 0.94       |
| Recall      | 0.95     | 0.93       |

> **Best Model:** Random Forest (100 trees, max depth 10) achieved an AUC of 0.98 on hold‑out data.

## Solution & Insights

- **High‑Risk Profiles:**  
  - Age > 60 with ASY chest pain and ST slope “Down”  
  - Elevated Oldpeak (> 2.0) even when resting ECG is “Normal”  

- **Operational Takeaway:**  
  - Integrate this model into EHR systems to trigger automatic referrals to cardiology for flagged patients.  
  - Provide a “CVD risk score” dashboard for clinicians and case managers.

## Business Recommendations

1. **Healthcare Providers:** Embed automated CVD screening in annual checkups for patients over 50.  
2. **Insurance Payers:** Use risk scores to offer personalized premiums and preventative wellness programs.  
3. **Wearable & Telehealth Vendors:** Augment device firmware to capture ST slope and heart‑rate recovery metrics, feeding into on‑device risk alerts.  
4. **Public Health Agencies:** Target awareness campaigns at rural communities with limited access to advanced diagnostics.

## Future Work

- Expand feature set with lab tests (e.g., NT‑proBNP, troponin).  
- Experiment with deep learning on ECG signal waveforms.  
- Deploy model as a REST API with real‑time streaming support.  
- Validate model on external cohorts to assess generalizability.

#### Avanti, Chhaya

## Getting Started

1. **Clone the repo**  
   ```bash
   git clone https://github.com/your-org/cvd-prediction.git
   cd cvd-prediction
