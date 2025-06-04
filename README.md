
# Capstone Phase 5 Group 9
## Predicting Customer Churn Risk in Internet Service Subscribers using Machine Learning.

## Project Structure
- `data/`: Contains the dataset used in the project.
- `notebooks/`: Jupyter notebooks for EDA, modeling, and evaluation.
- `reports/`: Project proposal and documentation.

## Dataset
A CSV dataset containing 36,992 customer records with 23 features.

## Key target Variable 
'churn' (Yes/No recorded after 30 days of inactivity or cancellation)

## Objectives
- Predict customer churn risk with ≥ 70 % recall (high‑risk segment)
  ![image](https://github.com/user-attachments/assets/c4d9c752-ec99-4c4e-84a1-8c4fad0bd0b9)

- Uncover the drivers of churn at both global and segment levels
- Incorporate customer feedback & sentiment into predictive features
- Recommend actionable strategies to reduce churn by at least 10 % YoY

## Business Benefits
- Reduced Acquisition Cost - Retaining existing users is ~5× cheaper than 
  acquiring new ones
- Higher CLV - Targeted offers extend customer lifetime and loyalty
- Customer-Centric Innovation - Complaint and sentiment analysis fuels product/service improvements
- Revenue Stabilization - Early churn signals allow proactive interventions

# Methodology

## 1. Business Understanding
- Clearly defined churn as customers with 30+ days of inactivity or confirmed cancellation.
- Identified key target KPIs: retention rate and customer lifetime value (CLV).
- Established success criteria: predictive model must achieve recall ≥ 70 % on churned customers.
  
## 2. Data Understanding
- Performed descriptive statistics to understand distribution and outliers
- Conducted missing-value analysis to identify and handle gaps in data
- Merged all datasets using the unified customer_id key for consistency and 
  integration
  
## 3. Data Preprocessing & Feature Engineering
- **Handling Missing Values**: < 1.7 % rows dropped (negligible).
  *Missing Values Heatmap*
  ![image](https://github.com/user-attachments/assets/dc174dfa-7e5f-4e19-8c6c-73a07ad2f639)

- **Standardization**: Centered and scaled continuous variables.
![image](https://github.com/user-attachments/assets/99f711ca-bf92-448c-a4f4-a3c0cfec93d2)

- **Normalization**: Rescaled skewed features (e.g., avg_session_length).
![image](https://github.com/user-attachments/assets/65c33441-5998-4752-96c1-76e191fb16f5)

- **Encoding**: One‑hot for nominal, ordinal for membership tiers.

- **New Features**: Complaint‑resolution latency, rolling 30‑day login 
   frequency, sentiment score via VADER.
## 4. Modeling
- Evaluated five models:

1. **Logistic Regression**: Baseline
![image](https://github.com/user-attachments/assets/0f57b376-1572-4fdd-9bb3-2b173b9accc5)

2. **Random Forest**: Interpretable and robust
![image](https://github.com/user-attachments/assets/e770edc3-e43c-4e97-b78c-b279e38d1507)

3. **XGBoost**: Best overall, ROC-AUC: 0.94, Recall: 0.85
4. **LightGBM**: Fast and memory-efficient
5. **MLP (Keras)**: Good for non-linear patterns

## 5. Evaluation & Interpretation

- **Confusion Matrix**: Assessed the number of true/false positives and negatives to evaluate model accuracy.
- **ROC‑AUC**: Measured the model’s ability to distinguish between churned and retained customers.
- **Precision‑Recall Curves**: Evaluated performance on imbalanced data, focusing on the trade‑off between precision and recall.
- **SHAP Values**: Identified and visualized top features driving churn predictions.

## Exploratory Data Analysis Highlights

- Churn rate peaks at 27 % among Basic members vs. 8 % for Gold.
- Complaints double churn likelihood—even when resolved within SLA.
- Wallet engagement (points earned/redeemed) inversely correlates with churn (Pearson = ‑0.47).
- Urban customers churn 1.6× more than suburban equivalents, likely due to fierce competition.

## Key Insights & Recommendations

- **Upgrade Incentives** - Offer tiered discounts to nudge Basic members to Silver within first 90 days.
- **Complaint Concierge** - Introduce a “white‑glove” resolution team targeting high‑value accounts.
- **Dynamic Loyalty Points** - Gamify wallet usage with streak bonuses and badges.
- **Geo‑Targeted Retention** - Prioritize retention ads in high‑churn metro postcodes.
- **Early‑Warning Triggers** - Automate offers when churn‑probability > 0.65 and loyalty engagement < 10 pts/mo.

# Deployment

**Best Model**: XGBoost
**Serialization**: Joblib used to save and load model

<pre><code>```python import joblib 
  # Save the model joblib.dump(model, 'churn_model.pkl') 
  # Load the model loaded_model = joblib.load('churn_model.pkl') ``` </code></pre>

**Web Service**: Served via Flask API
**Route**: /predict accepts JSON input and returns prediction
**Testing**: Use Postman or curl to validate endpoints

# Limitations & Future Work

- No automation for monthly retraining
- Cold-start issue for new users (< 30 days)
- Imbalanced classes: SMOTE not yet used
- Temporal patterns not captured: explore RNNs, Transformers
- A/B testing for validating retention strategies


   


