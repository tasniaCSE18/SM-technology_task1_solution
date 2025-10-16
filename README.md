# SM-technology_task1_solution
Customer Retention Prediction
Project Overview
This project implements a machine learning pipeline to predict customer churn based on demographic and behavioral data. The pipeline includes exploratory data analysis (EDA), preprocessing, feature engineering, model training, and evaluation to identify customers at risk of churning and understand key factors driving churn.
Dataset Description

File: dataset.csv
Rows: 625
Features:
Customer_ID: Unique identifier (dropped during analysis)
Age: Customer age (numeric)
Gender: Customer gender (categorical: Male, Female, Other)
Annual_Income: Yearly income in thousands (numeric)
Total_Spend: Total amount spent (numeric)
Years_as_Customer: Years as a customer (numeric)
Num_of_Purchases: Number of purchases (numeric)
Average_Transaction_Amount: Average purchase amount (numeric)
Num_of_Returns: Number of returns (numeric)
Num_of_Support_Contacts: Number of support interactions (numeric)
Satisfaction_Score: Customer satisfaction score (1-5, numeric)
Last_Purchase_Days_Ago: Days since last purchase (numeric)
Email_Opt_In: Email subscription status (boolean)
Promotion_Response: Response to promotions (categorical: Responded, Ignored, Unsubscribed)


Target: Target_Churn (boolean: True/False, converted to 1/0 for modeling)

Setup Instructions

Install Dependencies:pip install pandas numpy matplotlib seaborn scikit-learn imbalanced-learn xgboost lightgbm torch


Prepare Dataset:
Ensure dataset.csv is in the project directory or upload it to Google Colab.


Run the Script:
For local execution, run python churn_prediction.py.
For Google Colab:
Create a new notebook.
Install dependencies (see Google Colab instructions below).
Upload dataset.csv via the Files tab or programmatically:from google.colab import files
uploaded = files.upload()


Copy and run the script in a code cell.




Outputs:
Console: Data info, statistics, correlations, performance metrics table, and evaluation report.
Visualizations: Saved as PNG files (e.g., churn_distribution.png, cm_LightGBM.png).



Google Colab Instructions

Open Google Colab and create a new notebook.
Install required libraries:!pip install imbalanced-learn xgboost lightgbm torch


Upload dataset.csv using the Files tab or:from google.colab import files
uploaded = files.upload()


Copy the churn_prediction.py script into a code cell and run it.
Download generated plots from the Files tab.

Top Features:
Num_of_Support_Contacts: Higher contacts strongly correlate with churn.
Last_Purchase_Days_Ago: Longer inactivity increases churn risk.
Num_of_Purchases: Fewer purchases reduce retention.


Findings:
Modest model performance due to weak feature-target correlations.
Gradient boosting models (XGBoost, LightGBM) outperform others by capturing interactions.
SMOTE improves recall for imbalanced churn classes (52.8% churn).
Customers with high support contacts or long periods since last purchase are churn risks.
"Other" gender and "Responded" to promotions show slightly higher churn.



Repository Structure

churn_prediction.py: Main script with EDA, preprocessing, model training, and evaluation.
dataset.csv: Input dataset.
*.png: Generated plots (e.g., churn distribution, boxplots, correlation heatmap, confusion matrix, ROC curve).
README.md: This file.

Notes

Run the script to populate exact performance metrics and generate visualizations.
For faster execution, comment out the ANN section or reduce epochs (e.g., from 20 to 10).
To enhance performance, consider hyperparameter tuning or additional feature engineering (e.g., interaction terms).
For questions or issues, refer to the script comments or contact the repository owner.
