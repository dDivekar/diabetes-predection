# Diabetes-Prediction

Predicts diabetes chances in a patient using Machine Learning (Logistic Regression) based on health-related attributes such as age, BMI, HbA1c level, blood glucose level, smoking history, hypertension, and heart disease.

## Features

- Data Cleaning and Preprocessing
- Categorical Feature Encoding
- Outlier Detection and Removal using IQR
- Logistic Regression Model
- Confusion Matrix Visualization
- Classification Report Generation
- Diabetes Risk Prediction

## Technologies Used

- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn

## Dataset

The dataset contains patient health information including:

- Gender
- Age
- Hypertension
- Heart Disease
- Smoking History
- BMI
- HbA1c Level
- Blood Glucose Level
- Diabetes (Target Variable)

## Machine Learning Workflow

1. Load Dataset
2. Data Cleaning
3. Encode Categorical Features
4. Remove Outliers
5. Train-Test Split
6. Train Logistic Regression Model
7. Evaluate Model Performance
8. Generate Predictions

## Evaluation Metrics

- Accuracy Score
- Confusion Matrix
- Precision
- Recall
- F1 Score

## Project Structure

```text
Diabetes-Prediction/
│
├── diabetes_prediction.ipynb
├── diabetes_prediction_dataset.csv
└── README.md
```

## Future Improvements

- Feature Scaling using StandardScaler
- One-Hot Encoding for Categorical Features
- Hyperparameter Tuning
- Cross Validation
- Comparison with Random Forest, SVM, and XGBoost

## Author

Dhananjay Divekar
