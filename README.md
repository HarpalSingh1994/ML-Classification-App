# Classification Model Deployment

## Problem Statement
The goal of this project is to predict whether a breast cancer tumor is **Malignant (0)** or **Benign (1)** based on digitized image features of a fine needle aspirate (FNA) of a breast mass. We implement and compare 6 different Machine Learning classifiers and deploy the best solution using Streamlit.

## Dataset Description
* **Source:** Sklearn (Breast Cancer Wisconsin Diagnostic Database)
* **Features:** 30 numeric features (radius, texture, perimeter, area, smoothness, etc.)
* **Instances:** 569 samples
* **Target:** Binary Class (Malignant vs Benign)

## Models Used & Comparison
The following 6 models were trained and evaluated:

| ML Model Name                | Accuracy | AUC    | Precision | Recall | F1 Score | MCC    |
|------------------------------|----------|--------|-----------|--------|----------|--------|
| Logistic Regression          | 0.9825   | 0.9954 | 0.9861    | 0.9861 | 0.9861   | 0.9623 |
| Decision Tree                | 0.9123   | 0.9157 | 0.9559    | 0.9028 | 0.9286   | 0.8174 |
| KNN                          | 0.9561   | 0.9788 | 0.9589    | 0.9722 | 0.9655   | 0.9054 |
| Naive Bayes                  | 0.9386   | 0.9878 | 0.9452    | 0.9583 | 0.9517   | 0.8676 |
| Random Forest (Ensemble)     | 0.9561   | 0.9937 | 0.9589    | 0.9722 | 0.9655   | 0.9054 |
| XGBoost (Ensemble)           | 0.9474   | 0.9917 | 0.9459    | 0.9722 | 0.9589   | 0.8864 |


### Observations
* **Logistic Regression** performed exceptionally well due to the linear separability of the high-dimensional feature space.
* **XGBoost** provided robust results, slightly outperforming the standalone Decision Tree.
* **Random Forest (Ensemble)** showed high recall, while maintaining second highest accuracy as well.

## How to Run
1.  Install dependencies: `pip install -r requirements.txt`
2.  Train models: `python train_models.py` (This generates the `model/` folder)
3.  Run App: `streamlit run app.py`