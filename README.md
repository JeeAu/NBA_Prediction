# Predicting NBA Rookie Career Longevity with Machine Learning  

The project aimed to build a predictive model to estimate whether an NBA rookie will last at least five years in the league based on their first-season statistics.



---

## Project Overview  

Accurately predicting an athlete’s career trajectory can be invaluable for coaches, teams, and analysts.  
This project developed an **end-to-end data analytics and machine learning pipeline** to:

- Clean and preprocess raw NBA rookie statistics  
- Handle missing, duplicate, and outlier values  
- Train and evaluate multiple classifiers  
- Select the best-performing model for prediction  

---

## Methodology  

### 1. Data Mining Problem  

The goal was to predict whether an NBA rookie would last **≥5 years** (target variable `TARGET_5Yrs = 1`) or **<5 years** (`TARGET_5Yrs = 0`) in the league based on game, scoring, rebounding, and playmaking stats.

### 2. Input Data  

Features included:

- **Game Participation:** Games played, minutes played  
- **Scoring:** Points per game, field goals made/attempted, percentages, 3-point stats, free throws  
- **Rebounding:** Offensive, defensive, total rebounds  
- **Playmaking:** Assists, steals, blocks, turnovers  

### 3. Output  

A **binary classification model** predicting:
- `1` → Player lasts 5+ years  
- `0` → Player lasts less than 5 years  

### 4. Data Preparation  

- **Missing Values Analysis:** Used `.info()` and `.isnull()` to detect null values  
- **Duplicate Value Analysis:** Used `.duplicated()`; no duplicates found  
- **Outlier Analysis:** Generated box plots for all columns to identify anomalies  
- **Normalization:** Applied `MinMaxScaler` to scale features to [0,1]  
- **Data Transformation:** Used `OneHotEncoder` to convert categorical variables to numerical  
- **Data Splitting:** Used `train_test_split` (80% train, 20% test) with `random_state=42` for reproducibility  

---

## Models Built  

We experimented with multiple classifiers, using `GridSearchCV` for hyperparameter tuning and `classification_report` for evaluation:

- Decision Tree Classifier  
- K-Nearest Neighbors (KNN)  
- Random Forest Classifier  
- Support Vector Classifier (SVC)  
- Light Gradient Boosting Machine (LGBM)  

Each model followed a systematic pipeline:
1. Preprocess and split data  
2. Define hyperparameter grids  
3. Perform cross-validation with `GridSearchCV`  
4. Train final model on best parameters  
5. Evaluate on validation/test set  

---

## Key Results  

| Model | Accuracy | Recall (Class 1) | F1 Score (Class 1) | AUC |
|-------|---------:|-----------------:|------------------:|----:|
| Decision Tree | 0.84 | 1.0 | 0.91 | 0.68 |
| KNN | 0.83 | 0.99 | 0.90 | 0.64 |
| Random Forest | 0.83 | 0.99 | 0.91 | 0.68 |
| SVC | 0.83 | 1.0 | 0.91 | 0.57 |
| **LGBM (best)** | **0.84** | **1.0** | **0.91** | **0.69** |

- Most models performed strongly on **Class 1** (players lasting ≥5 years)  
- All models struggled with **Class 0** (players lasting <5 years) due to class imbalance  
- **LGBM achieved the highest overall metrics** and was selected as the final model  

---

## Impact  

This project demonstrates how data analytics can be used to forecast sports careers, enabling:

- More informed draft and training decisions  
- Early identification of at-risk rookies  
- Evidence-based player development strategies  

---

## Technologies Used  

- **Python (Pandas, NumPy, Matplotlib)**  
- **scikit-learn** (Decision Tree, KNN, Random Forest, SVC, preprocessing, GridSearchCV)  
- **LightGBM**  
- **OneHotEncoder** & **MinMaxScaler** for data transformation  
- **Classification Reports / ROC Curves** for evaluation  

---

## Future Work  

- Address **class imbalance** using SMOTE or other resampling techniques  
- Explore additional features (college stats, injuries, etc.)  
- Test ensemble approaches combining multiple models  
- Deploy the model as an API or interactive dashboard for team analysts  

---

### If you find this project interesting, feel free to fork it or star it on GitHub!
