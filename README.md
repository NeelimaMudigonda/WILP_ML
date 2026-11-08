# Online Shopper Purchase Intention: Classification with Five From-Scratch Models
 
**Course:** Machine Learning (S1-26_AIMLZG565) | **Assignment 2**

**Student ID:** 2026AK05030 | **Student Name:** Mudigonda Neelima
---
## a. Problem statement
Online shops lose many visitors who browse but never buy. The task is to predict, from a visitor's
browsing behaviour in a session, whether the session ends in a purchase (**Buy = 1**) or not
(**No Buy = 0**). This is a **binary classification** problem. Five classifiers were implemented
**from scratch using NumPy** (no scikit-learn) and compared on six evaluation metrics.


## b. Dataset description
- **Source:** UCI Machine Learning Repository, *Online Shoppers Purchasing Intention Dataset*
- **Original size:** 12,330 sessions, 17 raw features plus the target `Revenue`
- **After preprocessing:** 19 input features, target renamed `class` (0 = No Buy, 1 = Buy)
- **Class balance:** about 85% No Buy and 15% Buy (imbalanced)
- **Problem type:** Binary classification
- 
**Preprocessing steps**
1. `Revenue` (False/True) converted to 0/1 and renamed `class`.
2. `Month` names converted to numbers 1 to 12, `Weekend` converted to 0/1.
3. `VisitorType` one-hot encoded into 3 columns (17 features became 19).
4. Duplicate rows removed, then a stratified sample of 3,000 rows taken (keeps the class ratio and keeps KNN light).
5. Stratified 80/20 split: 2,401 train rows and 599 test rows. Features standardised using the training mean and standard deviation.
   
**Input features:** Administrative, Administrative_Duration, Informational, Informational_Duration,
ProductRelated, ProductRelated_Duration, BounceRates, ExitRates, PageValues, SpecialDay, Month,
OperatingSystems, Browser, Region, TrafficType, Weekend, VisitorType_New_Visitor,
VisitorType_Returning_Visitor, VisitorType_Other.


## c. GitHub repository
Repository: https://github.com/NeelimaMudigonda/WILP_ML
Streamlit app: https://NeelimaMudigonda.streamlit.app
```
WILP_ML/
|-- app.py                 Streamlit app
|-- my_models.py           From-scratch models, scaler and metrics
|-- model_training.ipynb   Training and evaluation notebook
|-- requirements.txt
|-- test_data.csv          Test set (599 rows) for the app
|-- .streamlit/config.toml
|-- model/                 Saved models, scaler, results.csv, confusion_matrices.png


## d. Models used
All metrics are calculated on the 599-row test set.
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8932 | 0.8777 | 0.7843 | 0.4301 | 0.5556 | 0.5299 |
| Decision Tree | 0.8765 | 0.8859 | 0.6301 | 0.4946 | 0.5542 | 0.4885 |
| KNN | 0.8815 | 0.7981 | 0.7750 | 0.3333 | 0.4662 | 0.4578 |
| Naive Bayes | 0.7546 | 0.8137 | 0.3500 | 0.6774 | 0.4615 | 0.3524 |
| Support Vector Machine | 0.8948 | 0.8431 | 0.8000 | 0.4301 | 0.5594 | 0.5373 |
**Settings:** Logistic Regression (learning rate 0.1, 2000 iterations), Decision Tree (Gini, max depth 6),
KNN (k = 7, Euclidean distance), Gaussian Naive Bayes, SVM (linear kernel, hinge loss, lambda 0.01, 2000 iterations).


### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Accuracy 0.8932 and AUC 0.8777. High precision (0.7843) with only 11 false alarms, but recall is 0.4301, so 53 of 93 buyers were missed. Second best MCC (0.5299). A reliable, conservative linear model. |
| Decision Tree | Best AUC (0.8859) and it catches more buyers than Logistic Regression (recall 0.4946, 46 of 93). It makes 27 false alarms, so precision falls to 0.6301 and MCC to 0.4885. |
| KNN | Weakest ranking (AUC 0.7981) and lowest recall (0.3333): only 31 of 93 buyers found. About 85% of rows are No Buy, so most of the 7 neighbours vote No Buy. Fewest false alarms (9) but most missed buyers. |
| Naive Bayes | Highest recall (0.6774, 63 of 93 buyers caught) but lowest accuracy (0.7546), precision (0.3500) and MCC (0.3524) because of 117 false alarms. Correlated features (page counts and durations) break the independence assumption. |
| Support Vector Machine | Best Accuracy (0.8948), F1 (0.5594) and MCC (0.5373), precision 0.8000 with only 10 false alarms. The linear margin generalises well, but recall (0.4301) is modest because of the class imbalance. |
| **Overall winner** | **Support Vector Machine.** It leads on Accuracy, F1 and MCC (MCC is the most reliable metric on imbalanced data). Logistic Regression is very close behind and Decision Tree has the best AUC. If reaching every buyer matters more than false alarms, Naive Bayes is the better choice. |


## Streamlit app
Features: CSV upload (test data), model selection dropdown, six evaluation metrics, confusion matrix
and classification report, comparison of all five models, and a Performance tab with the tables above.
Models are loaded from `model/` and are **not retrained** on the uploaded data.


## How to run locally
pip install -r requirements.txt
streamlit run app.py


## Note on implementation
All five models, the scaler, the train/test split and all six metrics are written from scratch with NumPy in `my_models.py` and the notebook. 
