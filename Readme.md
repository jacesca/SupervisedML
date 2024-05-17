# Supervised Machine Learning

Features:
- Supervised Algorithms
    - Binary Classification
- Regressions
    - LinearRegression
    - Regularized Regression
- scikit-learn models
    - Classification problems
        - KNeighborsClassifier
        - DecisionTreeClassifier
    - Linear Regression problems
        - Ridge
        - Lasso
            - Feature Selection
    - Logistic Regression
        - ROC Curve
- Metrics
    - Classification problems
        - Accuracy
        - Precision
        - Recall
        - F1-score
        - Confusion Matrix
        - Classification Report
    - Linear Regression problems
        - Mean Squared Error MSE
        - Root Mean Squared Error RMSE
        - MAE
        - R2 Score
- Hyperparameters tunning
    - GridSearchCV
    - RandomizedSearchCV
- Preprocessing
    - Encoding Dummy Variables
        - OneHotEncoder from scikit-learn
        - getDummies from pandas
    - Handling missing data
        - Dropping missing data
        - Imputing values
            - SimpleImputer
        - Normalizing (scaling) or Standarizing (centering) data
            - StandardScaler
    - Models
        - Affected by scaling
            - KNN
            - LinearRegression
            - LogisticRegression
            - Artificial Neural Network

## Run ML model
```
python churn_kneighbors.py          # Bynary Classification with KNeighborsClassifier
python churn_n-neighbors.py         # Bynary Classification with KNeighborsClassifier 
python glucose_pred.py              # Logistic Regression with LinearRegression
python sales_linearreg.py           # Logistic Regression with LinearRegression
python sales_crossval.py            # LinearRegression and cross_val_score
python sales_ridge.py               # Regularized Regression with Ridge
python glucose_ridge.py             # Regularized Regression with Ridge
python sales_lasso.py               # Regularized Regression with Lasso
python glucose_lasso.py             # Regularized Regression with Lasso
python glucose_feature.py           # Lasso for feature selection
python sales_feature.py             # Lasso for feature selection
python diabetes_kneighbors.py       # Bynary Classification with KNeighborsClassifier
python churn_logisticreg.py         # Bynary Classification with LogisticRegression 
python diabetes_logisticreg.py      # Bynary Classification with LogisticRegression 
python sales_ridge_tune_gs.py       # Hyperparameter tunning using GridSearchCV
python sales_ridge_tune_rs.py       # Hyperparameter tunning using RandomizedSearchCV
python diabetes_lasso_gs.py         # Hyperparameter tunning using GridSearchCV
python diabetes_logisticreg_rs.py   # Hyperparameter tunning using RandomizedSearchCV
python music_linearreg.py           # Linear Regression with dummies variables
python music_ridge.py               # Ridge Regression with dummies variable
python music_logreg_missing.py      # LogisticRegression with missed data
python music_knn_missingdata.py     # KNeighborsClassifier with missed data
python music_knn_gs.py              # KNN with GridSearchCV and StandardScaler
python music_lasso_gs.py            # Lasso with GridSearchCV and StandardScaler
python music_logreg_gs.py           # LogReg. with GridSearchCV and StandardScaler
python music_classifier_eval.py     # Comparing KNN, LogisticReg, and DecisionTree
python music_linreg_eval.py         # Comparing LinearRegression, Ridge, Lasso
```

## Installing using GitHub
- Fork the project into your GitHub
- Clone it into your dektop
```
git clone https://github.com/jacesca/SupervisedML.git
```
- Setup environment (it requires python3)
```
python -m venv venv
source venv/bin/activate  # for Unix-based system
venv\Scripts\activate  # for Windows
```
- Install requirements
```
pip install -r requirements.txt
```

## Others
- Proyect in GitHub: https://github.com/jacesca/SupervisedML
- Commands to save the environment requirements:
```
conda list -e > requirements.txt
# or
pip freeze > requirements.txt

conda env export > flask_env.yml
```
- For coding style
```
black model.py
flake8 model.py
```

## Extra documentation
- [Customizing Matplotlib rcparams](https://matplotlib.org/stable/users/explain/customizing.html)
- [Precision, recall, accuracy. How to choose?](https://www.yourdatateacher.com/2021/06/07/precision-recall-accuracy-how-to-choose/#:~:text=We%20can%20use%20accuracy%20when,many%20real%201%20as%20possible.)
> We can use accuracy when we are interested in predicting both 0 and 1 correctly and our dataset is balanced enough. We use precision when we want the prediction of 1 to be as correct as possible and we use recall when we want our model to spot as many real 1 as possible. 
>
> Ex. of Recall cases (Minimize false negatives)
> - A model predicting the presence of cancer as the positive class.
> - A classifier predicting the positive class of a computer program containing malware.
>
> Ex. of Precision cases (Maximize true positives)
> - A model predicting if a customer is a high-value lead for a sales team with limited capacity.
>   * With limited capacity, the sales team needs the model to return the highest proportion of true positives compared to all predicted positives, thus minimizing wasted effort.
- [Metrics and scoring: quantifying the quality of predictions](https://scikit-learn.org/stable/modules/model_evaluation.html)