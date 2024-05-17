"""
Ridge Hyperparameter tunning with RandomizedSearchCV
"""
import pandas as pd
import numpy as np

from environment import SEED, print, prepare_environment
from sklearn.model_selection import train_test_split, KFold, RandomizedSearchCV
from sklearn.linear_model import LogisticRegression

from warnings import simplefilter
from sklearn.exceptions import ConvergenceWarning


# Prepare environment
prepare_environment()
simplefilter("ignore", category=ConvergenceWarning)

# Reading the data
df = pd.read_csv('datasets/diabetes_clean.csv')
print('Data head 3:', df.head(3))
print('Target value frequency:', df.diabetes.value_counts(normalize=True))

X = df.drop('diabetes', axis=1)
y = df['diabetes']
print('Features and target shapes:', X.shape, y.shape)

# Preparing the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Defining the hyperparameters to tune
param_grid = {
    "penalty": ["l1", "l2"],
    "tol": np.linspace(0.0001, 1.0, 50),
    "C": np.linspace(0.1, 1.0, 50),
    "class_weight": ["balanced", {0: 0.8, 1: 0.2}, {0: 0.65, 1: 0.35}],
    "solver": ['liblinear', 'saga'],
}
# # or add a list of params
# param_grid = [
#     {
#         "penalty": ["l1", "l2"],
#         "tol": np.linspace(0.0001, 1.0, 50),
#         "C": np.linspace(0.1, 1.0, 50),
#         "class_weight": ["balanced", {0: 0.8, 1: 0.2}, {0: 0.65, 1: 0.35}],
#         "solver": ['liblinear', 'saga'],
#     },
#     {
#         "penalty": ["l2"],
#         "tol": np.linspace(0.0001, 1.0, 50),
#         "C": np.linspace(0.1, 1.0, 50),
#         "class_weight": ["balanced", {0: 0.8, 1: 0.2}, {0: 0.65, 1: 0.35}],
#         "solver": ['lbfgs', 'newton-cg', 'newton-cholesky', 'sag'],
#     },
# ]

# Finding the best params
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

model = LogisticRegression()

model_cv = RandomizedSearchCV(model, param_grid, cv=kf)
model_cv.fit(X_train, y_train)

print('Best params:', model_cv.best_params_)
print('Best score:', model_cv.best_score_)

# Intiating the model with the best params
model = LogisticRegression(**model_cv.best_params_)
model.fit(X_train, y_train)

# Evaluating the model
score = model.score(X_test, y_test)
print('Model Score:', score)
print('CV Model Score:', model_cv.score(X_test, y_test))
