"""
Ridge Hyperparameter tunning with GridSearchCV
Using the optimal hyperparameters does not guarantee a high performing model!
"""
import pandas as pd
import numpy as np

from environment import SEED, print, prepare_environment
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Lasso


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/diabetes_clean.csv')
print('Data head 3:', df.head(3))

X = df.drop('diabetes', axis=1)
y = df['diabetes']
print('Features and target shapes:', X.shape, y.shape)

# Preparing the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Defining the hyperparameters to tune
param_grid = {
    'alpha': np.linspace(0.00001, 1, 20)
}

# Finding the best params
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

model = Lasso()

model_cv = GridSearchCV(model, param_grid, cv=kf)
model_cv.fit(X_train, y_train)

print('Best params:', model_cv.best_params_)
print('Best score:', model_cv.best_score_)

# Intiating the model with the best params
model = Lasso(**model_cv.best_params_)
model.fit(X_train, y_train)

# Evaluating the model
score = model.score(X_test, y_test)
print('Model Score:', score)
print('CV Model Score:', model_cv.score(X_test, y_test))
