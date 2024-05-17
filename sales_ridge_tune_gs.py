"""
Ridge Hyperparameter tunning with GridSearchCV
"""
import pandas as pd
import numpy as np

from environment import SEED, print, prepare_environment
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import Ridge


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/advertising_and_sales_clean.csv')
print('Data head 3:', df.head(3))

X = df.drop(['influencer', 'sales'], axis=1).values
y = df['sales'].values

# Preparing the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Defining the hyperparameters to tune
param_grid = {
    'alpha': np.arange(0.0001, 1, 10),
    'solver': ['sag', 'lsqr']
}

# Finding the best params
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

model = Ridge()

model_cv = GridSearchCV(model, param_grid, cv=kf)
model_cv.fit(X_train, y_train)

print('Best params:', model_cv.best_params_)
print('Best score:', model_cv.best_score_)

# Intiating the model with the best params
model = Ridge(**model_cv.best_params_)
model.fit(X_train, y_train)

# Evaluating the model
score = model.score(X_test, y_test)
print('Model Score:', score)
print('CV Model Score:', model_cv.score(X_test, y_test))
