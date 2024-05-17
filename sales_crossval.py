"""
Regression problem using LinearRegression and cross_val_score to predict sales.
By using cross-validation, we can see how performance varies depending on how
the data is split!
"""
import pandas as pd
import numpy as np

from environment import (SEED, print, prepare_environment,
                         calculate_regressionmetrics)
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/advertising_and_sales_clean.csv')
print('Data head 3:', df.head(3))

# Preparing the training and testing set
X = df.drop(['influencer', 'sales'], axis=1).values
y = df['sales'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Initiating the model
kf = KFold(n_splits=6, shuffle=True, random_state=SEED)
model = LinearRegression()
cv_results = cross_val_score(model, X_train, y_train, cv=kf)

print('Cross Validation Results:', cv_results)
print(f'Mean Score: {np.mean(cv_results)}')
print(f'Std Score: {np.std(cv_results)}')
print(f'95% CI: {np.quantile(cv_results, [0.025, 0.975])}')

model.fit(X_train, y_train)

# Predicting on testing set
y_pred = model.predict(X_test)

# Evaluating the model
score = model.score(X_test, y_test)
print('Observe that the score is inside the reported 95% CI!',
      'Model Score:', score)
calculate_regressionmetrics(y_test, y_pred)
