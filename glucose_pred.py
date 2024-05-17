"""
Regression problem using LinearRegression to predict glucose.
"""
import pandas as pd
import matplotlib.pyplot as plt

from environment import (SEED, print, hprint, prepare_environment,
                         calculate_regressionmetrics)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/diabetes_clean.csv')
print('Data head 3:', df.head(3))

X = df.drop('glucose', axis=1)
y = df['glucose']
print('Features and target shapes:', X.shape, y.shape)

##########################################################
# Making prediction from a single feature
##########################################################
# Goal is to assess the relationship between the feature and target.
hprint('Making prediction from a single feature')

# Splitting data into training and testing sets
# X_bmi = X[:, 3].reshape(-1, 1)
X_bmi = X[['bmi']]
print('Features and target shapes:', X_bmi.shape, y.shape)

# Initiating the model
model = LinearRegression()
model.fit(X_bmi, y)

# Evaluating the model
score = model.score(X_bmi, y)
print('Model Score:', score)

# Predicting
y_pred = model.predict(X_bmi)

# Plotting resultsplt.figure()
plt.scatter(X_bmi, y, label='Dataset')
plt.plot(X_bmi, y_pred, c='red', label='Model')
plt.xlabel('Body Mass Index')
plt.ylabel('Blood Glucose (mg/dl)')
plt.title('Glucose vs Body Mass Index')
plt.legend()
plt.tight_layout()


##########################################################
# Making prediction with feature
##########################################################
hprint('Making prediction with all feature')

# Preparing the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Initiating the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on testing set
y_pred = model.predict(X_test)

# Evaluating the model
score = model.score(X_test, y_test)
print('Model Score:', score)
calculate_regressionmetrics(y_test, y_pred)

# Plotting all graphs (at the end)
plt.show()
