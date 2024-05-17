"""
Regression problem using LinearRegression to predict sales.
Goal is to assess the relationship between the feature and target.
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
df = pd.read_csv('datasets/advertising_and_sales_clean.csv')
print('Data head 3:', df.head(3))

##########################################################
# Making prediction from a single feature
##########################################################
hprint('Making prediction from a single feature')

# Splitting data into training and testing sets
# X_radio = X[:, 3].reshape(-1, 1)
X_radio = df[['radio']].values
y = df['sales'].values
print('Features and target shapes:', X_radio.shape, y.shape)

# Initiating the model
model = LinearRegression()
model.fit(X_radio, y)

# Evaluating the model
score = model.score(X_radio, y)
print('Model Score:', score)

# Predicting
y_pred = model.predict(X_radio)
print('Predictions:', y_pred)

# Plotting resultsplt.figure()
plt.scatter(X_radio, y, label='Dataset')
plt.plot(X_radio, y_pred, c='red', label='Model')
plt.xlabel('Body Mass Index')
plt.ylabel('Blood Glucose (mg/dl)')
plt.title('Glucose vs Body Mass Index')
plt.legend()
plt.tight_layout()


##########################################################
# Making prediction with all features
##########################################################
hprint('Making prediction with all feature')

# Preparing the training and testing set
X = df.drop(['influencer', 'sales'], axis=1).values
y = df['sales'].values
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Initiating the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predicting on testing set
y_pred = model.predict(X_test)
print(f"Predictions: {y_pred[:5]}", f"Actual Values: {y_test[:5]}")

# Evaluating the model
score = model.score(X_test, y_test)
print('Model Score:', score)
calculate_regressionmetrics(y_test, y_pred)

# Plotting all graphs (at the end)
plt.show()
