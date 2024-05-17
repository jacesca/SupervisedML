"""
Classification problem using KNeighborsClassifier
"""
import pandas as pd

from environment import SEED, print, prepare_environment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/telecom_churn_clean.csv')
print('Data head 3:', df.head(3))

X = df[['total_day_charge', 'total_eve_charge']].values
y = df['churn'].values
print('Features and target shapes:', X.shape, y.shape)

# Preparing the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Initiating the model
model = KNeighborsClassifier(n_neighbors=7)
model.fit(X_train, y_train)

# Predictin on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
score = model.score(X_test, y_test)
print('Model Score:', score)
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))
# Accuracy = (tp + tn) / (tp + fp + tn + fn)
# Precision = (tp) / (tp + fp) >> positive predicted value
#                                 Maximize tp rate
# Recall = (tp) / (tp + fn) >> Sensitivity
#                              Minimize fn rate
# F1-score =  2 * (precision * recall) / (precision + recall)
#                           >> Harmonic mean of precision and recall
# We can use accuracy when we are interested in predicting both 0 and 1
# correctly and our dataset is balanced enough.
# We use precision when we want the prediction of 1 to be as correct as
# possible and we use recall when we want our model to spot as many real
# 1 as possible.

# Simulating new data
X_new = [[56.8, 17.5], [24.4, 24.1], [50.1, 10.9]]
print('X_new:', X_new)

# Predicting
y_pred = model.predict(X_new)
print('Predictions:', y_pred)
