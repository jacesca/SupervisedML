"""
Classification problem using KNeighborsClassifier
The goal is to predict whether or not each individual is likely to have
diabetes based on the features body mass index (BMI) and age (in years).
"""
import pandas as pd

from environment import SEED, print, prepare_environment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/diabetes_clean.csv')
print('Data head 3:', df.head(3))

X = df[['bmi', 'age']]
y = df['diabetes']
print('Features and target shapes:', X.shape, y.shape)

# Preparing the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Initiating the model
model = KNeighborsClassifier(n_neighbors=6)
model.fit(X_train, y_train)

# Predictin on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
score = model.score(X_test, y_test)
print('Model Score:', score)
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))
