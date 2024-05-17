"""
Binary classification problem using LogisticRegression
"""
import pandas as pd
import matplotlib.pyplot as plt

from environment import SEED, print, prepare_environment
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (classification_report, confusion_matrix,
                             roc_curve, roc_auc_score)


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
model = LogisticRegression().fit(X_train, y_train)

# Predictin on the testing set
y_pred = model.predict(X_test)

# Evaluating the model
score = model.score(X_test, y_test)
print('Model Score:', score)
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
print('Classification Report:', classification_report(y_test, y_pred))

# Predicting probabilities
y_pred_prob = model.predict_proba(X_test)[:, 1]
print('Real values:', y.values[:5])
print('Predicted values:', y_pred[:5])
print('Predicted probailities to be 1:', y_pred_prob[:5])

# Calculating the AUC (Area under the ROC Curve)
auc = roc_auc_score(y_test, y_pred_prob)
print('AUC (Area under the ROC curve):', auc)

# Ploting the ROC curve
fpr, tpr, threshold = roc_curve(y_test, y_pred_prob)
# print('Threshold:', threshold)

plt.figure()
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False positive rate')
plt.ylabel('True Positive Rate')
plt.suptitle('Logistic Regression ROC Curve')
plt.title(f'AUC: {auc}')

# Plotting the graph
plt.show()
