"""
Classification problem using KNeighborsClassifier to predict churn
Reviewing how n_neighbors parameter affect the model KNeighborsClassifier
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from environment import SEED, print, prepare_environment
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/telecom_churn_clean.csv')

X = df[['total_day_charge', 'total_eve_charge']].values
y = df['churn'].values
print('Features and target shapes:', X.shape, y.shape)

# Preparing the training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED, stratify=y)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Starting the cycle
train_accuracies = {}
test_accuracies = {}
neighbors = np.arange(1, 26)  # 1, 2, .., 25
for k in neighbors:
    model = KNeighborsClassifier(n_neighbors=k)
    model.fit(X_train, y_train)
    train_accuracies[k] = model.score(X_train, y_train)
    test_accuracies[k] = model.score(X_test, y_test)

# Plotting the results
plt.figure()
plt.title('KNN: Varying Number of Neighbors')
plt.plot(neighbors, train_accuracies.values(), label='Training data')
plt.plot(neighbors, test_accuracies.values(), label='Testing data')
k = 13
plt.axvline(k, ls='--', label=f'n-neighbors={k}')
plt.legend()
plt.xlabel('Number of neighbors')
plt.ylabel('Accuracy')
plt.tight_layout()
plt.show()
