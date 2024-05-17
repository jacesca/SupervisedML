"""
Regularized Regression using Lasso
"""
import pandas as pd

from environment import SEED, print, prepare_environment
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Lasso


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/diabetes_clean.csv')
print('Data head 3:', df.head(3))

# Preparing the training and testing set
X = df.drop('glucose', axis=1)
y = df['glucose']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa
print('Training set shape:', X_train.shape, y_train.shape)
print('Testing set shape:', X_test.shape, y_test.shape)

# Evaluating the hyperparameter alpha
alphas = [0.01, 1.0, 10.0, 20.0, 50.0]
scores = []
for alpha in alphas:
    model = Lasso(alpha=alpha)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print('Scores:', scores)
