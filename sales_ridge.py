"""
Regularized Regression using Ridge
"""
import pandas as pd

from environment import SEED, print, prepare_environment
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge


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

# Evaluating the hyperparameter alpha
alphas = [0.1, 1.0, 10.0, 100.0, 1000.00]
scores = []
for alpha in alphas:
    model = Ridge(alpha=alpha)
    model.fit(X_train, y_train)
    scores.append(model.score(X_test, y_test))

print('Scores:', scores)
