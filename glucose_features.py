"""
Feature Selection using Lasso
"""
import pandas as pd
import matplotlib.pyplot as plt

from environment import print, prepare_environment
from sklearn.linear_model import Lasso


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/diabetes_clean.csv')
print('Data head 3:', df.head(3))

X = df.drop('glucose', axis=1)
y = df['glucose']
names = df.drop('glucose', axis=1).columns

# Instantiate Lasso
model = Lasso(alpha=0.1)

# Fitting the model and extract the coefficient
lasso_coef = model.fit(X, y).coef_

plt.figure()
plt.bar(names, lasso_coef)
plt.suptitle('Feature Selection')
plt.title('Diabetes is the most important feature')
plt.tight_layout()
plt.show()
