"""
Regression problem using LinearRegression.
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from environment import SEED, hprint, print, prepare_environment
from sklearn.model_selection import train_test_split, cross_val_score, KFold
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/music.csv')
hprint('Reading the data')
print('Data head 3:', df.head(3))
print('Genre distribution in dataset:', df.genre.value_counts(normalize=True))

# Reviewing the gender column
plt.figure()
sns.boxplot(data=df, x='genre', y='popularity', hue='genre')
plt.xlabel('')
plt.ylabel('popularity')
plt.title('Music Popularity by Genre')
plt.tight_layout()

# Encoding dummy variables
hprint('Encoding dummy variables using Pandas: option 1')
df_dummies = pd.get_dummies(df.genre, drop_first=True,
                            dtype=int, prefix='genre_')
df_dummies = pd.concat([df, df_dummies], axis=1)
df_dummies.drop('genre', axis=1, inplace=True)
print('Data head 3 (after encoding):', df_dummies.head(3))

# or
hprint('Encoding dummy variables using Pandas: option 2')
df_dummies = pd.get_dummies(df, drop_first=True, dtype=int, prefix='genre_')
print('Df dummies:', df_dummies.head(3))

# or
hprint('Encoding dummy variables using scikit-learn: option 3')
onehot = OneHotEncoder()
encoded = onehot.fit_transform(df[['genre']]).toarray()
df_dummies = df.drop(['genre'], axis=1)
df_dummies[onehot.get_feature_names_out()] = encoded
print('Df dummies:', df_dummies.head(3))

# Splitting data into training and testing sets
X = df_dummies.drop('popularity', axis=1)
y = df_dummies.popularity

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)  # noqa

# Initiating the model
hprint('Working with the LinearRegression Model')
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
model = LinearRegression()
# cv_results = cross_val_score(model, X_train, y_train, cv=kf,
#                              scoring='neg_mean_squared_error') # metrics.mean_squared_error MSE  # noqa
cv_results = cross_val_score(model, X_train, y_train, cv=kf,
                             scoring='neg_root_mean_squared_error') # RMSE  # noqa

# Evaluating the model
# result = np.sqrt(-cv_results) # When MSE to get RMSE
result = -cv_results
print('Cross Validation Results:', result)
print(f'Mean Score: {np.mean(result)}')
print(f'Std Score: {np.std(result)}')
print(f'95% CI: {np.quantile(result, [0.025, 0.975])}')

# Plotting graph
plt.show()
