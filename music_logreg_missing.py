"""
Regression problem using LogisticRegression.
"""
import pandas as pd
import numpy as np

from environment import SEED, hprint, print, prepare_environment
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


# Prepare environment
prepare_environment()

# Reading the data
hprint('Reading the data')
df = pd.read_csv('datasets/music_dirty.csv')
print('Data types:', df.info())
print('Data head 3:', df.head(3))
print('Null values in each feature:', df.isna().sum().sort_values())

hprint('Handling missing data')
# Dropping rows with missing values
df_clean = df.copy().dropna(subset=['genre', 'popularity',
                                    'loudness', 'liveness', 'tempo'])
print('Null values in each feature:', df_clean.isna().sum().sort_values())

# Imputing values
# Splitting between categorical and numerical features
X_cat = df[['genre']]
X_num = df.drop(['genre', 'popularity'], axis=1)
y = df['popularity']

# Splitting data into training and testing sets
X_train_c, X_test_c, y_train, y_test = train_test_split(X_cat, y, test_size=0.2, random_state=SEED)  # noqa
X_train_n, X_test_n, y_train, y_test = train_test_split(X_num, y, test_size=0.2, random_state=SEED)  # noqa

# Instantiate the imputer
imp_cat = SimpleImputer(strategy='most_frequent')
X_train_cat = imp_cat.fit_transform(X_train_c)
X_test_cat = imp_cat.transform(X_test_c)

imp_num = SimpleImputer()  # mean by default
X_train_num = imp_num.fit_transform(X_train_n)
X_test_num = imp_num.transform(X_test_n)

X_train = np.append(X_train_num, X_train_cat, axis=1)
X_test = np.append(X_test_num, X_test_cat, axis=1)
print('Final result (3 first rows):', X_train[:3])

# or preserving the pandas structure
imp_cat = SimpleImputer(strategy='most_frequent')
imp_num = SimpleImputer()

X = df.drop('popularity', axis=1)
y = df['popularity']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)  # noqa

imp_cat.fit(X_train.select_dtypes(include=[object]))
imp_num.fit(X_train.select_dtypes(exclude=[object]))

X_train[imp_cat.get_feature_names_out()] = imp_cat.transform(X_train.select_dtypes(include=[object]))  # noqa
X_train[imp_num.get_feature_names_out()] = imp_num.transform(X_train.select_dtypes(exclude=[object]))  # noqa

X_test[imp_cat.get_feature_names_out()] = imp_cat.transform(X_test.select_dtypes(include=[object]))  # noqa
X_test[imp_num.get_feature_names_out()] = imp_num.transform(X_test.select_dtypes(exclude=[object]))  # noqa
print('Final result (preserving the dataframe)',
      X_train.isna().sum().sort_values(),
      X_test.isna().sum().sort_values())

# or in a pipeline
clean_df = df.copy().dropna(subset=['genre', 'popularity', 'loudness', 'liveness', 'tempo'])  # noqa
# # clean_df['genre'] = np.where(clean_df.genre == 'Rock', 1, 0)
clean_df['genre'] = clean_df['genre'].apply(lambda genre: int(genre == 'Rock'))

X = clean_df.drop('genre', axis=1).values
y = clean_df['genre'].values

steps = [
    ('imputation', SimpleImputer()),
    ('scaler', StandardScaler()),  # To avoid warning: Increase the number of iterations (max_iter) or scale the data as shown in  # noqa
    ('logistic_regression', LogisticRegression())
]
pipeline = Pipeline(steps)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa

pipeline.fit(X_train, y_train)
print('Score:', pipeline.score(X_test, y_test))
