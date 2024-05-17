"""
Regression problem using Lasso, GridSearchCV and StandardScaler.
"""
import pandas as pd

from environment import SEED, print, prepare_environment
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.linear_model import Lasso
from sklearn.preprocessing import StandardScaler


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/music_clean.csv', index_col=0)
print('Data head 3:', df.head(3))

# Defining the target and features
X = df.drop('loudness', axis=1).values
y = df['loudness'].values

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa

# Build steps for the pipeline
steps = [
    ('scaler', StandardScaler()),
    ('lasso', Lasso())
]

# Create the pipeline
pipeline = Pipeline(steps)

# Defining the parameters
params = {
    'lasso__alpha': [0.5],
}

# Finding the best params
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)
model_cv = GridSearchCV(pipeline, param_grid=params, cv=kf)
model_cv.fit(X_train, y_train)

# Checking the model parameters
print('Best params:', model_cv.best_params_)
print('Best score:', model_cv.best_score_)

# Make predictions on the test set
y_pred = model_cv.predict(X_test)

# Evaluate the model
print('Score:', model_cv.score(X_test, y_test))
