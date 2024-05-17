"""
Regression problem using KNeighborsClassifier, GridSearchCV and StandardScaler.
"""
import pandas as pd
import numpy as np

from environment import SEED, hprint, print, prepare_environment
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# Prepare environment
prepare_environment()

# Reading the data
hprint('Reading the data')
df = pd.read_csv('datasets/music_clean.csv', index_col=0)
print('Data head 3:', df.head(3))

# Defining the target and features
X = df.drop('genre', axis=1).values
y = df['genre'].values

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa

# Build steps for the pipeline
steps = [
    ('scaler', StandardScaler()),
    ('knn', KNeighborsClassifier())
]

# Create the pipeline
pipeline = Pipeline(steps)

# Defining the parameters
params = {
    'knn__n_neighbors': np.arange(1, 50),
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
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
