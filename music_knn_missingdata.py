"""
Regression problem using KNeighborsClassifier.
"""
import pandas as pd

from environment import SEED, hprint, print, prepare_environment
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import confusion_matrix


# Prepare environment
prepare_environment()

# Reading the data
hprint('Reading the data')
df = pd.read_csv('datasets/music_dirty.csv')
print('Data types:', df.info())
print('Data shape:', df.shape)
print('Data head 3:', df.head(3))
print('Null values in each feature:', df.isna().sum().sort_values())

hprint('Handling missing data')
# Remove values for all columns with 50 or fewer missing values.
df_clean = df.copy().dropna(subset=['genre', 'popularity', 'loudness',
                                    'liveness', 'tempo'])
print('Data shape:', df_clean.shape)

# Convert genre to a binary feature
df_clean['genre'] = df_clean['genre'].apply(lambda genre: int(genre == 'Rock'))

# Defining the target and features
X = df_clean.drop('genre', axis=1).values
y = df_clean['genre'].values

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa

# Build steps for the pipeline
steps = [
    ('imputation', SimpleImputer()),
    ('scaler', StandardScaler()),  # To improve the model  # noqa
    ('knn', KNeighborsClassifier(n_neighbors=3))
]

# Create the pipeline
pipeline = Pipeline(steps)

# Fit the pipeline to the training data
pipeline.fit(X_train, y_train)

# Make predictions on the test set
y_pred = pipeline.predict(X_test)

# Evaluate the model
print('Score:', pipeline.score(X_test, y_test))
print('Confusion Matrix:', confusion_matrix(y_test, y_pred))
