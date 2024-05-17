"""
Regression problem using Lasso, GridSearchCV and StandardScaler.
"""
import pandas as pd
import matplotlib.pyplot as plt

from environment import (SEED, print, prepare_environment,
                         calculate_regressionmetrics)
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso


# Prepare environment
prepare_environment()

# Reading the data
df = pd.read_csv('datasets/music_clean.csv', index_col=0)
print('Data head 3:', df.head(3))

# Defining the target and features
X = df.drop('energy', axis=1).values
y = df['energy'].values

# Split data into training and testing set
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=SEED)  # noqa

# Scaling the data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Evaluating different models
models = {
    'Linear Regression': LinearRegression(),
    'Ridge': Ridge(alpha=0.1),
    'Lasso': Lasso(0.1)
}
results = []
for model in models.values():
    kf = KFold(n_splits=6, random_state=SEED, shuffle=True)
    cv_results = cross_val_score(model, X_train_scaled, y_train, cv=kf)
    results.append(cv_results)

# Plotting the results
plt.figure()
plt.boxplot(results, labels=models.keys())
plt.xlabel('Models')
plt.ylabel('Accuracy')
plt.title('Model Performance Comparisson')
plt.tight_layout()

# Testing the model performance
for name, model in models.items():
    model.fit(X_train_scaled, y_train)

    # Make predictions on the test set
    y_pred = model.predict(X_test_scaled)

    # Evaluate the model
    print(f'{name}:', end='\n')
    calculate_regressionmetrics(y_test, y_pred)

# Plotting the graph
plt.show()
