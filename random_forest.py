# Import the model we are using
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
from image_processing_helpers import *
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV

# Load the data
data_prep_obj = ImageDataPrep()
train_hog = pd.read_csv('train_hog_features.csv', header=None)
test_hog = pd.read_csv('test_hog_features.csv', header=None)
test = pd.read_csv('test_hog_features.csv')
class_labels = data_prep_obj.labels_to_class_num('train_labels.txt')
test_labels = data_prep_obj.labels_to_class_num('test_labels.txt')

# Instantiate model with 1000 decision trees
rf = RandomForestClassifier(n_estimators=1000, min_samples_leaf=5, random_state=42)

# Train the model on training data
rf.fit(train_hog, class_labels)

# Use the forest's predict method on the test data
predictions = rf.predict(test_hog)
# Calculate the absolute errors
errors = abs(predictions - test_labels)

# Print out the mean absolute error (mae)
print('Mean Absolute Error:', round(np.mean(errors), 2), 'degrees.')


# Calculate mean absolute percentage error (MAPE)
accuracy = accuracy_score(test_labels, predictions)
print('Random forest model accuracy is: ' + str(accuracy))

# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start=200, stop=2000, num = 10)]
# Number of features to consider at every split
max_features = ['auto', 'sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(10, 110, num=11)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [2, 5, 10]
# Minimum number of samples required at each leaf node
min_samples_leaf = [1, 2, 4]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(random_grid)

# Use the random grid to search for best hyperparameters
# First create the base model to tune
rf = RandomForestClassifier()
# Random search of parameters, using 3 fold cross validation,
# search across 100 different combinations, and use all available cores
rf_random = RandomizedSearchCV(estimator=rf, param_distributions=random_grid, n_iter=5, cv=3, verbose=2,
                               random_state=42, n_jobs=-1)
# The most important arguments in RandomizedSearchCV are n_iter, which controls the number of different combinations to try


# Fit the random search model
rf_random.fit(train_hog, class_labels)
predictions = rf_random.predict(test_hog)
accuracy = accuracy_score(test_labels, predictions)
print(accuracy)

best_random = rf_random.best_estimator_
random_accuracy = accuracy_score(test_labels, predictions)
print(random_accuracy)

# Using cross validation to tune the parameters
# Create the parameter grid based on the results of random search
param_grid = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}
rf = RandomForestClassifier()
# Instantiate the grid search model
grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
# This will try out 1 * 4 * 2 * 3 * 3 * 4 = 288 combinations of settings.

# Fit the grid search to the data
grid_search.fit(train_hog, class_labels)
# grid_search.best_params_
best_grid = grid_search.best_estimator_
predictions = grid_search.predict(test_hog)
grid_accuracy = accuracy_score(test_labels, predictions)
print(grid_accuracy)
