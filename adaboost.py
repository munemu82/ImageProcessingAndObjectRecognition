# Load libraries
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from image_processing_helpers import *
from classifier_helpers import *
# Import train_test_split function
from sklearn.model_selection import train_test_split
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics
# Import Support Vector Classifier
from sklearn.svm import SVC
#Import scikit-learn metrics module for accuracy calculation
from sklearn import metrics

# Load data
data_prep_obj = ImageDataPrep()
train_hog = pd.read_csv('train_hog_features.csv', header=None)
test_hog = pd.read_csv('test_hog_features.csv', header=None)
test = pd.read_csv('test_hog_features.csv')
class_labels = data_prep_obj.labels_to_class_num('train_labels.txt')
test_labels = data_prep_obj.labels_to_class_num('test_labels.txt')

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50, # The most important parameters are base_estimator, n_estimators, and learning_rate
                         learning_rate=1)
# Train Adaboost Classifer
model = abc.fit(train_hog, class_labels)

# Predict the response for test dataset
y_pred = model.predict(test_hog)

# Model Accuracy, how often is the classifier correct?
print("Accuracy for decision tree base learner:", metrics.accuracy_score(test_labels, y_pred))

# Using Different Base Learners
svc = SVC(probability=True, kernel='linear')

# Create adaboost classifer object
abc = AdaBoostClassifier(n_estimators=50, base_estimator=svc, learning_rate=1)

# Train Adaboost Classifer
model = abc.fit(train_hog, class_labels)

# Predict the response for test dataset
y_pred = model.predict(test_hog)


# Model Accuracy, how often is the classifier correct?
print("Accuracy for SVC base learner: ", metrics.accuracy_score(test_labels, y_pred))

from sklearn import datasets

breast_cancer = datasets.load_breast_cancer()
X_cancer = breast_cancer.data
y_cancer = breast_cancer.target

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.svm import SVC

models1 = {
    'ExtraTreesClassifier': ExtraTreesClassifier(),
    'RandomForestClassifier': RandomForestClassifier(),
    'AdaBoostClassifier': AdaBoostClassifier(),
    'GradientBoostingClassifier': GradientBoostingClassifier(),
    'SVC': SVC()
}

params1 = {
    'ExtraTreesClassifier': { 'n_estimators': [16, 32] },
    'RandomForestClassifier': { 'n_estimators': [16, 32] },
    'AdaBoostClassifier':  { 'n_estimators': [16, 32] },
    'GradientBoostingClassifier': { 'n_estimators': [16, 32], 'learning_rate': [0.8, 1.0] },
    'SVC': [
        {'kernel': ['linear'], 'C': [1, 10]},
        {'kernel': ['rbf'], 'C': [1, 10], 'gamma': [0.001, 0.0001]},
    ]
}
helper1 = EstimatorSelectionHelper(models1, params1)
helper1.fit(X_cancer, y_cancer, scoring='f1', n_jobs=2)
helper1.score_summary(sort_by='max_score')