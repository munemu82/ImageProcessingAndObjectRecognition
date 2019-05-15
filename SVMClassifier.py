# IMPORT REQUIRED LIBRARIES AND MODULES
import argparse  # for defining command arguments
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from classifier_helpers import SVMClassifierParameterTuner
from image_processing_helpers import *
from datetime import datetime
from sklearn.metrics import accuracy_score

startTime = datetime.now()

# train = pd.read_csv('train.csv')
train_lbp = pd.read_csv('train_lbp_features.csv', header=None)
test_lbp = pd.read_csv('test_lbp_features.csv', header=None)
test = pd.read_csv('test.csv')

data_prep_obj = ImageDataPrep()
class_labels = data_prep_obj.labels_to_class_num('train_labels.txt')
class_names = data_prep_obj.get_unique_class_names('class_names.txt')
class_names = np.array(class_names)
print(type(class_names))
print(class_names)
test_labels = data_prep_obj.labels_to_class_num('test_labels.txt')

# Initialize and setting up command arguments# parse cmd args
parser = argparse.ArgumentParser(
    description=" Beginning data preparation"
)
# Add command prompt argument variables
parser.add_argument('--kfold_cv', action="store", dest="kfold_cv", required=True)  # Must be full path
parser.add_argument('--c_multiplier', action="store", dest="c_multiplier", required=True)  # Must be a full path
args = vars(parser.parse_args())

# Get the user input paths from command prompt
cross_val_k_fold = args['kfold_cv']      # python MLClassifier --kfold_cv 5 --c_multiplier 10
multiplier_for_c = args['c_multiplier']

cross_val_k_fold = int(cross_val_k_fold)
multiplier_for_c = int(multiplier_for_c)

# Set the parameters by cross-validation
tuned_parameters = [{'kernel': ['rbf'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.001*multiplier_for_c, 0.001*(multiplier_for_c**2),
                           0.001*(multiplier_for_c**3), 0.001*(multiplier_for_c**4), 0.001*(multiplier_for_c**5),
                           0.001*(multiplier_for_c**6), 0.001*(multiplier_for_c**7)]},
                    {'kernel': ['sigmoid'], 'gamma': [1e-2, 1e-3, 1e-4, 1e-5],
                     'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]},
                    {'kernel': ['linear'], 'C': [0.001, 0.10, 0.1, 10, 25, 50, 100, 1000]}
                   ]

scores = ['precision', 'recall']
print(class_labels.shape)
print(train_lbp.shape)
# Perform parameter tuning to find best parameter combinations
classifer_obj = SVMClassifierParameterTuner(scores, tuned_parameters,cross_val_k_fold, train_lbp, class_labels)
# print(classifer_obj.get_best_parameters())
np.set_printoptions(precision=2)

linear_svm = SVC(C=1000, kernel="linear", probability=True)
# Perform classifier evaluations

y_pred = linear_svm.fit(train_lbp, class_labels).predict(test_lbp)
# Plot non-normalized confusion matrix
#classifer_obj.plot_confusion_matrix(test_labels, y_pred, classes=class_names,
                     # title='Linear SVM Confusion matrix, without normalization')

# Plot normalized confusion matrix
classifer_obj.plot_confusion_matrix(test_labels, y_pred, classes=class_names, normalize=True,
                      title='Linear SVM Normalized confusion matrix')

linear_svm_accuracy = accuracy_score(test_labels, y_pred)
print('The Linear SVM model accuracy is: ' + str(linear_svm_accuracy))

# RBF CLASSIFIER

rbf_svm = SVC(gamma=1e-4, C=1000, kernel="rbf", probability=True)
y_pred = rbf_svm.fit(train_lbp, class_labels).predict(test_lbp)
classifer_obj.plot_confusion_matrix(test_labels, y_pred, classes=class_names, normalize=True,
                      title='RBF SVM Normalized confusion matrix')
rbf_svm_accuracy = accuracy_score(test_labels, y_pred)
print('The RBF SVM model accuracy is: ' + str(rbf_svm_accuracy))

est_time = datetime.now() - startTime
print('It took about ' + str(est_time) + ' seconds')
plt.show()

