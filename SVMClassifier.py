# IMPORT REQUIRED LIBRARIES AND MODULES
import argparse  # for defining command arguments
import pandas as pd
import warnings
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn import decomposition
from sklearn.preprocessing import LabelEncoder

def warn(*args, **kwargs): pass


warnings.warn = warn
train = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')

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

# Swiss army knife function to organize the data

def encode(train, test):
    le = LabelEncoder().fit(train.species)
    labels = le.transform(train.species)  # encode species strings
    classes = list(le.classes_)  # save column names for submission
    test_ids = test.id  # save test ids for submission

    train = train.drop(['species', 'id'], axis=1)
    test = test.drop(['id'], axis=1)

    return train, labels, test, test_ids, classes


train, labels, test, test_ids, classes = encode(train, test)
train.head(1)
print(labels)
pca = decomposition.PCA()
pca.fit(train)
train_t = pca.transform(train)

print(1)

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

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print()

    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=cross_val_k_fold,
                       scoring='%s_macro' % score)
    clf.fit(train_t, labels)

    print("Best parameters set found on development set:")
    print()
    print(clf.best_params_)
    print()
    print("Grid scores on development set:")
    print()
    means = clf.cv_results_['mean_test_score']
    stds = clf.cv_results_['std_test_score']
    for mean, std, params in zip(means, stds, clf.cv_results_['params']):
        print("%0.3f (+/-%0.03f) for %r"
              % (mean, std * 2, params))
    print()

print(clf.best_params_)
print(clf.best_params_['C'])

