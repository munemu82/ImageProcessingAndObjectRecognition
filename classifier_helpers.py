# IMPORT REQUIRED LIBRARIES AND MODULES
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
import warnings
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.utils.multiclass import unique_labels
from sklearn.preprocessing import LabelEncoder


class SVMClassifierParameterTuner:
    def warn(*args, **kwargs):
        pass

    warnings.warn = warn

    def __init__(self, scores, tuned_parameters, k_folds, train_set, labels):  # Input variables
        # Define number of points and radius value
        self.scores = scores
        self.tuned_parameters = tuned_parameters
        self.train_set = train_set
        self.labels = labels
        self.k_folds = k_folds

    def get_best_parameters(self):
        for score in self.scores:
            print("# Tuning hyper-parameters for %s" % score)
            print()

            clf = GridSearchCV(SVC(C=1), self.tuned_parameters, cv=self.k_folds,
                               scoring='%s_macro' % score)
            clf.fit(self.train_set, self.labels)

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

        return clf.best_params_

    def plot_confusion_matrix(self, y_true, y_pred, classes, normalize=False, title=None, cmap=plt.cm.Blues):
        if not title:
            if normalize:
                title = 'Normalized confusion matrix'
            else:
                title = 'Confusion matrix, without normalization'

            # Compute confusion matrix
        cm = confusion_matrix(y_true, y_pred)
        # Only use the labels that appear in the data
        classes = classes[unique_labels(y_true, y_pred)]
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(cm)

        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
        ax.figure.colorbar(im, ax=ax)
        # We want to show all ticks...
        ax.set(xticks=np.arange(cm.shape[1]),
               yticks=np.arange(cm.shape[0]),
               # ... and label them with the respective list entries
               xticklabels=classes, yticklabels=classes,
               title=title,
               ylabel='True label',
               xlabel='Predicted label')

        # Rotate the tick labels and set their alignment.
        plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
                 rotation_mode="anchor")

        # Loop over data dimensions and create text annotations.
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, format(cm[i, j], fmt),
                        ha="center", va="center",
                        color="white" if cm[i, j] > thresh else "black")
        fig.tight_layout()
        return ax

class SVMClassifier:
    def __init__(self, c_val, kernel, train_set, labels):  # Input variables
        # Define number of points and radius value
        self.c_val = c_val
        self.kernel = kernel
        self.train_set = train_set
        self.labels = labels
