from statistics import mean
import numpy as np
import random as r
from scipy.special import expit
from sklearn.model_selection import KFold


def extract_features_labels():
    features = []
    labels = []
    with open('spam.data') as input_file:
        lines = input_file.readlines()
        r.shuffle(lines)
        for line in lines:
            newLine = line.strip().split()
            features.append(newLine[0:len(newLine) - 1])
            labels.append(newLine[-1])
    features = np.asarray(features, dtype='float')
    labels = np.asarray(labels, dtype='float')
    return features, labels


class LogisticRegression(object):

    def __init__(self, lr=0.01, n_iter=100, regularization='L2', tolerance=1e-4):
        self.learning_rate = lr
        self.max_iter = n_iter
        self.regularization = regularization
        self.tolerance = tolerance
        self.C = 1 / pow(2, 5)

    def fit(self, X, y):
        self.theta = np.zeros(X.shape[1] + 1)
        X = np.concatenate((np.ones((X.shape[0], 1)), X), axis=1)
        for _ in range(self.max_iter):
            errors = (self.__sigmoid(X @ self.theta)) - y
            N = X.shape[1]

            if self.regularization is not None:
                delta_grad = self.learning_rate * ((self.C * (X.T @ errors)) + np.sum(self.theta))
            else:
                delta_grad = self.learning_rate * (X.T @ errors)

            if np.all(abs(delta_grad) >= self.tolerance):
                self.theta -= delta_grad / N
            else:
                break

        return self

    def predict_proba(self, X):
        return self.__sigmoid((X @ self.theta[1:]) + self.theta[0])

    def predict(self, X):
        return np.round(self.predict_proba(X))

    def __sigmoid(self, z):
        return 1 / (1 + expit(-z))

    def get_params(self):
        try:
            params = dict()
            params['intercept'] = self.theta[0]
            params['coef'] = self.theta[1:]
            return params
        except:
            raise Exception('Fit the model first!')


# X_val, Y_val = extract_features_labels()
# kf5 = KFold(n_splits=5)
# accuracy = 0
# results = []
# for train_index, test_index in kf5.split(X_val):
#     x_train, x_test = X_val[train_index], X_val[test_index]
#     y_train, y_test = Y_val[train_index], Y_val[test_index]
#     clf = SMO(x_train, y_train, 0.02)
#     # clf = LogisticRegression().fit(x_train, y_train)
#     preds = clf.predict(x_test)
#     correct = 0
#     for i, pred in enumerate(preds):
#         if y_test[i] == pred:
#             correct = correct + 1
#     accuracy = (correct / len(x_test)) * 100
#     results.append(accuracy)
# with open('Results.txt', 'a') as f:
#     f.write("Mean Accuracy at C = 2 ^ (5) is: " + str(mean(results)))
