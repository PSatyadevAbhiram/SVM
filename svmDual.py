import copy
import math

from sklearn.model_selection import KFold
import time
from logisticRegression import extract_features_labels
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

import numpy as np
import random as rand


def accuracy(predicted, y_test):
    error = 0
    for i in range(predicted.shape[0]):
        check = np.where(predicted[i] == max(predicted[i]))[0][0]
        if check == 0:
            if y_test[i] != 0:
                error += 1
        elif check == 1:
            if y_test[i] != 1:
                error += 1
    return 1 - (error / predicted.shape[0])


def predict(predicted):
    y_predicted = np.zeros((predicted.shape[0]))
    for i in range(predicted.shape[0]):
        y_predicted[i] = np.where(predicted[i] == max(predicted[i]))[0][0]
    return y_predicted


def labelEncoding(y):
    y_encoded = np.zeros((y.shape[0]))
    for i in range(y.shape[0]):
        if y[i] == 0:
            y_encoded[i] = 0
        elif y[i] == 0:
            y_encoded[i] = 1
    return y_encoded


def f_x(X, y, a, b, x, degree):
    predicted_value = 0.0
    # using polynomial kernel
    for k in range(X.shape[0]):
        predicted_value += (a[k] * y[k] * ((X[k, :].T @ x + 1) ** degree))
    return predicted_value + b


def SMO(X, y, C, tol=math.pow(10, -3), max_passes=5, degree=3):
    # size of input vector in feature space
    m = X.shape[0]
    a = np.zeros((m, 1))
    b = 0
    passes = 0
    E = np.zeros((m, 1))
    a_old = copy.deepcopy(a)
    # Polynomial Kernel
    kernel = lambda xi, yi: math.pow((np.dot(xi.T, yi) + 1), degree)
    while passes < max_passes:
        num_changed_alphas = 0
        for i in range(m):
            E[i] = f_x(X, y, a, b, X[i, :], degree) - y[i]
            if (y[i] * E[i] < -tol and a[i] < C) or (y[i] * E[i] > tol and a[i] > 0):
                j = rand.randrange(m)
                while j == i:
                    j = rand.randrange(m)
                E[j] = f_x(X, y, a, b, X[j, :], degree) - y[j]
                # print(E[j])
                a_old[i] = a[i]
                a_old[j] = a[j]
                if y[i] != y[j]:
                    L = max(0, a[j] - a[i])
                    H = min(C, C + a[j] - a[i])
                else:
                    L = max(0, a[i] + a[j] - C)
                    H = min(C, a[i] + a[j])
                if L == H:
                    continue
                # calculating the value of eeta
                n = 2 * kernel(X[i, :], X[j, :]) - kernel(X[i, :], X[i, :]) - kernel(X[j, :], X[j, :])
                if n >= 0:
                    continue
                # updating the value of alpha j
                a[j] = a[j] - ((y[j] * (E[i] - E[j])) / n)
                if a[j] > H:
                    a[j] = H
                elif a[j] < L:
                    a[j] = L
                # print(a[j])
                if abs(a[j] - a_old[j]) < tol:
                    continue
                # updating the value of alpha i
                a[i] = a[i] + y[i] * y[j] * (a_old[j] - a[j])
                # print(a[i])
                # now need to calculate the bias term
                b1 = b - E[i] - (y[i] * (a[i] - a_old[i]) * kernel(X[i, :], X[i, :])) - \
                     (y[j] * (a[j] - a_old[j]) * kernel(X[i, :], X[j, :]))
                b2 = b - E[j] - (y[i] * (a[i] - a_old[i]) * kernel(X[i, :], X[j, :])) - \
                     (y[j] * kernel(X[j, :], X[j, :]) * (a[j] - a_old[j]))
                if a[i] > 0 and a[i] < C:
                    b = b1
                elif a[j] > 0 and a[j] < C:
                    b = b2
                else:
                    b = (b1 + b2) / 2
                num_changed_alphas += 1
            # End if
        # End for
        if num_changed_alphas == 0:
            passes += 1
        else:
            passes = 0
    # end while
    return a, b


# divide dataset wrt to each class e.g classes = ['0', '1']
def combination(X_train, y_train, classes):
    X_subclass = np.array([])
    y_subclass = np.array([])
    for j in range(X_train.shape[0]):
        if classes[0] == y_train[j]:
            X_subclass = np.append(X_subclass, np.array(X_train[j]), axis=0)
            y_subclass = np.append(y_subclass, 1)
        elif classes[1] == y_train[j]:
            X_subclass = np.append(X_subclass, np.array(X_train[j]), axis=0)
            y_subclass = np.append(y_subclass, -1)
    return X_subclass, y_subclass


X, Y = extract_features_labels()
ohe = OneHotEncoder(categories='auto')
# X_transformed = ohe.fit_transform(X).toarray()
kf5 = KFold(n_splits=5)
C = [0.02, 0.5, 1, 10]
start = time.time()
for train_index, test_index in kf5.split(X):
    x_train, x_test = X[train_index], X[test_index]
    y_train, y_test = Y[train_index], Y[test_index]
    predicted_values = np.zeros((y_test.shape[0], 2))
    p = 0
    a = {}
    b = {}
    for c in C:
        alpha, bias = SMO(x_train, y_train, c)
        a[p] = alpha
        b[p] = bias
        p += 1
        y_predict = []
        for l in range(x_test.shape[0]):
            if f_x(x_train, y_train, alpha, bias, x_test[1, :], 3) >= 0:
                y_predict.append(1)
            else:
                y_predict.append(-1)
        for j in range(len(y_predict)):
            if y_predict[j] == 1:
                predicted_values[j][0] += 1
            else:
                predicted_values[j][1] += 1
        print(c)
        print("\n")
        print(accuracy(predicted_values, y_test))
        print("\n")
end = time.time()
print(end - start)

"""
Input:
   C: regularization paramter
   tol: numerical tolerance
   max_passes: max num of times to iterate over alpha's without changing
   (x_1, y_1)...(x_m,...y_m): training data
Output:
   alpha \belongs R: Lagrange multipliers for solution
   b \belongs R: threshold for solution
Algorithm:
   num_changed_alphas = 0
   for i=1,..m
        E_i =f(x_i) - y_i
        if((y_i E_i < -tol && alpha_i < C) || (y_i E_i > tol && a_i > 0)){
            Select j!=i randomly
            Calculate E_j = f(x_j) - y_j
            Save old a: a_i(old) = a_i, a_j(old) = a_j
            Compute L and H
            if(L==H)
                continue to next i
            Compute n
            if(n>0)
                continue to next i
            Compute and clip new value for a_j
            if(a_j - a_j(old) < 10^-5)
                continue to next i
            Determine value for a_i
            Compute b1 and b2
            Compute b
            num_changed_alphas = num_changed_alphas+1
        end if
    end for
"""
