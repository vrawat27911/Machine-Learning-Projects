# whether add 0 valued columns

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import metrics
import numpy as np
import time
import cvxopt.solvers
from cvxopt import matrix


MIN_SUPPORT_VECTOR_MULTIPLIER = 1e-5


class SVM(object):
    def __init__(self, c):
        #self._kernel = kernel
        self._c = c

    def trainSVM(self, X, y):
        lagrange_multipliers = self.compMultipliers(X, y)
        return self.predictor(X, y, lagrange_multipliers)

    def calculateGramMatrix(self, X):
        n_samples, n_features = X.shape
        K = np.zeros((n_samples, n_samples))

        for i, x_i in enumerate(X):
            for j, x_j in enumerate(X):
                K[i,j] = np.inner(x_i, x_j)
        return K

    def predictor(self, X, y, lagrange_multipliers):
        support_vector_indices = lagrange_multipliers > MIN_SUPPORT_VECTOR_MULTIPLIER

        support_multipliers = lagrange_multipliers[support_vector_indices]
        support_vectors = X[support_vector_indices]

        Y_transpose = pd.DataFrame(y).transpose()
        support_vector_labels = Y_transpose[support_vector_indices]

        bias = np.mean(
            [y_k - SVMPredictor(
                bias=0.0,
                weights=support_multipliers,
                support_vectors=support_vectors,
                support_vector_labels=support_vector_labels).predict(x_k)
             for (y_k, x_k) in zip(support_vector_labels, support_vectors)])

        return SVMPredictor(
            bias=bias,
            weights=support_multipliers,
            support_vectors=support_vectors,
            support_vector_labels=support_vector_labels)

    def compMultipliers(self, X, y):
        n_samples, n_features = X.shape

        K = self.calculateGramMatrix(X)

        # Solves
        # min 1/2 x^T P x + q^T x
        # s.t.
        #  Gx \leq h
        #  Ax = b

        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-1 * np.ones(n_samples))

        G_std = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
        h_std = cvxopt.matrix(np.zeros(n_samples))

        G_slack = cvxopt.matrix(np.diag(np.ones(n_samples)))
        h_slack = cvxopt.matrix(np.ones(n_samples) * self._c)

        G = cvxopt.matrix(np.vstack((G_std, G_slack)))
        h = cvxopt.matrix(np.vstack((h_std, h_slack)))

        A = cvxopt.matrix(y, (1, n_samples))
        A = matrix(np.double(A))

        b = cvxopt.matrix(0.0)

        solution = cvxopt.solvers.qp(P, q, G, h, A, b)

        return np.ravel(solution['x'])


class SVMPredictor(object):
    def __init__(self,
                 bias,
                 weights,
                 support_vectors,
                 support_vector_labels):
        self._bias = bias
        self._weights = weights
        self._support_vectors = support_vectors
        self._support_vector_labels = support_vector_labels

    def predict(self, x):
        result = self._bias
        labels = pd.DataFrame(self._support_vector_labels).transpose();
        weights = pd.DataFrame(self._weights).transpose();

        for z_i, x_i, y_i in zip(weights, self._support_vectors, labels):
            result += (np.asarray(z_i, dtype = float) * np.asarray(y_i, dtype = float) * np.inner(np.asarray(x_i, dtype = float), np.asarray(x, dtype = float))) #self._kernel(x_i, x)

        return np.sign(result)


def main():
    colNames = list(range(0,31))

    df = pd.read_csv('hw2_question3.csv', names=colNames)
    #print (df)

    y = df.iloc[:,30]
    #print(y)

    X = df.iloc[:,:30]
    #print(X)

    columnsToTransform = [2, 7, 8, 14, 15, 16, 26, 29]
    columnsToTransform = [x - 1 for x in columnsToTransform]

    # preprocessed dataframe
    X_procsd = pd.get_dummies(X, columns=columnsToTransform)
    #print(X_procsd.columns)

    X_train, X_test, Y_train, Y_test = train_test_split(X_procsd, y, test_size=0.33)
    #print(len(X_train.index))

    #misclassification cost C is to be varied
    c = 0.0001
    accuracy = []

    print ('linear')
    #cv sets nfold cross val
    while c < 1000:
        starttime = time.time()
        clf = svm.SVC(kernel='linear', C=c)
        scores = cross_val_score(clf, X_train, Y_train, cv=3)
        cross_acc = np.mean(scores)
        predicted = cross_val_predict(clf, X_test, Y_test, cv=3)
        accuracy.append(metrics.accuracy_score(Y_test, predicted))
        #print(accuracy)
        print('C : ', c, ' avg score: ', cross_acc, ' time taken: ', (time.time() - starttime))
        c *= 10

    print(accuracy)

    c = 0.001
    gamma = 0.01
    degree = 1
    maxiter = 1
    randstate = 10
    accuracy1 = []
    print('poly')

    #kernel type (poly or rbf) n parameters will you choose
    while c < 1000:
        starttime = time.time()
        clf1 = svm.SVC(kernel='poly', C=c)
        scores1 = cross_val_score(clf1, X_train, Y_train, cv=3)
        cross_acc = np.mean(scores1)
        predicted1 = cross_val_predict(clf1, X_test, Y_test, cv=3)
        accuracy1.append(metrics.accuracy_score(Y_test, predicted1))
        #print(accuracy1)
        print('C : ', c, ' cross_acc: ', cross_acc, ' time taken: ', (time.time() - starttime))
        c *= 10

    while gamma < 1000:
        starttime = time.time()
        clf1 = svm.SVC(kernel='poly', gamma=gamma)
        scores1 = cross_val_score(clf1, X_train, Y_train, cv=3)
        cross_acc = np.mean(scores1)
        predicted1 = cross_val_predict(clf1, X_test, Y_test, cv=3)
        accuracy1.append(metrics.accuracy_score(Y_test, predicted1))
        #print(accuracy1)
        print('Gamma : ', gamma, ' cross_acc: ', cross_acc, ' time taken: ', (time.time() - starttime))
        gamma *= 10

    while degree < 7:
        starttime = time.time()
        clf1 = svm.SVC(kernel='poly', degree=degree)
        scores1 = cross_val_score(clf1, X_train, Y_train, cv=3)
        cross_acc = np.mean(scores1)
        predicted1 = cross_val_predict(clf1, X_test, Y_test, cv=3)
        accuracy1.append(metrics.accuracy_score(Y_test, predicted1))
        print('Degree : ', degree, ' cross_acc: ', cross_acc, ' time taken: ', (time.time() - starttime))
        degree += 1

    while randstate < 50:
        starttime = time.time()
        clf1 = svm.SVC(kernel='poly', random_state=randstate)
        scores1 = cross_val_score(clf1, X_train, Y_train, cv=3)
        cross_acc = np.mean(scores1)
        predicted1 = cross_val_predict(clf1, X_test, Y_test, cv=3)
        accuracy1.append(metrics.accuracy_score(Y_test, predicted1))
        #print(accuracy1)
        print('Randstate : ', randstate, ' cross_acc: ', cross_acc, ' time taken: ', (time.time() - starttime))
        randstate += 10

    #print(accuracy1)
    c = 0.001
    accuracy2 = []
    print('rbf')

    while c < 1000:
        starttime = time.time()
        clf2 = svm.SVC(kernel='rbf', C=c)
        scores2 = cross_val_score(clf2, X_train, Y_train, cv=3)
        cross_acc = np.mean(scores2)
        predicted2 = cross_val_predict(clf2, X_test, Y_test, cv=3)
        accuracy2.append(metrics.accuracy_score(Y_test, predicted2))
        #print(accuracy2)
        print('C : ', c, ' avg score: ', cross_acc, ' time taken: ', (time.time() - starttime))
        c *= 10

    while gamma < 1000:
        starttime = time.time()
        clf1 = svm.SVC(kernel='poly', gamma=gamma)
        scores1 = cross_val_score(clf1, X_train, Y_train, cv=3)
        cross_acc = np.mean(scores1)
        predicted1 = cross_val_predict(clf1, X_test, Y_test, cv=3)
        accuracy1.append(metrics.accuracy_score(Y_test, predicted1))
        #print(accuracy1)
        print('Gamma : ', gamma, ' cross_acc: ', cross_acc, ' time taken: ', (time.time() - starttime))
        gamma *= 10

    while randstate < 50:
        starttime = time.time()
        clf1 = svm.SVC(kernel='poly', random_state=randstate)
        scores1 = cross_val_score(clf1, X_train, Y_train, cv=3)
        cross_acc = np.mean(scores1)
        predicted1 = cross_val_predict(clf1, X_test, Y_test, cv=3)
        accuracy1.append(metrics.accuracy_score(Y_test, predicted1))
        #print(accuracy1)
        print('Randstate : ', randstate, ' cross_acc: ', cross_acc, ' time taken: ', (time.time() - starttime))
        randstate += 10

    while maxiter < 7:
        starttime = time.time()
        clf1 = svm.SVC(kernel='poly', degree=degree)
        scores1 = cross_val_score(clf1, X_train, Y_train, cv=3)
        cross_acc = np.mean(scores1)
        predicted1 = cross_val_predict(clf1, X_test, Y_test, cv=3)
        accuracy1.append(metrics.accuracy_score(Y_test, predicted1))
        print('Maxiter : ', maxiter, ' cross_acc: ', cross_acc, ' time taken: ', (time.time() - starttime))
        maxiter += 1

    mysvm = SVM(1)
    predictor = mysvm.trainSVM(np.matrix(X_train[:]), np.matrix(Y_train[:]))
    prediction = predictor.predict(np.matrix(X_test[:]))
    myaccuracy = metrics.accuracy_score(np.array(Y_test[:], np.float),prediction[0])
    print(myaccuracy)

main()