# Example of kNN implemented from Scratch in Python

import csv
import random
import math
import operator
import pandas as pd
from sklearn import preprocessing
import numpy as np
import matplotlib.pyplot as plt

def euclideanDistance(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += pow((instance1[x] - instance2[x]), 2)
    return math.sqrt(distance)

def ManhattanDist(instance1, instance2, length):
    distance = 0
    for x in range(length):
        distance += abs(instance1[x] - instance2[x])
    return distance

def findKneighbors(traindf, traindfForRegression, testdfForRegression, k):
    distances = []
    length = len(testdfForRegression) - 1
    for x in range(len(traindf)):
        dist = euclideanDistance(testdfForRegression, traindfForRegression.iloc[x], length)
        distances.append((traindf.iloc[x], dist))
    distances.sort(key=operator.itemgetter(1))
    neighbors = []
    for x in range(k):
        neighbors.append(distances[x][0])
    return neighbors


def predict(neighbors):
    countsForAreaClassificn = {}
    countsForAreaClassificn[0] = 0
    countsForAreaClassificn[1] = 0

    for x in range(len(neighbors)):
        response = neighbors[x]['area']

        if response == 0:
            countsForAreaClassificn[0] += 1
        else:
            countsForAreaClassificn[1] += 1

    if countsForAreaClassificn[0] > countsForAreaClassificn[1]:
        return 0
    else:
        return 1


def getAccuracy(test_output, predictions):
    correct = 0
    for x in range(0, len(test_output) - 1):
        i = 1 if test_output.iloc[x] > 0 else 0
        if i == predictions[x]:
            correct += 1
    return (correct / float(len(test_output))) * 100.0


def normalize_column(A, col):
    A.iloc[:, col] = (A.iloc[:, col] - np.mean(A.iloc[:, col])) /(np.std(A.iloc[:, col]))


def calcAccuracy(traindata, testdata, k):
    traindfForRegression = traindata.iloc[:,:12]
    testdfForRegression = testdata.iloc[:,:12]

    totallen = len(testdata)
    predictions = []

    for x in range(0, totallen - 1):
        neighbors = findKneighbors(traindata, traindfForRegression, testdfForRegression.iloc[x], k)
        result = predict(neighbors)
        predictions.append(result)
        #print('> predicted= ' + repr(result) + ', actual= ' + repr(test_output.iloc[x])) # 'Area']))

    accuracy = getAccuracy(testdata.iloc[:,12], predictions)

    return accuracy
    #print('Accuracy:'+ repr(accuracy) + '%')


def CrossValidation(traindf_orig, traindfForRegression):
    totalrows = len(traindf_orig)

    bestk = 0
    bestaccuracy = 0
    avg_accuracy = 0
    kplot = []
    accuracyplot = []

    for k in range(1,25,2):
        print(k)
        accuracysum = 0
        for i in range(0,5):
            teststartrow = int((i*totalrows)/5)
            testendrow = int(((i+1)*totalrows)/5)
            localtestdata = traindf_orig.iloc[teststartrow:testendrow]
            localtraindata = traindf_orig.drop(traindf_orig.index[teststartrow:testendrow], inplace=False)
            accuracysum += calcAccuracy(localtraindata, localtestdata, k)

        avg_accuracy = accuracysum/5;

        kplot.append(k)
        accuracyplot.append(avg_accuracy)
        print(avg_accuracy)

        if avg_accuracy > bestaccuracy:
            bestaccuracy = avg_accuracy
            bestk = k

    plt.plot(kplot,accuracyplot)
    plt.ylabel('Accuracy')
    plt.xlabel('K')
    plt.show()

    return bestk



def main():

    traindf_orig = pd.read_csv('train.csv')
    traindfForRegression = traindf_orig.iloc[:, :12]

    #traindfForRegression = traindfForRegression **2
    train_output = traindf_orig.iloc[:,12]

    testdf_orig = pd.read_csv('test.csv')
    testdfForRegression = testdf_orig.iloc[:, :12]

    #testdfForRegression ** 2
    test_output = testdf_orig.iloc[:,12]

    #print(traindfForRegression)

    columns = traindfForRegression.shape[1]
    for x in range(0, columns - 1):
        normalize_column(traindfForRegression, x)
        normalize_column(testdfForRegression, x)


    #finding value of k
    k = CrossValidation(traindf_orig, traindfForRegression)
    #print(k)

    predictions = []
    totallen = len(testdfForRegression)

    for x in range(0, totallen - 1):
        neighbors = findKneighbors(traindf_orig, traindfForRegression, testdfForRegression.iloc[x], k)
        result = predict(neighbors)
        predictions.append(result)
        #print('> predicted= ' + repr(result) + ', actual= ' + repr(test_output.iloc[x])) # 'Area']))
    accuracy = getAccuracy(test_output, predictions)
    print('Accuracy:'+ repr(accuracy) + '%')


#from sklearn.neighbors import KNeighborsClassifier

# def main1():
#     traindf_orig = pd.read_csv('train.csv')
#     traindfForRegression = traindf_orig.iloc[:, :12]
#     train_output = traindf_orig.iloc[:, 12]
#
#     testdf_orig = pd.read_csv('test.csv')
#     testdfForRegression = testdf_orig.iloc[:, :12]
#     test_output = testdf_orig.iloc[:, 12]
#
#     train_output[train_output.iloc[:] > 0] = 1
#     train_output = train_output.astype(int)
#     knn = KNeighborsClassifier()
#     knn.fit(np.array(traindfForRegression), train_output)
#     predictions = knn.predict(np.array(testdfForRegression))
#     accuracy = getAccuracy(test_output, predictions)
#     print(knn.p)

main()
#main1()