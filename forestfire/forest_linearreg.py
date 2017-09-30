# Simple Linear Regression on the Swedish Insurance Dataset
from random import seed
from random import randrange
from csv import reader
from math import sqrt
from numpy.linalg import inv
import pandas as pd
import numpy as np


# Convert string column to float
def str_column_to_float(dataset, column):
    for row in dataset:
        row[column] = float(row[column].strip())


# Split a dataset into a train and test set
def train_test_split(dataset, split):
    train = list()
    train_size = split * len(dataset)
    dataset_copy = list(dataset)
    while len(train) < train_size:
        index = randrange(len(dataset_copy))
        train.append(dataset_copy.pop(index))
    return train, dataset_copy

# Evaluate an algorithm using a train/test split
def evaluate_algorithm(dataset, algorithm, split, *args):
    train, test = train_test_split(dataset, split)
    test_set = list()
    for row in test:
        row_copy = list(row)
        row_copy[-1] = None
        test_set.append(row_copy)
    predicted = algorithm(train, test_set, *args)
    actual = [row[-1] for row in test]
    rmse = rmse_metric(actual, predicted)
    return rmse


# Calculate the mean value of a list of numbers
def mean(values):
    return sum(values) / float(len(values))


# Calculate covariance between x and y
def covariance(x, mean_x, y, mean_y):
    covar = 0.0
    for i in range(len(x)):
        covar += (x[i] - mean_x) * (y[i] - mean_y)
    return covar


# Calculate the variance of a list of numbers
def variance(values, mean):
    return sum([(x - mean) ** 2 for x in values])

def coefficients(traindfForRegression, train_output):
    #traindf_transpose = traindfForRegression.transpose()
    train_array = np.array(traindfForRegression)
    transpose_val = train_array.transpose()
    product_1 = np.dot(transpose_val, train_array)
    inverse = inv(product_1)
    product_2 = np.dot(inverse, transpose_val)
    product_3 = np.dot(product_2,  np.array(train_output))

    #print(product_3)
    return product_3

def computeRSS(testdfForRegression, optimal_weights, test_output):
    predicted_output = np.dot(np.array(testdfForRegression), np.array(optimal_weights))
    diff_output_prediction = test_output - predicted_output
    return np.dot(diff_output_prediction.transpose(), diff_output_prediction)

def normalize_column(A, col):
    A.iloc[:, col] = (A.iloc[:, col] - np.mean(A.iloc[:, col])) /(np.std(A.iloc[:, col]))

def corr2_coeff(A,B):
    # Rowwise mean of input arrays & subtract from input arrays themeselves
    A_mA = A - A.mean()
    B_mB = B - B.mean()

    # Sum of squares across rows
    ssA = (A_mA**2).sum()
    ssB = (B_mB**2).sum()

    # Finally get corr coeff
    return np.dot(A_mA,B_mB.T)/np.sqrt(np.dot(ssA,ssB))

def main():

    traindf_orig = pd.read_csv('train.csv')
    traindfForRegression = traindf_orig[traindf_orig['area'] > 0].iloc[:,:12]

    traindfForRegression = traindfForRegression**2
    train_output = traindf_orig[traindf_orig['area'] > 0].iloc[:,12]

    testdf_orig = pd.read_csv('test.csv')
    testdfForRegression = testdf_orig[testdf_orig['area'] > 0].iloc[:,:12]

    testdfForRegression = testdfForRegression**2
    test_output = testdf_orig[testdf_orig['area'] > 0].iloc[:,12]

    #print(traindfForRegression)
    columns = traindfForRegression.shape[1]

    for x in range(0, columns - 1):
         normalize_column(traindfForRegression, x)
         normalize_column(testdfForRegression, x)

    #print(traindfForRegression)

    #initial weigths/coefficients for linear regression
    optimal_weights = coefficients(traindfForRegression, train_output)
    #print(optimal_weights)

    # Compute RSS for test data
    RSSvalue = computeRSS(testdfForRegression, optimal_weights, test_output)

    print(RSSvalue)
    predicted_output = np.dot(np.array(testdfForRegression), np.array(optimal_weights))

    print(corr2_coeff(test_output, predicted_output))
    #print(np.corrcoef(test_output,predicted_output))

    # evaluate algorithm
    #split = 0.6
    #rmse = evaluate_algorithm(dataset, simple_linear_regression, split)
    #print('RMSE: %.3f' % (rmse))


#from scipy.stats import linregress
# def main1():
#
#     from sklearn import linear_model
#     clf = linear_model.LinearRegression()
#     traindf_orig = pd.read_csv('train.csv')
#     traindfForRegression = traindf_orig[traindf_orig['area'] > 0].iloc[:, :12]
#     train_output = traindf_orig[traindf_orig['area'] > 0].iloc[:, 12]
#     clf.fit(np.array(traindfForRegression), np.array(train_output))
#
#     testdf_orig = pd.read_csv('test.csv')
#     testdfForRegression = testdf_orig[testdf_orig['area'] > 0].iloc[:, :12]
#     test_output = testdf_orig[testdf_orig['area'] > 0].iloc[:, 12]
#     print(((clf.predict(testdfForRegression)- test_output)**2).sum())
#
#
# main1()

main()