# import matplotlib.pyplot as plt
# import pandas as pd
# import math
# import numpy as np
#
# def main():
#
#     traindf_orig = pd.read_csv('train.csv')
#     testdf_orig = pd.read_csv('test.csv')
#
#     train_output = traindf_orig[traindf_orig['area'] > 0].iloc[:, 12]
#     test_output = testdf_orig[testdf_orig['area'] > 0].iloc[:, 12]
#
#
#     logarea = np.log(np.array(train_output[:]))
#
#     #print(traindfForRegression)
#     plt.hist(np.array(train_output))
#     #plt.interactive(False)
#     plt.hist(logarea)
#     #plt.title("Histogram for area")
#     plt.title("Histogram for area")
#     plt.show(block=True)
#
#
# main()