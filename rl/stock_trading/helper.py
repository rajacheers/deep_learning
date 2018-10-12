import numpy as np
import math

# returns formatted price
"""Has a few borrowed functions from Siraj's stock prediction with DQNs"""


def formatPrice(n):
    if n >= 0:
        curr = "$"
    else:
        curr = "-$"
    return (curr + "{0:.2f}".format(abs(n)))


# Return a vector of stock data from csv file
def getStockData(key):
    datavec = []
    lines = open("data/" + key + ".csv", "r").read().splitlines()

    for line in lines[1:]:
        datavec.append(float(line.split(",")[4]))

    return datavec


# Generate states from input vector
def getState(data, t, window):
    if t - window >= -1:
        vec = data[t - window + 1:t + 1]
    else:
        vec = -(t - window + 1) * [data[0]] + data[0: t + 1]
    scaled_state = []
    for i in range(window - 1):
        scaled_state.append(1 / (1 + math.exp(vec[i] - vec[i + 1])))  # scale state vector to [0 1] with sigmoids

    return np.array([scaled_state])