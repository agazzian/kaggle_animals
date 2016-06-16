#!/usr/bin/env python3

# main program to use the neural network on the animal data

import pandas

from importf import filtertrain
from shallownetwork import ShallowNet

# Constants
maxRuns = 2000
validationSize = 5000

if __name__ == '__main__':
    df = pandas.read_csv('data/train.csv')
    X, Y, names = filtertrain(df)
    X = X / X.max()  # need to normalize the data for the neural network
    X = X.values
    Y = Y.values
    print(names)
    trainData = X[validationSize:]
    trainTarget = Y[validationSize:]
    validationData = X[0:validationSize]
    validationTarget = Y[0:validationSize]

    # need to convert train targets into probabilities: lists with exactly one one,
    # and many zeros
    trainTarget = [[float(i==0), float(i==1), float(i==2), float(i==3), float(i==4)] for i in trainTarget]
    validationTarget = [[float(i==0), float(i==1), float(i==2), float(i==3), float(i==4)] for i in validationTarget]

    print("Number of samples: %d" % len(trainData))

    net = ShallowNet(nInputs = len(names), nHidden = 512)
    net.setupNetwork()
    for i in range(maxRuns):
        if i % 100 == 0:
            print('Run %d / %d, Current accuracy: %g' %
                  (i, maxRuns,
                  net.trainOnRandomBatch(trainData, trainTarget, N = 1000, returnAccuracy = True)))
        else:
            net.trainOnRandomBatch(trainData, trainTarget, N = 1000)
    print('Final cross entropy on validation data: %g' % net.currentCrossEntropy(validationData, validationTarget))
    print('Accuracy on validation data: %g' % net.currentAccuracy(validationData, validationTarget))
