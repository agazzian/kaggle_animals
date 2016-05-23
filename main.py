#!/usr/bin/env python3

# main program

from animaldata import AnimalData
from shallownetwork import ShallowNet

# Constants
maxRuns = 2000
validationSize = 5000

if __name__ == '__main__':
    animals, ids = AnimalData('data/train.csv').makeArrays()
    # the first column is the outcome label, which is what we want to predict
    trainData = animals[validationSize:,1:]
    trainTarget = animals[validationSize:,0]
    validationData = animals[0:validationSize,1:]
    validationTarget = animals[0:validationSize,0]

    # need to convert train targets into probabilities: lists with exactly one one,
    # and many zeros
    trainTarget = [[float(i==0), float(i==1), float(i==2), float(i==3), float(i==4)] for i in trainTarget]
    validationTarget = [[float(i==0), float(i==1), float(i==2), float(i==3), float(i==4)] for i in validationTarget]

    print("Number of samples: %d" % len(trainData))

    net = ShallowNet(nInputs = 36, nHidden = 512)
    net.setupNetwork()
    for i in range(maxRuns):
        if i % 100 == 0:
            print('Run %d / %d, Current accuracy: %g' %
                  (i, maxRuns,
                  net.trainOnRandomBatch(trainData, trainTarget, N = 2000, returnAccuracy=True)))
        else:
            net.trainOnRandomBatch(trainData, trainTarget, N = 1000)
    print('Final cross entropy on validation data: %g' % net.currentCrossEntropy(validationData, validationTarget))
    print('Accuracy on validation data: %g' % net.currentAccuracy(validationData, validationTarget))
