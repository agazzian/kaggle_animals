#!/usr/bin/env python3

# main program

from animaldata import AnimalData
from shallownetwork import ShallowNet

if __name__ == '__main__':
    animals, ids = AnimalData('data/train.csv').makeArrays()
    # the first column is the outcome label, which is what we want to predict
    trainData = animals[0:,1:]
    trainTarget = animals[0:,0]

    # need to convert train targets into probabilities: lists with exactly one one,
    # and many zeros
    trainTarget = [[float(i==0), float(i==1), float(i==2), float(i==3), float(i==4)] for i in trainTarget]

    net = ShallowNet()
    net.setupNetwork()
    maxRuns = 10000
    for i in range(maxRuns):
        if i % 100 == 0:
            print('Run %d / % d, Current accuracy: %g' %
                  (net.trainOnRandomBatch(trainData, trainTarget, returnAccuracy=True),
                  i, maxRuns))
        else:
            net.trainOnRandomBatch(trainData, trainTarget)
