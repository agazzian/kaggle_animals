#!/usr/bin/env python3

# main program

from animaldata import AnimalData
from pipeline import Pipe

if __name__ == '__main__':
    animals, ids = AnimalData('data/train.csv').makeArrays()
    # the first column is the outcome label, which is what we want to predict
    trainData = animals[0:,1:]
    trainTarget = animals[0:,0]

    pipe = Pipe(trainData, trainTarget)
    pipe.setpipe(['RF'])
    pipe.crossgrid({ })
    print(pipe.return_score())
