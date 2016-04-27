#!/usr/bin/env python3

# Reading and preprocessing the data on shelter animals

import pandas

class AnimalData(object):
    def __init__(self, filename):
        self.outcomeTypes = {
            'Return_to_owner': 0,
            'Euthanasia': 1,
            'Died': 2,
            'Adoption': 3,
            'Transfer': 4
        }
        self.animalTypes = {
            'Dog': 0,
            'Cat': 1
        }
        self.sexes = {
            'Intact Female': 0,
            'Intact Male': 1,
            'Spayed Female': 0,
            'Neutered Male': 1,
            'Unknown': 2
        }
        self.manipulations = {
            'Intact Female': 0,
            'Intact Male': 0,
            'Spayed Female': 1,
            'Neutered Male': 1,
            'Unknown': 3
        }
        self.df = None
        self.dfToLearn = None
        self.readFile(filename)


    def readFile(self, filename):
        self.df = pandas.read_csv(filename, header=0)

        # we don't need the AnimalID, name or outcome subtype for learning
        self.dfToLearn = self.df.drop(['AnimalID', 'Name', 'OutcomeSubtype'], axis=1)

        # we will also drop the data, since it is not predictive. it is however a
        # source of data leakage and will lead to better results, although such
        # a model would not be very useful in practice
        self.dfToLearn = self.dfToLearn.drop('DateTime', axis=1)

    @property
    def dfToLearn(self):
        return self.__dfToLearn

    @dfToLearn.setter
    def dfToLearn(self, dfToLearn):
        self.__dfToLearn = dfToLearn

if __name__ == '__main__':
    a = AnimalData('data/train.csv')
    print('Number of different breeds: %d' % len(a.dfToLearn['Breed'].unique()))
    
