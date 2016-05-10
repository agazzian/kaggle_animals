#!/usr/bin/env python3

# Reading and preprocessing the data on shelter animals

import pandas

class AnimalData(object):
    def __init__(self, filename=None):
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
            'Unknown': 2
        }
        self.colors = {
            'Brown': 0,
            'Cream': 1,
            'Blue': 2,
            'Tan': 3,
            'Black': 4,
            'Tabby': 5,
            'Red': 6,
            'White': 7,
            'Silver': 8,
            'Orange': 9,
            'Chocolate': 10,
            'Calico': 11,
            'Torbie': 12,
            'Yellow': 13,
            'Tricolor': 14,
            'Tortie': 15,
            'Gray': 16,
            'Buff': 17,
            'Ruddy': 18,
            'Sable': 19,
            'Agouti': 20,
            'Apricot': 21,
            'Flame': 22,
            'Lynx': 23,
            'Liver': 24,
            'Lilac': 25,
            'Fawn': 26,
            'Gold': 27
        }
        self.ageUnitsInDays = {
            'day': 1,
            'week': 7,
            'month': 30,
            'year': 365
        }
        self.df = None
        if filename: self.readFile(filename)


    def readFile(self, filename):
        """ read in the file using pandas and convert the data in all the fields
        into numbers to make it ready for machine learning """
        self.df = pandas.read_csv(filename, header=0)

        # we don't need the name or outcome subtype for learning
        self.df = self.df.drop(['Name', 'OutcomeSubtype'], axis=1)

        # we will also drop the date, since it is not predictive. it is however a
        # source of data leakage and will lead to better results, although such
        # a model would not be very useful in practice
        self.df = self.df.drop('DateTime', axis=1)

        # There are too many different breeds. Group them by mixed / pure for now.
        self.df['Mixed'] = self.df['Breed'].str.contains(r'Mix').map( { True: 1, False: 0 } )

        # Make new columns that indicate the presence of any given color
        for c in self.colors.keys():
            self.df['has'+c] = self.df['Color'].str.contains(c).map({True: 1, False: 0})
        self.df = self.df.drop('Color', axis=1)

        # also map outcomeTypes, animalTypes, sexes and manipulations
        self.df['OutcomeType'] = self.df['OutcomeType'].map(self.outcomeTypes).astype(int)
        self.df['AnimalType'] = self.df['AnimalType'].map(self.animalTypes).fillna(2).astype(int)
        self.df['AnimalType'] = self.df['AnimalType'] / self.df['AnimalType'].max()
        self.df['Sex'] = self.df['SexuponOutcome'].map(self.sexes).fillna(2).astype(int)
        self.df['Sex'] = self.df['Sex'] / self.df['Sex'].max()
        self.df['Manipulation'] = self.df['SexuponOutcome'].map(self.manipulations).fillna(2).astype(int)
        self.df['Manipulation'] = self.df['Manipulation'] / self.df['Manipulation'].max()

        # convert age to age in days
        self.df['AgeNumber'] = self.df['AgeuponOutcome'].str \
                                   .extract('([0-9]+)', expand=False).astype(float)
        regExp = '(' + '|'.join([s for s in self.ageUnitsInDays.keys()]) + ')'
        self.df['AgeUnit'] = self.df['AgeuponOutcome'].str \
                                  .extract(regExp, expand=False).map(self.ageUnitsInDays)
        self.df['Age'] = (self.df['AgeNumber'] * self.df['AgeUnit'])
        meanAge = self.df['Age'].mean()
        self.df['Age'] = self.df['Age'].fillna(meanAge)
        self.df['Age'] = self.df['Age'] / self.df['Age'].max()

        self.df = self.df.drop(['AgeuponOutcome', 'AgeNumber', 'AgeUnit', 'SexuponOutcome', 'Breed', 'Mixed'],
                               axis=1)

    def makeArrays(self):
        """ return a tuple of arrays: the data and the ids """
        try:
            return (self.df.drop('AnimalID', axis=1).values, self.df['AnimalID'].values)
        except AttributeError:
            print('Warning: the data frame has not been initialized yet, returning empty arrays')
            return ([], [])

    @property
    def df(self):
        return self.__df

    @df.setter
    def df(self, df):
        self.__df = df

if __name__ == '__main__':
    a = AnimalData('data/train.csv')
    print(a.makeArrays()[0])
