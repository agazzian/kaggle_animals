#!/usr/bin/env python3

"""
Some functions for decoding the data at hand:

"""

import pandas as pd
import numpy as np

def age_to_days(item):
    """
    Converts the 'age' string into days (int, output)
    """
    # convert item to list if it is one string
    if type(item) is str:
        item = [item]
    ages_in_days = np.zeros(len(item))
    for i,x in enumerate(item):
        ages_in_days[i] = get_age(x)
    return ages_in_days

def get_age(x):
    # check if item[i] is str
    if type(x) is str:
        if 'day' in x:
            ages_in_days = int(x.split(' ')[0])
        if 'week' in x:
            ages_in_days = int(x.split(' ')[0])*7
        if 'month' in x:
            ages_in_days = int(x.split(' ')[0])*30
        if 'year' in x:
            ages_in_days = int(x.split(' ')[0])*365
    else:
        # item[i] is not a string but a nan
        ages_in_days = 0
    return ages_in_days

    # functions to get new parameters from the column
def is_male(x):
    """
    Returns the sex of the animal.
    """
    x = str(x)
    if x.find('Male') >= 0: return True
    if x.find('Female') >= 0: return False
    return None

def is_neutered(x):
    """
    Returns true if the animal was neutered, False if not and None if no information is given.
    """
    x = str(x)
    if x.find('Spayed') >= 0: return True
    if x.find('Neutered') >= 0: return True
    if x.find('Intact') >= 0: return False
    return None

def get_age_category(x):
    """
    Separates the age into categories
    """
    if x < 365: return 'puppy'
    if x < 3*365: return 'young'
    if x < 5*365: return 'young adult'
    if x < 10*365: return 'adult'
    return 'old'

    # Defining has name method
def has_name(name):
    """
    Returns True if the input value is not NaN, False otherwise
    """
    if name is np.nan:
        return False
    return True

def time_to_min(x):
    """
    Converts DateTime feature into min
    """
    return int(x[11:13])*60+int(x[14:16])

def breeds_to_n(df):
    """
    returns a dictionary containing the names of the breeds and the associated number
    """
    # Load data
    feature = 'Breed'

    feature_values_dog = df.loc[df['AnimalType'] == 'Dog',feature]

    feature_values_cat = df.loc[df['AnimalType'] == 'Cat',feature]

    # collect unique breeds:
    # split up mixed breeds and merge the sublists
    feature_values = [i.split('/') for i in feature_values_dog]
    feature_values = [j for i in feature_values for j in i]
    # remove 'Mix' from the strings, but add it as a unique element
    feature_values = [i == i[:-4] if i[-3:] == 'Mix' else i for i in feature_values]
    feature_values = feature_values + ['Mix']
    unique_breeds_dog = np.unique(feature_values)

    # same for cats
    feature_values = [i.split('/') for i in feature_values_cat]
    feature_values = [j for i in feature_values for j in i]
    # remove 'Mix' from the strings, but add it as a unique element
    feature_values = [i == i[:-4] if i[-3:] == 'Mix' else i for i in feature_values]
    feature_values = feature_values + ['Mix']
    unique_breeds_cat = np.unique(feature_values)

    # unique outcomes:
    unique_outcomes = np.unique(np.append(unique_breeds_dog,unique_breeds_cat))

    return dict(unique_outcomes,range(unique_outcomes))

def color_to_n(df):
    """
    returns a dictionary containing the names of the breeds and the associated number
    """
    # Load data
    feature = 'Color'

    feature_values_dog = df.loc[df['AnimalType'] == 'Dog',feature]

    feature_values_cat = df.loc[df['AnimalType'] == 'Cat',feature]

    # collect unique breeds:
    # split up mixed breeds and merge the sublists
    feature_values = [i.split('/') for i in feature_values_dog]
    feature_values = [j for i in feature_values for j in i]
    # remove 'Mix' from the strings, but add it as a unique element
    feature_values = [i == i[:-4] if i[-3:] == 'Mix' else i for i in feature_values]
    feature_values = feature_values + ['Mix']
    unique_color_dog = np.unique(feature_values)

    # same for cats
    feature_values = [i.split('/') for i in feature_values_cat]
    feature_values = [j for i in feature_values for j in i]
    # remove 'Mix' from the strings, but add it as a unique element
    feature_values = [i == i[:-4] if i[-3:] == 'Mix' else i for i in feature_values]
    feature_values = feature_values + ['Mix']
    unique_color_cat = np.unique(feature_values)

    # unique outcomes:
    unique_outcomes = np.unique(np.append(unique_color_dog,unique_color_cat))

    print(unique_outcomes)

    return dict(unique_outcomes,range(unique_outcomes))

if __name__ == '__main__':
    animals = pd.read_csv('data/train.csv')

    # modify the age of the puppies in days
    animals.AgeuponOutcome = age_to_days(animals.AgeuponOutcome)

    # Creating parameter HasName.
    animals['HasName'] = animals.Name.apply(has_name)

    # creating dics
    #colordic = color_to_n(animals)
    #breedsdic = breeds_to_n(animals)

    animals['AgeCat'] = animals.AgeuponOutcome.apply(get_age_category)

    animals['IsMale'] = animals.SexuponOutcome.apply(is_neutered)

    animals['IsNeutered'] = animals.SexuponOutcome.apply(is_neutered)

    animals['TimeInM'] = animals.DateTime.apply(time_to_min)


    print(animals)
