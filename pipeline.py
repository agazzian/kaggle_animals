#! /usr/bin/env python

"""
Copyright of the program:   Andrea Agazzi, UNIGE
                            David Dasenbrook, UNIGE


"""
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import logging
import cv_sampling as cv
from sklearn import *
import sklearn as sk
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from datetime import datetime
import time
from importf import *
#import xgboost as xgb

class Pipe(object):
    """
    pipeline class
    """
    cvcounter = 0


    def __init__(self, Xdata, Ydata, feat_names, pipe=None, crossval=None):
        self.X = Xdata
        self.Y = Ydata
        self.feat_names = feat_names
        #self.weeks = weeks
        self._pipe = pipe
        self._pipelst = []
        self._bestscore = None
        self._gridsearch = None
        self._biolst = [None for n in feat_names]
        self.crossval = crossval

    def newpipe(self):
        """ if pipe is empty return true else reset all """
        pass

    def setpipe(self, nlst):
        """ set a new pipe attribute """
        steplst = []
        # check if there are no doubles in the pipeline or more in general if the pipeline is valid
        for n in nlst:
            steplst.append(Pipe.addpipeel(n))
            print(n+' added to the pipeline')

        self._pipe = Pipeline(steps=steplst)
        self._pipelst = nlst[:]

    @staticmethod
    def addpipeel(estimatorname):
        """
        Associates the names in the list [PCA, thresh, logreg, FDA, RF] a sklearn estimator.

        Returns a tuple (sk.estimator, estimatorname) where
            sk.estimator is the sklearn estimator object corresponding to the input string
            estimatorname is the input string
        In case the input string is not in the list of accepted inputs the method
        returns an error message and None
        """

        estimatorSwitcher = {
            'PCA': sk.decomposition.PCA(copy=True),
            'FFS': sk.feature_selection.SelectKBest(score_func=corr_analysis),
            'L1LogReg': sk.linear_model.LogisticRegression(penalty='l1'),
            'L2LogReg': sk.linear_model.LogisticRegression(),
            'FDA': sk.discriminant_analysis.LinearDiscriminantAnalysis(n_components = 1,solver='svd'),
            'RF': sk.ensemble.RandomForestClassifier(),
            'GB': sk.ensemble.GradientBoostingClassifier()
        }

        try:
            result = (estimatorname, estimatorSwitcher[estimatorname])
        except KeyError:
            print('Error: estimator '+estimatorname+' not in the list!\n')
            result = None

        return result

    def crossgrid(self, griddic, scoring=None, crossval=None):
        """
        perform a crossvalidation procedure for mparameters in grid and cv-sample cv

        inputs
            grid: dictionary of the form
                dict(ESTIMATORNAME1__PARAMETER1 = PARAMLST1,
                     ESTIMATORNAME2__PARAMETER2 = PARAMLST2,
                     ...
                     )
            cv: crossvalidation array of the form [IDn1, IDn2,...IDnN]
        """
        # initialize cv procedure
        if crossval != None:
            # if cv not empty overwrite existing cv procedure
            self.crossval = crossval
        elif crossval == None and self.crossval == None:
            # if no cv procedure has been specified set the classical l20o
            print('ATTENTION:\tNo CV procedure specified, proceeding with reduced l20o, all weeks included.')
            crossval = cv.leave_x_out(self.Y, 20)

        # initialize the _gridsearch attribute
        # need to include how to create the dictionary from the input
        self._gridsearch = sk.grid_search.GridSearchCV(self._pipe, griddic, n_jobs=-1, scoring = scoring, cv = self.crossval)

        # fit the CV grid
        self._gridsearch.fit(self.X, self.Y)

    def return_score(self):
        """ returns the best score of the fitted model """
        return self._gridsearch.best_score_

    def return_grid_scores(self):
        """ return the grid scores of the fitted model """
        return self._gridsearch.grid_scores_

    def return_prediction(self, X):
        """ return a prediction Y for the data X using the fitted model """
        return self._gridsearch.predict_proba(X)

    @staticmethod
    def coeffs(estimator):
        """
        Extracts the coefficients associated to each metabolite

        Works only for supervised algorithms
        """
        coeflst = []
        if hasattr(estimator,'feature_importances_'):
            # RandomForestClassifier
            coeflst = estimator.feature_importances_
        elif hasattr(estimator,'coef_'):
            # FDA or LogReg
            coeflst = [abs(c) for c in estimator.coef_]
        elif hasattr(estimator,'scores_') and hasattr(estimator,'get_support'):
            # Greedy correlation filter
            supp = estimator.get_support()
            coeflst = [abs(c) if supp[i] else 0 for i,c in enumerate(estimator.scores_)]
        return coeflst

    def return_rank(self,*args):
        """
        Returns the biomarker ranking of the best performing algorithm if no other estimator is given as input

        The second argument of the function (optional) can be used to declare another
        sk.pipeline.Pipeline object from which to extract the biomarker rankings.
        """

        if len(args) == 0:
            estimator = self._gridsearch.best_estimator_
        else:
            estimator = args[0]

        counter = 0
        for stepname in self._pipelst[::-1]:
            # tle last step is the classificator
            pipestep = estimator.named_steps[stepname]
            if counter == 0:
                clst = Pipe.coeffs(pipestep)
                if stepname == 'FDA':
                    clst = list(clst[0])
            else:
                # todo: add logistic regression as a preprocessing step
                if hasattr(pipestep,'inverse_transform'):
                    clst = pipestep.inverse_transform(np.array(clst).reshape(1,-1))[0]
                else:
                    print('ERROR:\tclassifier used as preprocessing step')
            counter += 1
        # todo: write best performer onto self._biolst
        return clst

    def return_ranks(self,tol,printtofile=False):
        """ returns the averaged biomarker rankings for all fits that perform within (tol*bestscore, bestscore)"""

        ranks = []

        for score_triple in self._gridsearch.grid_scores_[:]:
            # if the score is above the tolerance refit the model and return the ranking
            if score_triple[1] >= tol*self._bestscore:
                steplst = []
                # check if there are no doubles in the pipeline or more in general if the pipeline is valid
                for n in self._pipelst[:]:
                    steplst.append(Pipe.addpipeel(n))

                pipetofit = Pipeline(steps=steplst)

                print(score_triple[0])
                estimator = pipetofit.set_params(**score_triple[0]).fit(self.X,self.Y)
                ranks.append((score_triple[1],score_triple[0],self.return_rank(estimator)))

        if printtofile == True:
            self.save_ranks(ranks)

        return ranks

    def save_ranks(self,ranks):
        """
        save the ranks list in a file

        The saved rank list is sorted according to the score
        """
        now = datetime.now()
        with open('./results/ranks/'+'_'.join(self._pipelst)+'_'+now.strftime('%Y_%m_%d_%H_%M')+'_'+str(Pipe.cvcounter)+'.dat','w') as f:
            f.write('# pipeline:\t'+'+'.join(self._pipelst)+'\n cv:\tleave-'+str(len(list(self.crossval[0][1])))+'-out \t samples: \t'+str(len(list(self.crossval)))+'\n\n')
            for l in sorted(ranks,key= lambda x: x[0],reverse=True):
                f.write('score:\t'+str(l[0])+'\nparameters:\t'+str(l[1])+'\n'+'\n'.join([a+'\t'+b+'\t'+c for (a,b,c) in sorted(zip(map(str,l[2]),map(str,range(len(l[2]))),self.feat_names),key = lambda x: abs(float(x[0])),reverse=True)])+'\n\n------------------------------------------------\n')

def corr_analysis(Xdata,Ydata):
    """
	Pearson correlation analysis fo the features in xdata and ydata
	"""
    # To do: simplify this, make it more readable
    return np.array([abs(np.corrcoef(np.array(Xdata)[:,j],Ydata)[0][1]) for j in range(len(Xdata[0]))]), np.array([1 for j in range(len(Xdata[0]))])


if __name__ == '__main__':

    df = pd.read_csv('data/train.csv')
    print(df)

    X, Y, names = filtertrain(df)
    X = np.array(X)
    Y = np.array(Y)

    print(corr_analysis(X, Y))

    print(X)

    # run an initialization test for a pipeline with ffs and fda
    pipe = Pipe(X,Y,names)

    pipe.setpipe(['GB'])

    # cvcounter test
    print('Pipe.cvcounter =\t'+str(pipe.cvcounter))

    print("X size: " + str(np.shape(X)))
    print("Y size: " + str(np.shape(Y)))
    # test initialization of grid parameters

    griddic = {
        'GB__n_estimators': [100, 200, 300],
        'GB__learning_rate': [0.1, 0.3, 0.5],
        'GB__max_features': ['auto', 100],
        'GB__max_depth': [3, 5, 10]
    }
    pipe.crossgrid(griddic,crossval=cv.leave_x_out(pipe.Y, 50, nsamples=100))
    print(pipe.return_score())
    print(pipe.return_grid_scores())
    print(pipe.return_rank())
    print(pipe.return_ranks(.5, printtofile=True))

    # prediction

    df2 = pd.read_csv('data/test.csv')

    X2, IDS, names2 = filtertrain(df2,'test')

    Y2 = pipe.return_prediction(X2)

    print(Y2)

    output = pd.DataFrame({'ID': np.array(IDS)})

    output['Adoption'] = Y2[:,0]
    output['Died'] = Y2[:,1]
    output['Euthanasia'] = Y2[:,2]
    output['Return_to_owner'] = Y2[:,3]
    output['Transfer'] = Y2[:,4]

    output.to_csv('submission.csv', index=False)
