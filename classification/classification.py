# -*- coding: utf-8 -*-


import os
import time
import numpy as np
from scipy.stats import randint as sp_randint
from scipy.interpolate import interp1d
import logging
import progressbar
import tqdm
from tqdm.contrib.concurrent import process_map
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import astropy.io.fits as fits
from astropy.coordinates import SkyCoord
from astropy.table import Table
from astropy import units as u
from astropy.table import join, vstack
from astropy.io.ascii import write, read
import itertools
import warnings
import math as m
from sklearn.cross_validation import cross_val_score, train_test_split, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, confusion_matrix, make_scorer
from operator import itemgetter
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV

from des_sn_hosts.classification.classification_utils import (is_correct_host, prep_classification,
                                            get_features, find_nearest, score_func, score_func_CV)

warnings.simplefilter('ignore')

class Classifier():
    def __init__(self,df,config_path):
        self._df=df
        self.config = self._config(config_path)
        self._train,self._test = self._split_data()
        self.X_train,self.y_train,self.X_test,self.y_test = self._prep_features(self,features)
        self.scorer_PFE = make_scorer(score_func_CV,needs_proba=True)

    def _config(self,config_path):
        config = yaml.load(open(config_path))
        setattr(self,'features',config['features'])
        for k,v in config['sizes']:
            setattr(self,k,v)
        self._params = config['params']
        for k, v in config['params'].items():
        if type(v)==dict:
            if v['distr']=='uniform':
                if v['args'][1]=='n':
                    v['args'][1]= len(config['features'])
                self._params[k] = stats.randint(v['args'][0],v['args'][1])
        return config
    def _get_labels(self):
        ''' Return correct and incorrect fake host matches '''
        self._df['Correct'] = self._df['GALID_diff'].apply(is_correct_host)

    def _split_data(self):
        '''Split the data into training and test sets'''
        if self.split =='even':
            n_wrong = n_right = self.n_train/2.
            wrong_train = self._df[self._df['Correct']==0].sample(n_wrong)
            correct_train = self._df[self._df['Correct']==1].sample(n_right)
            train = pd.concat([wrong_train,correct_train])

        elif self.split =='random':
            train = matched_closest.sample(self.n_train)

        test = self._df.loc[~self._df.index.isin(train.index)].sample(self.n_test)
        return train, test

    def _prep_features(self,features):
        X_train = get_features(features, self._train)
        y_train = self._train['Correct'] # class 1=correct match, 0=wrong match
        X_test = get_features(features, self._test)
        y_test = self._test['Correct']
        return X_train, y_train, X_test, y_test
    @property
    def df(self):
        return self._df

    @property
    def train(self):
        return self._train
    @property
    def test(self):
        return self._test

    def CV(self,n_iter=None,cv=None,verbose=None,n_jobs=None,seed=None,dump=False,sf=None):
        if not n_iter:
            n_iter = self.config['crossVal']['n_iter']
        if not cv:
            cv = self.config['crossVal']['cv']
        if not verbose:
            verbose = self.config['crossVal']['verbose']
        if not n_jobs:
            n_jobs = self.config['crossVal']['n_jobs']
        if not seed:
            seed = self.config['crossVal']['seed']
        self.clf = RandomizedSearchCV(RandomForestClassifier(),self._params,n_iter = n_iter, scoring = self.scorer_PFE,
                random_state = seed, cv = cv,n_jobs=n_jobs)
        self.clf.fit(self.X_train, self.y_train)
        if dump:
            pickle.dump(self.clf,open(sf,'wb'))
        return self.clf
    def load_clf(self,sf):
        self.clf =pickle.load(open(sf,'rb'))

    def fit_test(self,effT=None):
        probs = self.clf.predict_proba(self.X_test)[:, 1] # good matches are class 1
        pred = self.clf.predict(self.X_test)
        correct = (self.y_test==1)
        wrong = (self.y_test==0)
        self.test['Prob'] = probs
        pur, eff, thresh = precision_recall_curve(self.y_test, probs, pos_label=1)
        x = eff[::-1][30001:-1]
        y = thresh[::-1][30000:-1]
        efficiency_func = interp1d(x, y, kind='linear') # reverse-order so x is monotonically increasing
        if not effT:
            effT = self.config['effT']
        P_effT = efficiency_func(effT) # threshold probability at efficiency=98%
        print ('\nProb (eff=98%) =', P_effT)
        print ('Purity (P_thresh=0) = ', pur[0])
        score = score_func(probs, self.y_test)
        print ('SCORE (pur @ eff=98%) = ', score)
        correct_match = self.test['Prob']>P_effT
        print ('number of correct matches with P_thresh {} = {}'.format(P_effT, np.sum(correct_match)))
        print ('number of wrong matches with P_thresh {} = {}'.format(P_effT, np.sum(~correct_match)))
        self.P_effT = P_effT
