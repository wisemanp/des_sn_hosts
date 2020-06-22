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

from des_mismatch.classification.classification_utils import (is_correct_host, prep_classification,
                                            get_features, find_nearest, score_func, score_func_CV)

warnings.simplefilter('ignore')

class Classifier():
    def __init__(self,df,config_path,split='even'):
        self._df=df
        self._config(config_path)
        self._train,self._test = self._split_data()
        self.X_train,self._y_train,self._X_test,self._y_test = self._prep_features(self,features)
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
