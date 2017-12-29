from __future__ import division

import sys
import gc
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/Load"
)
from load_loans import load_data
from reference import (compress_columns)

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/configs/"
)
from config import configs

"""
Logistic Regression for loans being more than 2 months delinquent
Only looking at most recent month

TODO:
add origination information
standardize costs
add time sensitive metrics / flags for past statuses

"""

sample_run = True
timePeriod = 'Q42015'
source = 'freddie'
random_forest = False
logistic_regression = False
kneighbors = True
performance_dict = {}

prepped_outpath = '/Users/peteraltamura/Documents/GitHub/' \
                  'mortgageResearch/Data/monthly/preppedFeatures/'

# Read in
x_raw = pd.read_csv(prepped_outpath + 'featureSelectionFinal.csv')
y_raw = pd.read_csv(prepped_outpath + 'featureSelectionFinal_Y.csv')


# Prep for Model
train_df, test_df, train_y, test_y = train_test_split(x_raw, y_raw)
testLoanSeqNumbers = test_df['loanSeqNumber']

# Handle ID Col
train_df.drop(labels=['loanSeqNumber',
                      'delinquencyFlag_mostRecentMth'],
              axis=1,
              inplace=True)
test_df.drop(labels=['loanSeqNumber',
                     'delinquencyFlag_mostRecentMth'],
             axis=1,
             inplace=True)

train_df = np.array(train_df)
test_df = np.array(test_df)
train_y = np.array(train_y)
test_y = np.array(test_y)

#
# ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ----
# ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ----

# Random Forest
# Model Fit
if random_forest:
    print "Random Forest Model: Training"
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(train_df, train_y)

    # Predict
    resp = rf.predict(test_df)


# Model Fit
if logistic_regression:
    print "Logistic Regression Model: Training"
    logisticRegression = LogisticRegression()
    logisticRegression.fit(train_df, train_y)

    # Predict
    resp = logisticRegression.predict(test_df)

if kneighbors:
    print "KNearest Neighbor Model: Training"
    for nn in range(1, 11):
        knn = KNeighborsClassifier(n_neighbors=nn,
                                   weights='uniform',
                                   algorithm='auto')
        knn.fit(train_df, train_y)

        # Predict
        resp = knn.predict(test_df)
        sc = knn.score(test_df, test_y)
        performance_dict.update({nn: sc})

# Results
df_results = pd.DataFrame({
    'loanSeqNumber': list(testLoanSeqNumbers),
    'test_y': list(test_y),
    'predictedValue': list(resp)
})
df_results.to_csv(prepped_outpath + 'df_results_{}_.csv'.format(
    'randomForestClassifier' if random_forest else 'logRegressionClassifier'
))




