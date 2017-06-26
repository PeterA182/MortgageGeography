from __future__ import division

import sys
import gc
import os

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

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
random_forest = True
logistic_regression = False

prepped_outpath = '/Users/peteraltamura/Documents/GitHub/' \
                  'mortgageResearch/Data/monthly/preppedFeatures/'

# Read in
x_raw = pd.read_csv(prepped_outpath + 'featureSelectionFinal.csv')
y_raw = pd.read_csv(prepped_outpath + 'featureSelectionFinal_Y.csv')


# Prep for Model
train_df, test_df, train_y, test_y = train_test_split(x_raw, y_raw)

# Handle ID Col
train_df.drop(labels=['loanSeqNumber'], axis=1, inplace=True)
testLoanSeqNumbers = test_df['loanSeqNumber']
test_df.drop(labels=['loanSeqNumber'], axis=1, inplace=True)

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
    rf = RandomForestClassifier(n_estimators=1000)
    rf.fit(train_df, train_y)

    # Predict
    resp = rf.predict(test_df)


# Model Fit
if logistic_regression:
    logisticRegression = LogisticRegression()
    logisticRegression.fit(train_df, train_y)

    # Predict
    resp = logisticRegression.predict(test_df)


# Results
df_results = pd.DataFrame({
    'loanSeqNumber': list(testLoanSeqNumbers),
    'test_y': list(test_y),
    'predictedValue': list(resp)
})





