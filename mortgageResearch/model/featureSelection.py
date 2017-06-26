import sys
import gc
import os

import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

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

prepped_outpath = '/Users/peteraltamura/Documents/GitHub/' \
                  'mortgageResearch/Data/monthly/preppedFeatures/'
if not os.path.exists(prepped_outpath):
    os.makedirs(prepped_outpath)

# drop cols (for now)
drop_cols = ['MSA', 'postalCode', 'maturityDate',
             'firstPaymentDate', 'mthlyRepPeriod', 'borrowers',
             'repurchaseFlag', 'zeroBalanceEffectiveDate',
             'modificationCost', 'currLoanDelinqStatus', 'servicerName',
             'seller']

# Read in Feature maps
df_origination = pd.read_csv(
    prepped_outpath + 'preppedFeatureTable_origination.csv'
)

# Drop cols missing value
for col in ['origDTI_origination', 'mtgInsurancePct_origination']:
    df_origination.loc[(
        (df_origination[col] == '   ')
        |
        (df_origination[col].isnull())
        |
        (df_origination[col] == '000')), col] = np.median(df_origination[col])
# Handle columns for origination file
for col in list(df_origination.columns):
    if 'Unnamed' in col or 'nknown' in col or col == '':
        df_origination.drop(labels=[col], axis=1, inplace=True)

df_mostRecentMonth = pd.read_csv(
    prepped_outpath + 'preppedFeatureTable_mostRecentMonth.csv'
)

# Handle columns for monthly file
for col in list(df_mostRecentMonth.columns):
    if 'Unnamed' in col or 'nknown' in col or col == '':
        df_mostRecentMonth.drop(labels=[col], axis=1, inplace=True)

# Merge
df_allFeatures = pd.merge(
    left=df_origination,
    right=df_mostRecentMonth,
    how='inner',
    on=['loanSeqNumber']
)

# Drop columns
for col in list(df_allFeatures.columns):
    if any([x in col for x in drop_cols]):
        df_allFeatures.drop(labels=[col], axis=1, inplace=True)

# Get features together
features_final = [col for col in list(df_allFeatures.columns) if
                  col not in ['delinquency_Y']
                  and col not in [
                      'sellerName_origination', 'servicerName_origination',
                      'borrowers_origination'
                  ]]
predict_final = ['delinquencyFlag_mostRecentMth']


df_final = df_allFeatures.copy(deep=True)

del df_allFeatures
gc.collect()

for col in list(df_final.columns):
    print col

    msk = ((df_final[col].astype(str) == 'nan') |
            (df_final[col].isnull()))

    if len(df_final.loc[msk, :]) > 0:
        print "Mask succeeded"
        repl = np.median(df_final.loc[~msk, :][col].astype(float))
        df_final.loc[msk, col] = repl

    print " --- ---- ----"
    try:
        df_final.loc[:, col] = \
            df_final[col].astype(float)
    except ValueError as VE:
        if col != 'loanSeqNumber':
            raise VE
        else:
            continue

# X and Y
x_raw = df_final.loc[:, features_final]
y_raw = df_final.loc[:, predict_final]


# ---- ---- ----      ---- ---- ----
#
# Handle Colinearity
# VIF = list()
# for i in range(x_raw.shape[1]):
#     VIF.append(variance_inflation_factor(np.array(x_raw.dropna()), i))
#
# while np.max(VIF) > 3:
#     msk = np.argmax(VIF)
#     x_raw.drop(x_raw.columns[msk], axis=1, inplace=True)
#     VIF = list()
#     for i in range(x_raw.shape[1]):
#         VIF.append(variance_inflation_factor(np.array(x_raw.dropna()), i))
#
# endog = y_raw
# exog = sm.add_constant(x_raw)
#
# formula =


x_raw.to_csv(prepped_outpath + 'featureSelectionFinal.csv', index=False)
y_raw.to_csv(prepped_outpath + 'featureSelectionFinal_Y.csv', index=False)

