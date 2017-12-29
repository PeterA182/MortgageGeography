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

# ---- ---- ----
# Methods

def clean_features(table, origination_file=False, monthly_file=False,
                   **kwargs):
    """
    Handles out of place values that represent the same as a null
    
    PARAMETERS
    ----------
    table: DataFrame
        table of features
    """

    if not any([origination_file, monthly_file]):
        raise Exception("Must pick at least one type of "
                        "file for cleaning")
    if all([origination_file, monthly_file]):
        raise Exception("Must pick only one type of "
                        "file for cleaning at a time")

    # Origination file cleaning
    if origination_file:

        # Drop cols missing value
        for col in ['origDTI_origination', 'mtgInsurancePct_origination']:
            table.loc[(
                (table[col] == '   ')
                |
                (table[col].isnull())
                |
                (table[
                    col] == '000')), col] = np.median(table[col])

        # Handle columns for origination file
        for col in list(table.columns):
            if 'Unnamed' in col or 'nknown' in col or col == '':
                table.drop(labels=[col], axis=1, inplace=True)

    # Monthly file cleaning
    if monthly_file:

        # Handle columns for monthly file
        for col in list(table.columns):
            if 'Unnamed' in col or 'nknown' in col or col == '':
                table.drop(labels=[col], axis=1, inplace=True)

    # Drops
    if kwargs.get('drop_columns'):
        table.drop(labels=[col for col in list(table.columns)
                           if col in kwargs.get('drop_columns')],
                   axis=1,
                   inplace=True)

    return table


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

# Clean
df_origination = clean_features(table=df_origination,
                                origination_file=True,
                                drop_columns=drop_cols)


df_mostRecentMonth = pd.read_csv(
    prepped_outpath + 'preppedFeatureTable_mostRecentMonth.csv'
)

# Clean
df_mostRecentMonth = clean_features(table=df_mostRecentMonth,
                                    monthly_file=True,
                                    drop_columns=drop_cols)


# Merge Cleaned Tables
df_allFeatures = pd.merge(
    left=df_origination,
    right=df_mostRecentMonth,
    how='inner',
    on=['loanSeqNumber']
)


# Get features together
features_final = [col for col in list(df_allFeatures.columns) if
                  col not in ['delinquency_Y']
                  and col not in [
                      'sellerName_origination', 'servicerName_origination',
                      'borrowers_origination', 'loanSeqNumber'
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
        continue

# X and Y
x_raw = df_final.loc[:, features_final]

y_raw = df_final.loc[:, predict_final]


# ---- ---- ----      ---- ---- ----
#
# Handle Colinearity
# More extreme standard error w/ colinearity
VIF = list()
for i in range(x_raw.shape[1]):
    print x_raw.columns[i]
    val = variance_inflation_factor(np.array(x_raw.dropna()), i)
    if val == np.inf or val == np.nan:
        x_raw.drop(labels=x_raw.columns[i], axis=1, inplace=True)
    else:
        VIF.append(variance_inflation_factor(np.array(x_raw.dropna()), i))

print VIF
print "---- ---- ----"


while np.max(VIF) > 2:
    msk = np.argmax(VIF)
    x_raw.drop(x_raw.columns[msk], axis=1, inplace=True)
    VIF = list()
    for i in range(x_raw.shape[1]):
        t = np.array(x_raw.dropna())
        VIF.append(variance_inflation_factor(np.array(x_raw.dropna()), i))

endog = y_raw
exog = sm.add_constant(x_raw)
print exog
print [x for x in features_final if x not in list(x_raw.columns)]

#
# formula


x_raw.to_csv(prepped_outpath + 'featureSelectionFinal.csv', index=False)
y_raw.to_csv(prepped_outpath + 'featureSelectionFinal_Y.csv', index=False)

