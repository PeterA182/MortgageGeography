import sys
import glob
import os

import pandas as pd
import datetime as dt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from statsmodels.tsa.stattools import adfuller

from collections import OrderedDict

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/Load"
)
from load_loans import load_data
from reference import (
    originationFileColList, monthlyFileColList, compress_columns
)

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

# Remove non-features
all_cols = [
    'loanSeqNumber', 'mthlyRepPeriod', 'currUPB', 'currLoanDelinqStatus',
    'loanAge', 'remainingMonthsLegMaturity', 'repurchaseFlag',
    'modificationFlag', 'zeroBalanceCode', 'zeroBalanceEffectiveDate',
    'currInterestRate', 'currDeferredUPB', 'dueDateLastPaidInstallment',
    'mtgInsuranceRecovery', 'netSalesProceeds', 'nonMtgInsuranceRecovery',
    'expenses', 'legalCosts', 'maintenancePreservtnCosts',
    'taxesInsuranceOwed', 'miscExpenses', 'actualLossCalculation',
    'modificationCost', 'Unknown'
]

if __name__ == "__main__":

    # Load in from CSV
    if sample_run:
        df_monthly = load_data(
            path=configs[source]['sample_monthly_dir'] +
                 configs[source]['sample_file_monthly'],
            columns=monthlyFileColList,
            date_col_fmt_dict={'firstPaymentDate': '%Y%m'}
        )

    elif not sample_run:
        df_monthly = pd.DataFrame()
        for path in glob.glob(configs[source]['sample_monthly_dir'] + '*.txt'):
            print "Loading File: {}".format(path)
            df_monthly = pd.concat(
                [
                    df_monthly,
                    load_data(
                        path=path,
                        columns=monthlyFileColList,
                        nrows=None,
                        date_col_fmt_dict={'firstPaymentDate': '%Y%m'},
                        error_bad_lines=True
                    )
                ],
                axis=0
            )

    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Sort by loanSeqNumber and reportingMonth
    for col in list(df_monthly.columns):
        if len(df_monthly.loc[df_monthly[col].isnull(), :]) == \
                len(df_monthly):
            df_monthly.drop(labels=[col], axis=1, inplace=True)
            print "Column: {} was dropped due to containing all null values".\
                format(str(col))

    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Sort by loanSeqNumber and reportingMonth
    df_monthly.sort_values(
        by=['loanSeqNumber', 'mthlyRepPeriod'],
        ascending=[True, True],
        inplace=True
    )


    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Zero Balance Code
    zeroBalanceDict = {1: 'voluntaryPayoff',
                       3: 'foreclosureAlternative',
                       6: 'repurchasePreDisposition',
                       9: 'reoDisposition'}
    for k, v in zeroBalanceDict.iteritems():
        df_monthly.loc[:, v] = 0
        df_monthly.loc[df_monthly['zeroBalanceCode'] == k, v] = 1
    df_monthly.drop(labels=['zeroBalanceCode'], axis=1, inplace=True)

    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Current Loan Delinquency Status
    loanDelinqDict = {0: 'loanCurrent',
                      1: 'loan30_60_delinq',
                      2: 'loan60_90_delinq',
                      3: 'loan90_120_delinq',
                      4: 'loan120_150_delinq',
                      5: 'loan150_180_delinq',
                      6: 'loan180_210_delinq',
                      7: 'loan_210_240_delinq',
                      'R': 'lenderREO'}

    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Remove unneeded dates
    for col in [
        'dueDateLastPaidInstallment',
        'Unknown'
    ]:
        try:
            df_monthly.drop(labels=col, axis=1, inplace=True)
        except ValueError as VE:
            continue

    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Format target column for delinquency > 1 month
    df_monthly.loc[:, 'delinquencyFlag'] = 0
    msk = (
        (df_monthly['currLoanDelinqStatus'] >= 2) |
        (df_monthly['currLoanDelinqStatus'] == 'R')
    )
    df_monthly.loc[msk, 'delinquencyFlag'] = 1

    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Repurchase Map
    repurchaseDict = {'Y': 1, 'N': 0, np.NaN: 0}
    df_monthly.loc[:, 'repurchaseFlag'] = \
        df_monthly['repurchaseFlag'].\
            map(lambda x: repurchaseDict[x]).astype(int)

    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Standardize and currency columns
    for col in [
        'currUPB', 'currDeferredUPB', 'modificationCost'
    ]:
        df_monthly.loc[:, col + '_stzd'] = abs(
            df_monthly[col] - df_monthly[col].mean()
        )
        df_monthly.loc[:, col].fillna(0, inplace=True)
        df_monthly.drop(labels=[col], axis=1, inplace=True)

    renamingDict = {
        col: col+'_mostRecentMth' for col in list(df_monthly.columns) if col != 'loanSeqNumber'
    }
    df_monthly.rename(
        columns=renamingDict,
        inplace=True
    )
    for col in df_monthly.columns:
        if 'Unnamed' in col or 'nknown' in col or col == '':
            df_monthly.drop(labels=[col], axis=1, inplace=True)
    print df_monthly.columns
    for col in list(df_monthly.columns):
        print col
        print df_monthly[col].value_counts()
        print "---- ---- ----"
        print "---- ---- ----"
    df_monthly.to_csv(prepped_outpath + 'preppedFeatureTable_mostRecentMonth.csv',
                      index=False)




