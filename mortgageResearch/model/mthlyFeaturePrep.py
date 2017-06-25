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

sample_run = True
timePeriod = 'Q42015'
source = 'freddie'
min_pmts = 10
p_value_sig = .05

prepped_outpath = '/Users/peteraltamura/Documents/GitHub/' \
                  'mortgageResearch/Data/monthly/preppedFeatures/'
if not os.path.exists(figure_outpath):
    os.makedirs(figure_outpath)

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

# Base features
base_fts = [
    'currUPB', 'currLoanDelinqStatus', 'loanAge',
    'remainingMonthsLegMaturity', 'repurchaseFlag', 'modificationFlag',
    'zeroBalanceCode', 'currInterestRate', 'currDeferredUPB',
    'mtgInsuranceRecovery', 'netSalesProceeds', 'nonMtgInsuranceRecovery',
    'expenses', 'legalCosts', 'maintenancePreservtnCosts',
    'taxesInsuranceOwed', 'miscExpenses', 'actualLossCalculation',
    'modificationCost'
]

if __name__ == "__main__":

    # Load in from CSV
    if sample_run:
        df_origination = load_data(
            path=configs[source]['sample_monthly_dir'] +
                 configs[source]['sample_file_monthly'],
            columns=monthlyFileColList,
            date_col_fmt_dict={'firstPaymentDate': '%Y%m'}
        )

    elif not sample_run:
        df_origination = pd.DataFrame()
        for path in glob.glob(configs[source]['sample_monthly_dir'] + '*.txt'):
            print "Loading File: {}".format(path)
            df_origination = pd.concat(
                [
                    df_origination,
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

    # Zero Balance Code
    zeroBalanceDict = {1: 'voluntaryPayoff',
                       3: 'foreclosureAlternative',
                       6: 'repurchasePreDisposition',
                       9: 'reoDisposition'}
    for k, v in zeroBalanceDict.iteritems():
        df_origination.loc[:, v] = 0
        df_origination.loc[df_origination['zeroBalanceCode'] == k, v] = 1
    df_origination.drop(labels=['zeroBalanceCode'], axis=1, inplace=True)

    # Modification Flag
    df_origination.loc[
        df_origination['modificationFlag'].notnull(), 'modificationFlag'] = 1
    df_origination.loc[
        df_origination['modificationFlag'].isnull(), 'modificationFlag'] = 0

    # Remove unneeded dates
    for col in [
        'zeroBalanceEffectiveDate', 'dueDateLastPaidInstallment',
        'Unknown'
    ]:
        try:
            df_origination.drop(labels=col, axis=1, inplace=True)
        except KeyError as KE:
            continue

    df_origination.to_csv(prepped_outpath + 'preppedFeatureTable.csv')



