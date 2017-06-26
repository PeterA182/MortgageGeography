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
    col for col in originationFileColList if col not in monthlyFileColList
]

# drop cols (for now)
drop_cols = ['MSA', 'postalCode', 'maturityDate',
             'firstPaymentDate', 'propertyState']


if __name__ == "__main__":

    # Load in from CSV
    if sample_run:
        df_origination = load_data(
            path=configs[source]['sample_single_dir'] +
                 configs[source]['sample_single_file'],
            columns=originationFileColList,
            date_col_fmt_dict={'firstPaymentDate': '%Y%m'}
        )

    elif not sample_run:
        df_origination = pd.DataFrame()
        for path in glob.glob(configs[source]['sample_single_dir'] + '*.txt'):
            print "Loading File: {}".format(path)
            df_origination = pd.concat(
                [
                    df_origination,
                    load_data(
                        path=path,
                        columns=originationFileColList,
                        nrows=None,
                        date_col_fmt_dict={'firstPaymentDate': '%Y%m'},
                        error_bad_lines=True
                    )
                ],
                axis=0
            )

    # Subset to columns not found in the monthly file
    df_origination = df_origination.loc[:, ['loanSeqNumber'] + all_cols]

    # Drop cols of all nulls
    for col in list(df_origination.columns):
        if len(df_origination.loc[df_origination[col].isnull(), :]) == \
                len(df_origination):
            df_origination.drop(labels=[col], axis=1, inplace=True)

        # Drop cols if not null but all equal
        else:
            try:
                if df_origination[col].value_counts()[0] == len(df_origination):
                    df_origination.drop(labels=[col], axis=1, inplace=True)
                else:
                    continue
            except KeyError as KE:
                continue

    # Unpack Loan Purpose
    loanPurposeDict = {
        'P': 'purchaseIntent',
        'C': 'cashOutRefinanceIntent',
        'N': 'noCashOutRefinanceIntent'
    }
    for k, v in loanPurposeDict.iteritems():
        df_origination.loc[:, v] = 0
        df_origination.loc[df_origination['loanPurpose'] == k, v] = 1
    df_origination.drop(labels=['loanPurpose'], axis=1, inplace=True)

    # Unpack first time buy flag
    msk = (df_origination['firstTimeBuyFlag'] == 'Y')
    df_origination.loc[msk, 'firstTimeBuyFlag'] = 1
    df_origination.loc[~msk, 'firstTimeBuyFlag'] = 0

    # Unpack propertyType
    propertyTypeDict = {
        'CO': 'condo',
        'LH': 'leasehold',
        'PU': 'PUD',
        'MH': 'manufacturedHousing',
        'SF': '14feeSimple',
        'CP': 'co_op'
    }
    for k, v in propertyTypeDict.iteritems():
        msk = (df_origination['propertyType'] == k)
        df_origination[v] = 0
        df_origination.loc[msk, v] = 1

    df_origination.drop(labels=['propertyType'], axis=1, inplace=True)

    # Unpack Occupancy Status
    occMap = {
        'O': 'ownerOccupied',
        'I': 'investmentOccupied',
        'S': 'secondHomeOccupied'
    }
    for k, v in occMap.iteritems():
        df_origination.loc[:, v] = 0
        df_origination.loc[df_origination['occStatus'] == k, v] = 1
    df_origination.drop(labels=['occStatus'], axis=1, inplace=True)

    # Unpack Channel
    channelDict = {
        'R': 'retailChannel',
        'B': 'brokerChannel',
        'C': 'correspondentChannel',
        'T': 'tpoChannel'
    }
    for k, v in channelDict.iteritems():
        df_origination.loc[:, v] = 0
        df_origination.loc[df_origination['channel'].apply(lambda x: x.strip()) == k, v] = 1
    df_origination.drop(labels=['channel'], axis=1, inplace=True)
    print df_origination['tpoChannel'].value_counts()

    # Unpack Prepay Penalty
    msk = (df_origination['prepayPenaltyMtgFlag'] == 'Y')
    df_origination.loc[msk, 'prepayPenaltyMtgFlag'] = 1
    df_origination.loc[~msk, 'prepayPenaltyMtgFlag'] = 0

    # Unpack super conforming flag
    msk = (df_origination['superConformingFlag'] == 'Y')
    df_origination.loc[msk, 'superConformingFlag'] = 1
    df_origination.loc[~msk, 'superConformingFlag'] = 0

    # Drop dropcils
    for col in drop_cols:
        try:
            df_origination.drop(labels=[col], axis=1, inplace=True)
        except KeyError as KE:
            continue

    # Save out
    renamingDict = {
        col: col + '_origination' for col in list(df_origination.columns) if
    col != 'loanSeqNumber'
    }
    df_origination.rename(
        columns=renamingDict,
        inplace=True
    )
    print df_origination['tpoChannel_origination'].value_counts()

    # Remove any column not named by the columns list
    # (mismatch to online DataDict)
    for col in list(df_origination.columns):
        if 'Unnamed' in col or 'nknown' in col or col == '':
            df_origination.drop(labels=[col], axis=1, inplace=True)

    print df_origination['tpoChannel_origination'].value_counts()

    df_origination.to_csv(prepped_outpath + 'preppedFeatureTable_origination.csv')