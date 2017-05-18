import sys
import os

import pandas as pd
import numpy as np
import datetime as dt

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/Load"
)
from load_loans import load_data
from reference import (
    originationFileColList, monthlyFileColList, compress_columns
)


# Define out Variables
sample_dir = "/Users/peteraltamura/Documents/GitHub/mortgageResearch/Data/" \
             "historical_data1_Q12016/"
sample_file = "historicalData1_Q12016.csv"
sample_file_monthly = "historicalDataMonthly1_Q12016.csv"


def lender_PerformanceByMSA():
    """

    :return:
    """

    # Load in from CSV
    df_origination = load_data(
        path=sample_dir,
        filename=sample_file,
        columns=originationFileColList,
        date_col_fmt_dict={'firstPaymentDate': '%Y%m'}
    )
    print df_origination.columns

    df_origination.loc[:, 'creditScore'] = \
        pd.to_numeric(df_origination['creditScore'])

    # Credit Score Spread by MSA
    df_msa_to_score = df_origination.groupby(
        by=['MSA'],
        as_index=False
    ).agg({'creditScore': [np.mean, np.std, np.max, np.min, np.var]})
    df_msa_to_score = compress_columns(df_msa_to_score)

    df_msa_to_score.rename(
        columns={
            'mean': 'MSACreditScore_mean',
            'std': 'MSACreditScore_std',
            'amax': 'MSACreditScore_max',
            'amin': 'MSACreditScore_min',
            'var': 'MSACreditScore_var'
        },
        inplace=True
    )

    # Lender selection of Credit Score
    lender = df_origination.groupby(
        by=['servicerName', 'MSA'],
        as_index=False
    ).agg({'creditScore': [np.mean, 'count']})
    lender = compress_columns(lender)

    lender.rename(
        columns={
            'mean': 'LenderMSACreditScore_mean',
            'count': 'totalMortgages'
        },
        inplace=True
    )

    lender = pd.merge(
        lender,
        df_msa_to_score,
        how='left',
        left_on=['MSA'],
        right_on=['MSA']
    )

    # Z - Score of Lender Avg in MSA comp to MSA Avg
    lender['lenderAvg_zScore'] = 0
    lender.loc[:, 'zScore'] = (
        (lender['LenderMSACreditScore_mean'] - lender['MSACreditScore_mean'])
        / lender['MSACreditScore_std']
    )

    return lender


def lender_CreditScoreSelection():
    """
    
    :return: 
    """

    # Load in from CSV
    df_origination = load_data(
        path=sample_dir,
        filename=sample_file,
        columns=originationFileColList,
        date_col_fmt_dict={'firstPaymentDate': '%Y%m'}
    )

    df_origination.loc[:, 'creditScore'] = \
        pd.to_numeric(df_origination['creditScore'])

    # Credit Score Spread by MSA
    df_msa_to_score = df_origination.groupby(
        by=['MSA'],
        as_index=False
    ).agg({'creditScore': [np.mean, np.std, np.max, np.min, np.var]})
    df_msa_to_score = compress_columns(df_msa_to_score)

    df_msa_to_score.rename(
        columns={
            'mean': 'MSACreditScore_mean',
            'std': 'MSACreditScore_std',
            'amax': 'MSACreditScore_max',
            'amin': 'MSACreditScore_min',
            'var': 'MSACreditScore_var'
        },
        inplace=True
    )

    # Loan Level (Safety Merge -no Pandas alert if left adds rows)
    pre_len = len(df_origination)
    df_loan = pd.merge(
        df_origination,
        df_msa_to_score,
        how='left',
        on=['MSA']
    )
    if len(df_loan) != pre_len:
        raise Exception("Merge of type \'left\' has added rows unexpectedly.")

    # Outlier Over
    msk = (
        df_loan['creditScore'] >= (
            df_loan['MSACreditScore_mean'] +(df_loan['MSACreditScore_std']*3)
        )
    )
    df_loan['creditScoreOutlier_over'] = 0
    df_loan.loc[msk, 'creditScoreOutlier_over'] = 1

    # Outlier Under
    msk = (df_loan['creditScore'] <= (
            df_loan['MSACreditScore_mean'] - (df_loan['MSACreditScore_std'] * 3)
        )
    )
    df_loan['creditScoreOutlier_under'] = 0
    df_loan.loc[msk, 'creditScoreOutlier_under'] = 1

    df_loan['year'] = df_loan.loc[:, 'firstPaymentDate'].dt.year






if __name__ == "__main__":

    # Load MSA level performance by Lender
    lender_by_msa = lender_PerformanceByMSA()

    # Load performance by Lender
    lender_by_loan = lender_CreditScoreSelection()

    print lender_by_loan['creditScoreOutlier_over'].mean()
    print lender_by_loan['creditScoreOutlier_under'].mean()