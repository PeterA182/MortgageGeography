from __future__ import division

# Local
import sys

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/configs/"
)
from config import configs


# Other
import pandas as pd
import datetime as dt
import numpy as np

#
# Import run variales
d_outpath = sys.argv[1]
d_source = sys.argv[2]
model_name = sys.argv[3]
default_window_months = configs[d_source][model_name]['default_window_months']


#
# ---- ---- ----
# Methods
def add_target(df, all):
    """
    
    :param df: 
    :param all: 
    :return: 
    """

    # Aggregate
    tot = df.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'currLoanDelinqStatus': np.max})
    tot.rename(
        columns={'currLoanDelinqStatus': 'target'},
        inplace=True
    )

    # Reset to 0, 1
    tot.loc[:, 'default_in_window'] = 0
    tot.loc[tot['target'] > 0, 'default_in_window'] = 1
    tot.drop(labels=['target'], axis=1, inplace=True)

    # Merge back
    all_ret = pd.merge(
        all,
        tot[['loanSeqNumber', 'default_in_window']],
        how='left',
        on='loanSeqNumber'
    )
    return all_ret


def add_loan_age(months_level, loan_level):
    """

    :param months_level: 
    :param loan_level: 
    :return: 
    """
    # age of loan
    add = months_level.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'loan_months_total': np.mean})
    add = add.loc[:,
          ['loanSeqNumber', 'loan_months_total']].drop_duplicates(
        inplace=False
    )

    pre_len = len(loan_level)
    loan_level = pd.merge(
        loan_level,
        add,
        how='left',
        on=['loanSeqNumber']
    )
    if len(loan_level) > pre_len:
        raise Exception(
            "Merge of type \'left\' has added rows unexpectedly."
        )

    return loan_level


def add_prev_defaults(month_level, loan_level):
    """

    :param month_level: 
    :param loan_level: 
    :return: 
    """

    msk = (
        month_level.groupby(['loanSeqNumber'])
        ['currLoanDelinqStatus'].transform('max') > 0
    )
    month_level.loc[:, 'prev_defaults'] = 0
    month_level.loc[msk, 'prev_defaults'] = 1
    month_level = month_level.groupby(['loanSeqNumber'], as_index=False).agg(
        {'prev_defaults': np.max}
    )
    pre_len =len(loan_level)
    loan_level = pd.merge(
        loan_level,
        month_level,
        how='left',
        on=['loanSeqNumber']
    )
    if len(loan_level) > pre_len:
        raise Exception("LATER")


    # Feature: months spent in default before
    add = month_level.groupby(['loanSeqNumber'], as_index=False).agg(
        {'prev_defaults': np.sum})
    add.rename(columns={'prev_defaults': 'prev_defaults_sum'}, inplace=True)

    loan_level = pd.merge(
        loan_level,
        add,
        how='left',
        on=['loanSeqNumber']
    )
    if len(loan_level) > pre_len:
        raise Exception("LATER")

    return loan_level


def add_upb_max_change_rate(month_level, loan_level):
    """
    Calculate the max rate of change in UPB during life of mortgage across all 
    possible window sizes
    
    
    PARAMETERS
    ----------
    month_level: DataFrame
        contains monthly observations for each mortgage
    loan_level: DataFramr
        contains mortgage level observations for each mortgage
    
        
    RETURNS
    -------
    loan_level DataFrame with metrics appended
    """

    # Get max number of months for any loan
    #   (this -1 will serve as max rolling_avg look back)
    month_level.sort_values(by=['loanSeqNumber', 'mthlyRepPeriod'],
                            ascending=True,
                            inplace=True)
    month_level['max_months'] = month_level.groupby(
        by=['loanSeqNumber'],
        as_index=False
    )['mthlyRepPeriod'].transform('count')

    # Iterate
    windows = range(1, np.max(month_level['max_months']))
    for m in windows:
        month_level.loc[:, 'max_upb_range_{}'.format(str(m))] = \
            month_level.groupby('loanSeqNumber')['currUPB'].apply(
                pd.rolling_max, m, min_periods=m
            )
        month_level.loc[:, 'min_upb_range_{}'.format(str(m))] = \
            month_level.groupby('loanSeqNumber')['currUPB'].apply(
                pd.rolling_min, m, min_periods=m
            )
        month_level.loc[:, 'max_upb_change_rate_{}'.format(str(m))] = ((
            month_level['max_upb_range_{}'.format(str(m))] -
            month_level['min_upb_range_{}'.format(str(m))]
        ) / m)

    # Aggregate to loan level to merge back
    add = month_level.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'max_upb_change_rate_{}'.format(str(m)): 'max' for m in windows})

    # Merge back
    loan_level = pd.merge(
        loan_level,
        add,
        how='left',
        on=['loanSeqNumber']
    )

    return loan_level


def add_variability_measures(month_level, loan_level):
    """
    
    :param month_level: 
    :param loan_level: 
    :return: 
    """

    # ---------
    # IQR
    # Percentile 75
    add75 = month_level.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'currUPB': lambda x: np.percentile(x, q=75)})
    add75.rename(columns={'currUPB': 'pctl75'}, inplace=True)
    add25 = month_level.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'currUPB': lambda x: np.percentile(x, q=25)})
    add25.rename(columns={'currUPB': 'pctl25'}, inplace=True)
    pre_len = len(loan_level)
    add = pd.merge(
        add75,
        add25,
        how='left',
        on=['loanSeqNumber']
    )
    if len(add) > pre_len:
        raise Exception("LATER")

    add['loan_IQR'] = (add['pctl75'] - add['pctl25'])

    pre_len = len(loan_level)
    loan_level = pd.merge(
        loan_level,
        add[['loanSeqNumber', 'loan_IQR']],
        how='left',
        on=['loanSeqNumber']
    )
    if len(loan_level) > pre_len:
        raise Exception("Merge of type \'left\' has added rows.")

    # ---------
    # Range
    month_level['max_UPB'] = month_level.groupby(
        by=['loanSeqNumber'],
        as_index=False
    )['currUPB'].transform(np.max)
    month_level['min_UPB'] = month_level.groupby(
        by=['loanSeqNumber'],
        as_index=False
    )['currUPB'].transform(np.min)
    month_level['loan_Range'] = (
        month_level['max_UPB'] - month_level['min_UPB']
    )
    add = month_level.loc[:, ['loanSeqNumber', 'loan_Range']].\
        drop_duplicates(inplace=False)
    pre_len  = len(loan_level)
    loan_level = pd.merge(
        loan_level,
        add,
        how='left',
        on=['loanSeqNumber']
    )
    if len(loan_level) > pre_len:
        raise Exception("Merge of type \'left\' has added rows unexpectedly")

    # ---------
    # Variance and STD
    month_level['UPB_var'] = month_level.groupby(
        by=['loanSeqNumber'],
        as_index=False
    )['currUPB'].transform(np.var)

    add = month_level.loc[:, ['loanSeqNumber', 'UPB_var']]. \
        drop_duplicates(inplace=False)
    pre_len = len(loan_level)
    loan_level = pd.merge(
        loan_level,
        add,
        how='left',
        on=['loanSeqNumber']
    )
    if len(loan_level) > pre_len:
        raise Exception(
            "Merge of type \'left\' has added rows unexpectedly")

    return loan_level


if __name__ == "__main__":

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Prep / Initial

    # Read in prepped monthly data
    df_comb = pd.read_pickle(
        d_outpath + 
        configs[d_source][model_name]['combined_filenames']['prepped'] + '.pkl'
    )
    for c in df_comb.columns:
        if df_comb[c].dtype not in [object, '<M8[ns]']:
            df_comb.loc[:, c] = df_comb[c].astype(float)
        df_comb.loc[:, 'mthlyRepPeriod'] = \
            pd.to_datetime(df_comb['mthlyRepPeriod'])

    #
    # Rank each month's observation for each loan
    df_comb.sort_values(
        by=['loanSeqNumber', 'mthlyRepPeriod'],
        ascending=True,
        inplace=True
    )
    df_comb['mth_of_loan'] = df_comb.groupby(
        by=['loanSeqNumber'],
        as_index=True
    )['mthlyRepPeriod'].rank(method='min').astype(int)

    #
    # Set up table to append metrics to
    df_loans = df_comb.loc[:,
        ['loanSeqNumber', 'creditScore'] + [x for x in df_comb.columns if x[:4] == 'orig']
    ].drop_duplicates(inplace=False)
    assert len(df_loans) == len(set(df_loans.loanSeqNumber))

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Add Target

    # Remove recent "x" number of observations to create predict "window"
    df_comb['loan_months_total'] = df_comb.groupby(
        by=['loanSeqNumber'],
        as_index=False
    )['mth_of_loan'].transform(np.max)
    window_months_msk = (
        df_comb['mth_of_loan'] <
        (df_comb['loan_months_total'] -
         configs[d_source][model_name]['default_window_months'])
    )

    # Split into pre-prediction window and post-prediction window
    # chronologically
    df_comb_target_window = df_comb.loc[~window_months_msk, :]
    df_comb = df_comb.loc[window_months_msk, :]

    # Target
    df_loans = add_target(df=df_comb_target_window, all=df_loans)
    print len(df_loans)
    print "Length loans after target"
    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Feature Engineering

    # Feature: been in default before
    df_loans = add_prev_defaults(month_level=df_comb,
                                 loan_level=df_loans)
    print len(df_loans)
    print "Length loans after add_prev_defaults"

    # Feature: age of loan
    df_loans = add_loan_age(months_level=df_comb,
                            loan_level=df_loans)
    print len(df_loans)
    print "Length loans after age"

    # Feature: currUPB Variability Measures
    df_loans = add_variability_measures(month_level=df_comb,
                                        loan_level=df_loans)
    print len(df_loans)
    print "Length loans after variability"

    # Feature: currUPB max rate of change
    df_loans = add_upb_max_change_rate(month_level=df_comb,
                                       loan_level=df_loans)
    print len(df_loans)
    print "Length loans after upb_max"

    # Feature: LTV diff as function of time between origination and max
    # non-target month


    # UPB diff as function of time
    
    #
    # Prep to Send back out
    # ---- ---- ----
    
    # Remove all null observations in each columns
    for c in df_loans.columns:
        print c
        if sum(df_loans[c].isnull()) > 0:
            print str(sum(df_loans[c].isnull())) + " Null Values"
            df_loans = df_loans.loc[df_loans[c].notnull(), :]

    print len(df_loans)
    print "Above length df_loans after all features"
    df_loans.to_pickle(
        d_outpath +
        configs[d_source][model_name]['combined_filenames']['FE'] + '.pkl'
    )

