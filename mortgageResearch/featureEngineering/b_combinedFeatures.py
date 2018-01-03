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


# Methods
def add_feature(add, all, metric):
    """
    
    :param df: 
    :return: 
    """

    pre_len = len(all)
    all = pd.merge(
        all,
        add,
        how='left',
        on=['loanSeqNumber']
    )
    if len(all) != pre_len:
        raise Exception("Merge of tye \'left\' has added rows unexpectedly "
                        "for metrics {}".format(metric))

    return all


def add_target(df, all):
    """
    
    :param df: 
    :param all: 
    :return: 
    """
    print df.head()
    print df.columns

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
    print tot.head()
    print tot.columns
    tot.loc[:, 'default_in_window'] = 0
    tot.loc[tot['target'] > 0, 'default_in_window'] = 1
    tot.drop(labels=['target'], axis=1, inplace=True)

    # Merge back
    all_ret = pd.merge(
        all,
        tot,
        how='left',
        on='loanSeqNumber'
    )
    return all_ret


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

    # Add Target
    df_loans = add_target(df=df_comb_target_window, all=df_loans)
    
    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Feature Engineering

    # Feature: been in default before
    msk = (
        df_comb.groupby(['loanSeqNumber'])
        ['currLoanDelinqStatus'].transform('max') > 0
    )
    df_comb.loc[:, 'prev_defaults'] = 0
    df_comb.loc[msk, 'prev_defaults'] = 1
    add = df_comb.groupby(['loanSeqNumber'], as_index=False).agg({'prev_defaults': np.max})
    df_loans = add_feature(add=add, all=df_loans, metric='prev_defaults')

    # Feature: months spent in default before
    add = df_comb.groupby(['loanSeqNumber'], as_index=False).agg({'prev_defaults': np.sum})
    add.rename(columns={'prev_defaults': 'prev_defaults_sum'}, inplace=True)
    df_loans = add_feature(add=add, all=df_loans, metric='prev_defaults_sum')

    # age of loan
    add = df_comb.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'loan_months_total': np.mean})
    df_loans = add_feature(add=add, all=df_loans, metric='loan_months_total')

    # currUPB
    # Feature: currUPB variance between origination and max non-target month
    add = df_comb.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'currUPB': np.var})
    add.rename(columns={'currUPB': 'UPB_var'}, inplace=True)
    df_loans = add_feature(add=add, all=df_loans, metric='UPB_var')

    # Feature: currUPB max diff
    add = df_comb.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'currUPB': [np.max, np.min]})
    add.columns = [x[1] if (x[1] and x[1] != '') else x[0] for x in add.columns]
    add['UPB_max_diff'] = (add['amax'] - add['amin'])
    df_loans = add_feature(add=add, all=df_loans, metric='UPB_max_diff')

    # Feature: LTV diff as function of time between origination and max
    # non-target month


    # UPB diff as function of time
    
    #
    # Prep to Send back out
    # ---- ---- ----
    
    # Remove all null observations in each columns
    for c in df_loans.columns:
        if sum(df_loans[c].isnull()) > 0:
            df_loans = df_loans.loc[df_loans[c].notnull(), :]

    df_loans.to_pickle(
        d_outpath +
        configs[d_source][model_name]['combined_filenames']['FE'] + '.pkl'
    )

