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
# d_outpath = '/Users/peteraltamura/Documents/GitHub/mortgageResearch/output/' \
#             'combinedDefaultPred_logit/'
# d_source = 'freddie'
# model_name = 'comb_logisticRegression'
# default_window_months = 3


# Methods
def add_feature(add, all, metric):
    """
    
    :param df: 
    :return: 
    """

    pre_len = len(all)
    print sorted(list(all.columns))
    print sorted(list(add.columns))
    all = pd.merge(
        all,
        add,
        how='left',
        on=['loanSeqNumber']
    )
    if len(all) != pre_len:
        raise Exception("Merge of tye \'left\' has added rows unexpectedly "
                        "for metrics {}".format(metric))


def add_target(df, all):
    """
    
    :param add: 
    :param all: 
    :return: 
    """

    # Aggregate and create target
    df = df.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'currLoanDelinqStatus': np.max})
    df.loc[df['currLoanDelinqStatus'] > 0, 'currLoanDelinqStatus'] = 1

    # Merge back to main table
    pre_len = len(all)
    all = pd.merge(
        all,
        df,
        how='left',
        on=['loanSeqNumber']
    )
    if len(all) != pre_len:
        raise Exception("Merge of type \'left\' for target has added rows "
                        "unexpectedly.")



if __name__ == "__main__":

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Prep / Initial

    # Read in prepped monthly data
    df_monthly = pd.read_pickle(
        d_outpath + configs[d_source][model_name]['combined_filenames']['prepped'] + '.pkl'
    )
    for c in df_monthly.columns:
        if df_monthly[c].dtype not in [object, '<M8[ns]']:
            df_monthly.loc[:, c] = df_monthly[c].astype(float)
    df_monthly.loc[:, 'mthlyRepPeriod'] = \
        pd.to_datetime(df_monthly['mthlyRepPeriod'])

    #
    # Rank each month's observation for each loan
    df_monthly.sort_values(
        by=['loanSeqNumber', 'mthlyRepPeriod'],
        ascending=True,
        inplace=True
    )
    df_monthly['mth_of_loan'] = df_monthly.groupby(
        by=['loanSeqNumber'],
        as_index=True
    )['mthlyRepPeriod'].rank(method='min').astype(int)

    #
    # Set up table to append metrics to
    df_loans = df_monthly.loc[:, ['loanSeqNumber']].\
        drop_duplicates(inplace=False)

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Add Target

    # Remove recent "x" number of observations to create predict "window"
    df_monthly['loan_months_total'] = df_monthly.groupby(
        by=['loanSeqNumber'],
        as_index=False
    )['mth_of_loan'].transform(np.max)
    window_months_msk = (
        df_monthly['mth_of_loan'] <
        (df_monthly['loan_months_total'] -
         configs[d_source][model_name]['default_window_months'])
    )

    # Split into pre-prediction window and post-prediction window
    # chronologically
    df_monthly = df_monthly.loc[window_months_msk, :]
    df_monthly_target_window = df_monthly.loc[~window_months_msk, :]

    # Add Target
    add_target(df=df_monthly_target_window, all=df_loans)
    
    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----
    #
    # Feature Engineering

    # Feature: been in default before
    msk = (
        df_monthly.groupby(['loanSeqNumber'])
        ['currLoanDelinqStatus'].transform('max') > 0
    )
    df_monthly.loc[:, 'prev_defaults'] = 0
    df_monthly.loc[msk, 'prev_defaults'] = 1
    add = df_monthly.groupby(['loanSeqNumber'], as_index=False).agg({'prev_defaults': np.max})
    add_feature(add=add, all=df_loans, metric='prev_defaults')

    # Feature: months spent in default before
    add = df_monthly.groupby(['loanSeqNumber'], as_index=False).agg({'prev_defaults': np.sum})
    add_feature(add=add, all=df_loans, metric='prev_defaults_sum')

    # age of loan
    add = df_monthly.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'loan_months_total': np.mean})
    add_feature(add=add, all=df_loans, metric='loan_months_total')

    # currUPB
    # Feature: currUPB variance between origination and max non-target month
    add = df_monthly.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'currUPB': np.var})
    add.rename(columns={'currUPB': 'UPB_var'}, inplace=True)
    add_feature(add=add, all=df_loans, metric='UPB_var')

    # Feature: currUPB max diff
    add = df_monthly.groupby(
        by=['loanSeqNumber'],
        as_index=False
    ).agg({'currUPB': [np.max, np.min]})
    add['UPD_diff'] = (add['max'] - add['min'])
    add_feature(add=add, all=df_loans, metric='UPB_max_diff')

    # Feature: LTV diff as function of time between origination and max
    # non-target month


    # UPB diff as function of time


    df_monthly.to_pickle(
        d_outpath + configs[d_source][model_name]['combined_filenames']['FE'] + '.pkl'
    )

