from __future__ import division

import sys

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/configs/"
)
from config import configs

import pandas as pd


#
# Import run variables
# d_outpath = sys.argv[1]
# d_source = sys.argv[2]
# model_name = sys.argv[3]
d_outpath = '/Users/peteraltamura/Documents/GitHub/mortgageResearch/output/' \
            'combinedDefaultPred_logit/'
d_source = 'freddie'
model_name = 'comb_logisticRegression'
default_window_months = 3


if __name__ == "__main__":

    #
    # Read in prepped monthly data
    df_monthly = pd.read_pickle(
        d_outpath +
        configs[d_source][model_name]['monthly_filenames']['prepped'] + '.pkl'
    )
    for c in df_monthly.columns:
        if df_monthly[c].dtype not in [object, '<M8[ns]']:
            df_monthly.loc[:, c] = df_monthly[c].astype(float)

    # Read in Origination
    df_origination = pd.read_pickle(
        d_outpath +
        configs[d_source][model_name]['origination_filenames']['prepped'] + '.pkl'
    )
    for c in df_origination.columns:
        if df_origination[c].dtype not in [object, '<M8[ns]']:
            df_origination.loc[:, c] = df_origination[c].astype(float)

    # Combine
    df_origination = pd.merge(
        df_origination,
        df_monthly,
        how='inner',
        on=['loanSeqNumber']
    )
    df_origination = df_origination.loc[
                     df_origination['mthlyRepPeriod'].notnull(), :]

    df_origination.to_pickle(
        d_outpath +
        configs[d_source][model_name]['combined_filenames']['prepped'] + '.pkl'
    )
