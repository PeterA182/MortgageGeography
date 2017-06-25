import sys
import glob

import pandas as pd
import scipy.stats as stats
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

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
        "mortgageResearch/mortgageResearch/configs/"
)
from config import configs


# ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ----
# fRun Vars
sample_run = True
source = 'freddie'
agg_level = 'Month'
make_figs = True
aggs = ['count', 'mean', 'std', 'var', 'min', 'max']

agg_to_title = {
    'count': 'Volume',
    'mean': 'Average',
    'std': 'Standard Deviation',
    'var': 'Variance',
    'min': 'Minimum',
    'max': 'Maximum'
}


if __name__ == "__main__":

    # ---- ---- ----      ---- ---- ----
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
    else:
        raise

    # Agg Level
    df_origination.loc[:, agg_level] = \
        df_origination['firstPaymentDate'].dt.to_period('M').astype(str)
    print "Columns"
    print df_origination.columns

    # Remove "Other Servicers"
    msk = (df_origination['servicerName'] != 'Other servicers')
    df_origination = df_origination.loc[msk, :]

    # ---- ---- ----     ---- ---- ----
    #
    # T-Test for creditScore adverse Selection within MSA
    msas = list(set(df_origination['MSA']))
    servicers = list(set(df_origination['servicerName']))

    perf_vals = {}

    for msa in msas:

        for svcr in servicers:

            msk_svcr = (df_origination['servicerName'] == svcr)
            msk_msa = (df_origination['MSA'] == msa)

            # Define Sample for current MSA and servicer
            sample = df_origination.loc[msk_svcr & msk_msa, :]

            # Define remaining population
            pop = df_origination.loc[(~msk_svcr) & msk_msa, :]

            for aggr in list(set(sample[agg_level])):

                # Get current month or quarter
                sample_ = sample.loc[sample[agg_level] == aggr, :]
                pop_ = pop.loc[pop[agg_level] == aggr, :]

                # Sample now array with single servicer, single MSA, and qtr
                # Population now array with rest servicers, single MSA and qtr

                # Prepare population for weighted average
                pop_ = pop_.groupby(
                    by=[agg_level],
                    as_index=False
                ).agg({'creditScore': [np.mean, 'count']})
                pop_ = compress_columns(pop_)
                wght_avg = (
                    (pop_['count'] * pop_['mean'])/(pop_['count'].sum())
                ).mean()

                # Get calc of t and p_value
                ret = stats.ttest_1samp(a=sample_['creditScore'],
                                        popmean=wght_avg)
                if str(ret[0]) != 'nan' and str(ret[1]) != 'nan':
                    perf_vals.update({(svcr, msa, aggr): ret})




