import sys
import os
import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import datetime as dt
from string import zfill

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
    print "Columns"
    print df_origination.columns
    # Get list of borrowers
    print "Borrowers"
    print set(df_origination['servicerName'])
    originators = list(set(df_origination.servicerName))

    # Iterate over all parties servicing loans in time period
    org_performance = {}
    aggs = ['count', 'mean', 'std', 'var']
    for orig in originators:

        # Current
        df_org = df_origination.loc[df_origination['servicerName'] == orig, :]

        # Aggregate up by month
        df_org.loc[:, agg_level] = \
            df_org['firstPaymentDate'].dt.month
        df_org = df_org.groupby(
            by=['servicerName', agg_level],
            as_index=False
        ).agg({'creditScore': aggs})
        df_org.rename(
            columns={
                'count': 'creditScore_count',
                'mean': 'creditScore_mean',
                'std': 'creditScore_std',
                'var': 'creditScore_var'
            },
            inplace=True
        )

        # Compress MultiIndex Columns
        df_org = compress_columns(df_org)
        org_performance.update({orig: df_org})

    # Figure construction for descriptive stats table
    if make_figs:

        # For each item in saved dict
        for orig, perf in org_performance.iteritems():
            print "Now constructing graphs for: {}".format(orig.title())
            plt.figure(1+list(org_performance.keys()).index(orig),
                       figsize=(10, 10))
            months = sorted(set(perf[agg_level]))
            for mtr in aggs:
                print (" "*10) + "Now graphing: Credit Score {}".format(
                    mtr.title()
                )
                plt.subplot(2, 2, 1+aggs.index(mtr))
                plt.plot(months, perf['creditScore_{}'.format(mtr)],
                         label='{}'.format(mtr.title()))
                plt.title("{}".format(mtr))
                plt.xticks(rotation='horizontal')
                plt.xlabel("{}".format(agg_level))
                plt.ylabel("Credit Score {}".format(mtr.title()))
            plt.savefig(configs[source]['figure_dir'] +
                        "{} Descriptive Charts.png".format(orig.title()))
            plt.close()














