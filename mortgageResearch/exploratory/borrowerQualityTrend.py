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
sample_run = False
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
        df_origination['firstPaymentDate']
    print "Columns"
    print df_origination.columns

    # Get list of borrowers
    print "Borrowers"
    print set(df_origination['servicerName'])
    originators = list(set(df_origination.servicerName))

    # Iterate over all parties servicing loans in time period

    # Aggregate up by month

    df_desc_stats = df_origination.groupby(
        by=['servicerName', agg_level],
        as_index=False
    ).agg({'creditScore': aggs})
    df_desc_stats.rename(
        columns={
            'count': 'creditScore_count',
            'mean': 'creditScore_mean',
            'std': 'creditScore_std',
            'var': 'creditScore_var',
            'min': 'creditScore_min',
            'max': 'creditScore_max'
        },
        inplace=True
    )

    # Compress MultiIndex Columns
    df_desc_stats = compress_columns(df_desc_stats)

    # Figure construction for descriptive stats table
    if make_figs:

        # For each item in saved dict
        servicers = list(set(df_desc_stats['servicerName']))
        for orig in servicers:
            perf = df_desc_stats.loc[df_desc_stats['servicerName'] == orig, :]

            # Plot the figures
            print "Now constructing graphs for: {}".format(orig.title())
            plt.figure(1+servicers.index(orig),
                       figsize=(40, 20))

            # Subplot each descriptive statistic
            for mtr in aggs:
                print (" "*10) + "Now graphing: Credit Score {}".format(
                    mtr.title()
                )
                plt.subplot(2, 3, 1+aggs.index(mtr))
                plt.plot(perf[agg_level], perf['creditScore_{}'.format(mtr)],
                         label='{}'.format(mtr.title()))
                plt.gca().xaxis.set_major_formatter(
                    mdates.DateFormatter('%m-%Y'))
                plt.title("{}".format(agg_to_title[mtr]))
                plt.xticks(rotation='45')
                plt.xlabel("{}".format(agg_level))
                plt.ylabel("Credit Score {}".format(mtr.title()))

            # Save and Close
            plt.savefig(configs[source]['figure_dir'] +
                        "{} Descriptive Charts.png".format(orig.title()))
            plt.close()

    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    # Measure of Spread (Monthly)
    df_org = df_desc_stats.copy(deep=True)
    for pctl in [25, 75]:
        df_ = df_origination.groupby(
            by=['servicerName', agg_level],
            as_index=False
        ).agg({'creditScore': lambda x: np.percentile(x, pctl)})
        df_.rename(
            columns={'creditScore': 'pctl_{}'.format(str(pctl))},
            inplace=True
        )
        pre_len = len(df_org)
        df_org = pd.merge(
            df_org,
            df_,
            how='left',
            on=['servicerName', agg_level]
        )
        if pre_len != len(df_org):
            raise Exception(
                "Merge of type \'left\' has added rows unexpectedly"
            )

    # Coefficient of Variation
    df_org.loc[:, 'Coef_var'] = (
        (df_org['pctl_75'] - df_org['pctl_25']) / df_org['creditScore_mean']
    )

    # Separate graphs of spread (into subdirectory for now)
    if make_figs:

        # For each item in saved dict
        servicers = list(set(df_desc_stats['servicerName']))
        for orig in servicers:
            perf = df_org.loc[df_org['servicerName'] == orig, :]

            # Plot the figures
            print "Now constructing graphs for: {}".format(orig.title())
            print (" " * 10) + \
                  "Now graphing: Credit Score Coefficient of Variation"
            plt.figure(1 + servicers.index(orig),
                       figsize=(10, 10))
            plt.plot(perf[agg_level], perf['Coef_var'],
                     label='{}'.format('Coefficient of Variation'.title()))
            plt.gca().xaxis.set_major_formatter(
                mdates.DateFormatter('%m-%Y'))
            plt.title("Coefficient of Variation")
            plt.xticks(rotation='45')
            plt.xlabel("{}".format(agg_level))
            plt.ylabel("Credit Score Coefficient of Variation")

            # Set y axis
            axes = plt.gca()
            axes.set_ylim([0, 0.5])

            # Save and Close
            coef_var_path = configs[source]['figure_dir'] + "SpreadMeasure/"
            if not os.path.exists(coef_var_path):
                os.makedirs(coef_var_path)
            plt.savefig(coef_var_path +
                        "{} Spread of Credit Score_CoefficientofVariation.png".format(orig.title()))
            plt.close()






    #       Lender overall spread
    #       Lender by MSA
    #       Lender safest MSA
    #       Lender riskiest MSA













