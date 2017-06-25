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


sample_run = False
timePeriod = 'Q42015'
source = 'freddie'
min_pmts = 10
p_value_sig = .05

figure_outpath = '/Users/peteraltamura/Documents/GitHub/mortgageResearch/' \
                 'Figures/stationarityTesting/'
if not os.path.exists(figure_outpath):
    os.makedirs(figure_outpath)

# Measure Stats
stats = ['origLTV', 'origCLTV', 'origDTI', 'origUPB',
         'prepayPenaltyMtgFlag', 'creditScore']
window=3


def _read_in(timePeriod, source):
    for path in glob.glob(configs[source]['sample_monthly_dir'] + '*.txt'):
        if timePeriod in path:
            print "Loading File: {}".format(path)
            df_payments = load_data(
                path=path,
                columns=monthlyFileColList,
                nrows=None,
                date_col_fmt_dict={'firstPaymentDate': '%Y%m'},
                error_bad_lines=True
            )
            return df_payments


if __name__ == "__main__":

    # Load in from CSV
    if sample_run:
        df_origination = load_data(
            path=configs[source]['sample_single_dir'] +
                 configs[source]['sample_single_file'],
            columns=originationFileColList,
            date_col_fmt_dict={'firstPaymentDate': '%Y%m'}
        )
        df_origination = df_origination.loc[
                         df_origination['MSA'].notnull(), :]
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
            df_origination = df_origination.loc[
                             df_origination['MSA'].notnull(), :]

    # Results
    results = OrderedDict()

    # Iterate
    for stat in stats:

        # Groupby MSA
        df = df_origination.copy(deep=True)
        df.sort_values(by=['MSA', 'firstPaymentDate'],
                       ascending=True,
                       inplace=True)
        df = df.groupby(
            by=['MSA', 'firstPaymentDate'],
            as_index=False
        ).agg({stat: np.mean})

        msa_all = list(set(df['MSA']))
        for msa in msa_all:
            figure_no = msa_all.index(msa)
            df_msa = df.loc[df['MSA'] == msa, :]

            # Set vars for Dickey Fuller Stationarity Test
            # Null Hypothesis - Unit Root for series
            H_o = True
            # Alt Hypothesis - Stationarity exists
            H_a = False

            # Calculate Rolling Stats of the MSA-level average
            df_msa['{}_rollMean'.format(stat)] = df_msa.groupby(
                ['MSA']
            )[stat].apply(pd.rolling_mean, window=window, min_periods=window)

            df_msa['{}_rollStd'.format(stat)] = df_msa.groupby(
                ['MSA']
            )[stat].apply(pd.rolling_std, window=window, min_periods=window)

            # Dickey Fuller Test - null = Unit Root
            dftest = adfuller(df_msa.set_index('firstPaymentDate')[stat],
                              autolag='AIC')
            dfoutput = pd.Series(dftest[0:4],
                                 index=['Test Statistic', 'p-value',
                                        '#Lags Used',
                                        'Number of Observations Used'])
            testStatistic = dftest[:1]
            p_value = dftest[1:2]
            lagsUsed = dftest[2:3]
            obsUsed = dftest[3:]

            if p_value < p_value_sig:
                H_o = False
                H_a = True

            if H_o:
                print (" "*5) + "{}".format(dt.datetime.now().strftime("%H:%M:%S"))
                print (" "*10) + "MSA: {}".format(str(msa))
                print (" "*10) + "Statistic: {}".format(stat)
                print (" "*15) + "Null Hypothesis remains - Unit Root Exists"
                print (" "*15) + "...creating plot now"
                plt.figure(figure_no, figsize=(10, 10))
                plt.plot(df_msa['firstPaymentDate'], df_msa[stat])
                plt.plot(df_msa['firstPaymentDate'],
                         df_msa['{}_rollMean'.format(stat)],
                         color='red',
                         label='Rolling Mean')
                plt.plot(df_msa['firstPaymentDate'],
                         df_msa['{}_rollStd'.format(stat)],
                         color='green',
                         label='Rolling Std')
                plt.gca().xaxis.set_major_formatter(
                    mdates.DateFormatter('%m_%Y'))
                plt.xticks(rotation='45')
                plt.xlabel("Month-Year")
                plt.title("MSA: {}".format(str(msa)))
                plt.legend(prop={'size': 16})
                plt.savefig(figure_outpath + 'msa_{} stat_{}.png'.format(
                    str(msa), str(stat)
                ))
                plt.close()













