import sys
import os
import glob

import pandas as pd
import matplotlib.pyplot as plt

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


# Sample or Full
sample_run = False

# Data Sources
source = 'freddie'
census_source = 'acs'


if __name__ == "__main__":

    if not os.path.exists(configs[source]['figure_dir']):
        os.makedirs(configs[source]['figure_dir'])

    # Determine files to read in
    if sample_run:

        # Needs parser for pandas.io.common.CParserError 16364
        df_origination = load_data(
            path=configs[source]['sample_single_dir'] +
                 configs[source]['sample_single_file'],
            columns=originationFileColList,
            nrows=None,
            date_col_fmt_dict={'firstPaymentDate': '%Y%m'},
            error_bad_lines=True,
            engine='c'
        )

    # Append all otherwise
    elif not sample_run:
        df_origination = pd.DataFrame()
        for path in glob.glob(configs[source]['sample_single_dir'] + '*.txt'):
            print "Loading file: {}".format(path)
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

    # Overall Credit Score Distribution
    plt.figure(1, figsize=(100, 14))
    plt.hist(pd.Series(df_origination['creditScore']),
             alpha=0.7)
    plt.title('Credit Score Distribution')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    fig_all = plt.gcf()
    fig_all.show()
    fig_all.savefig(configs[source]['figure_dir'] + 'All Credit Score Dist.png')

    # Credit Score Distribution by Lender
    plt.figure(2, figsize=(100, 14))
    df_lender = df_origination[['servicerName', 'creditScore']]
    for lender in list(set(df_lender['servicerName'])):
        plt.hist(
            pd.Series(list(df_lender.loc[df_lender['servicerName'] == lender, :]
                           ['creditScore'])),
            label=str(lender),
            alpha=0.7
        )
    plt.legend(prop={'size': 16})
    plt.title('Credit Score Distribution by Lender')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency within Lender')
    fig_lender = plt.gcf()
    fig_lender.show()
    fig_lender.savefig(configs[source]['figure_dir'] + 'Credit Score Dist by Lender.png')

    # Create Lender Trend of Avg Credit Score over time



