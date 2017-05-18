import sys
import os

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

# Define out Variables
figure_dir = '/Users/peteraltamura/Documents/GitHub/mortgageResearch/Figs/'
sample_dir = "/Users/peteraltamura/Documents/GitHub/mortgageResearch/Data/" \
             "historical_data1_Q12016/"
sample_file = "historicalData1_Q12016.csv"
sample_file_monthly = "historicalDataMonthly1_Q12016.csv"



if __name__ == "__main__":

    if not os.path.exists(figure_dir):
        os.makedirs(figure_dir)

    df_origination = load_data(
        path=sample_dir,
        filename=sample_file,
        columns=originationFileColList,
        date_col_fmt_dict={'firstPaymentDate': '%Y%m'}
    )
    print df_origination.columns

    # Overall Credit Score Distribution
    plt.figure(1, figsize=(12, 14))
    plt.hist(pd.Series(df_origination['creditScore']))
    plt.title('Credit Score Distribution')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency')
    fig_all = plt.gcf()
    fig_all.show()
    fig_all.savefig(figure_dir + 'All Credit Score Dist.png')

    # Credit Score Distribution by Lender
    plt.figure(2, figsize=(12, 14))
    df_lender = df_origination[['servicerName', 'creditScore']]
    for lender in list(set(df_lender['servicerName'])):
        plt.hist(pd.Series(list(df_lender.loc[df_lender['servicerName'] == lender, :]['creditScore'])),
                 label=str(lender),
                 alpha=0.7)
    plt.legend(prop={'size': 16})
    plt.title('Credit Score Distribution by Lender')
    plt.xlabel('Credit Score')
    plt.ylabel('Frequency within Lender')
    fig_lender = plt.gcf()
    fig_lender.show()
    fig_lender.savefig(figure_dir + 'Credit Score Dist by Lender.png')


