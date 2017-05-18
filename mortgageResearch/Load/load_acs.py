import pandas as pd
import numpy as np
import os
import sys


# Pandas IO / Col List from FAQ User Guide
def load_acs(path, filename, nrows='sample',
             data_dict_path=None, date_col_fmt_dict=None,
             col_types=None, remap_columns=True):
    """

    """

    # ---- ---- ----      ---- ---- ----
    # Read in from File
    filename += '.csv' if filename[-4:] != '.csv' else ""
    df_sample = pd.read_csv(
        path + filename,
        nrows=10000 if nrows == 'sample' else nrows
    )

    # Assign column names from data dict
    if data_dict_path:
        col_map = pd.read_excel(
            data_dict_path,
            sheetname='columnNames'
        )
        col_dict = col_map.set_index('Field')['Renamed'].to_dict()
        df_sample.rename(
            columns=col_dict,
            inplace=True
        )

    # Assign Dtypes
    if col_types:
        for k, v in col_types.iteritems():
            df_sample.loc[:, k] = df_sample[k].astype(v)

    # Parse Dates if appropriate
    if date_col_fmt_dict:

        # Iterate over the dict
        for k, v in date_col_fmt_dict.iteritems():
            df_sample.loc[:, k] = pd.to_datetime(df_sample[k], format=v)

    return df_sample


if __name__ == "__main__":

    df = load_acs(
        path='/Users/peteraltamura/Documents/GitHub/mortgageResearch/Data/',
        filename='acs_export.csv',
        data_dict_path='/Users/peteraltamura/Documents/GitHub/'
                       'mortgageResearch/Data/dataDictionary.xlsx'
    )
    print df.head()