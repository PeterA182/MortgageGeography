import os
import sys

import datetime as dt
import pandas as pd
import numpy as np

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/Load"
)
from clean_table import clean_observations


# Pandas IO / Col List from FAQ User Guide
def load_data(path, nrows=None, columns=None, years=None, quarters=None,
              date_col_fmt_dict=None, col_types=None, error_bad_lines=False,
              engine='c'):
    """

    """
    qtr_convert = lambda x: pd.Period(x, freq='Q')

    # ---- ---- ----      ---- ---- ----
    # Read in from File
    if engine == 'c':
        df_data = pd.read_csv(
            path,
            nrows=nrows,
            error_bad_lines=error_bad_lines,
            delimiter='|'
        )
    elif engine == 'python':
        df_data = pd.read_csv(
            path,
            nrows=nrows,
            error_bad_lines=error_bad_lines,
            delimiter='|',
            engine=engine
        )

    # Assign columns if not in file already
    if columns:
        df_data.columns = columns
        print (" "*10) + "{} :: Columns assigned".format(
            dt.datetime.now().strftime("%H:%M:%S")
        )

    # Clean all observations found
    df_data = clean_observations(source='freddie', table=df_data)
    print (" "*10) + "{} :: Observations cleaned".format(
        dt.datetime.now().strftime("%H:%M:%S")
    )

    # Numeric creditScore
    if 'creditScore' in df_data.columns:
        df_data.loc[:, 'creditScore'] = \
            pd.to_numeric(df_data['creditScore'])
        print (" "*10) + "{} :: creditScore made numeric".format(
            dt.datetime.now().strftime("%H:%M:%S")
        )

    # Assign Dtypes
    if col_types:
        for k, v in col_types.iteritems():
            df_data.loc[:, k] = df_data[k].astype(v)
        print (" "*10) + "{} :: Types assigned successfully".format(
            dt.datetime.now().strftime("%H:%M:%S")
        )

    # Parse Dates if appropriate
    if date_col_fmt_dict:

        # Iterate over the dict
        for k, v in date_col_fmt_dict.iteritems():
            if k in df_data.columns:
                df_data = df_data.loc[(df_data[k].apply(lambda x: len(str(x))).isin([5, 6])), :]
                df_data.loc[:, k] = pd.to_datetime(df_data[k], format=v)
        print (" "*10) + "{} :: Date Columns formatted successfully".format(
            dt.datetime.now().strftime("%H:%M:%S")
        )

    # Return
    return df_data


if __name__ == "__main__":
    pass





