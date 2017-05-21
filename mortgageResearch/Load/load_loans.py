import pandas as pd
import numpy as np
import os
import sys

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/Load"
)
from clean_table import clean_observations


# Pandas IO / Col List from FAQ User Guide
def load_data(path, filename, nrows=None, columns=None,
               date_col_fmt_dict=None, col_types=None, error_bad_lines=False, engine='c'):
    """

    """

    # ---- ---- ----      ---- ---- ----
    # Read in from File
    filename += '.txt' if filename[-4:] != '.txt' else ""
    if engine == 'c':
        df_sample = pd.read_csv(
            path + filename,
            nrows=nrows,
            error_bad_lines=error_bad_lines,
            delimiter='|'
        )
    elif engine == 'python':
        df_sample = pd.read_csv(
            path + filename,
            nrows=nrows,
            error_bad_lines=error_bad_lines,
            delimiter='|',
            engine=engine
        )

    # Assign columns if not in file already
    if columns:
        df_sample.columns = columns

    # Clean all observations found
    df_sample = clean_observations(source='fannie', table=df_sample)

    # Assign Dtypes
    if col_types:
        for k, v in col_types.iteritems():
            df_sample.loc[:, k] = df_sample[k].astype(v)

    # Parse Dates if appropriate
    if date_col_fmt_dict:

        # Iterate over the dict
        for k, v in date_col_fmt_dict.iteritems():
            df_sample = df_sample.loc[(df_sample[k].apply(lambda x: len(str(x))).isin([5, 6])), :]
            df_sample.loc[:, k] = pd.to_datetime(df_sample[k], format=v)

    # Return
    return df_sample


if __name__ == "__main__":
    pass





