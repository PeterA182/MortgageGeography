import pandas as pd
import numpy as np
import os
import sys


# Pandas IO / Col List from FAQ User Guide
def load_data(path, filename, nrows='sample', columns=None,
               date_col_fmt_dict=None, col_types=None):
    """

    """

    # ---- ---- ----      ---- ---- ----
    # Read in from File
    filename += '.csv' if filename[-4:] != '.csv' else ""
    df_sample = pd.read_csv(
        path + filename,
        nrows=10000 if nrows == 'sample' else nrows
    )

    # Assign columns if not in file already
    if columns:
        df_sample.columns = columns

    # Assign Dtypes
    if col_types:
        for k, v in col_types.iteritems():
            df_sample.loc[:, k] = df_sample[k].astype(v)

    # Parse Dates if appropriate
    if date_col_fmt_dict:

        # Iterate over the dict
        for k, v in date_col_fmt_dict.iteritems():
            df_sample.loc[:, k] = pd.to_datetime(df_sample[k], format=v)

    # Return
    return df_sample


if __name__ == "__main__":
    pass





