import pandas as pd
import datetime as dt
import numpy as np


def clean_observations(source, table):
    """
    Method for cleaning known-issue columns where observations would fail to
    be formatted correctly as part of load process
    
    PARAMETERS
    ----------
    source: str
        source of the data being cleaned - used to determine which cols need adjustment
    table: DataFrame
        DataFrame object containing the data from given source that needs to be cleaned
    """


    # Source determination for following source-specific process
    if source.lower() in ['fannie', 'freddie']:

        # Remove null creditScore observations
        if 'creditScore' in list(table.columns):
            msk = (table['creditScore'].notnull())
            table = table.loc[msk, :]

            # Assert that creditScore is 2 or 3 chars and all are integers
            table = table.loc[~(table['creditScore'].str.contains(' ', na=False)), :]

            # Ensure all remaining creditScores are possible
            table.loc[:, 'creditScore'] = table['creditScore'].astype(int)
            msk = (table['creditScore'].apply(lambda x: len(str(x))).isin([2, 3]))
            table = table.loc[msk, :]

            # Ensure all chars in remaining creditScore observations are integers
            assert all([type(x) == np.int64 for x in set(table['creditScore'])])

            # Ensure all dates remaining in table are parse-able
            date_cols = ['firstPaymentDate']
            for dd in date_cols:
                try:
                    assert all([len(str(x)) in [5, 6] for x in set(table[dd])])
                except AssertionError as AE:
                    print AE
                    msk = (table[dd].apply(lambda x: len(str(x))).isin([5, 6]))
                    table = table.loc[msk, :]

        # Return
        return table

    elif source.lower() in ['acs']:
        date_cols = []

    else:
        raise Exception(
            "'source' argument is unknown.\n"
            "Please enter known source or update cleanining method")


if __name__ == "__main__":
    pass