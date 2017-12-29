import sys

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/configs/"
)
from config import configs

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/"
)
from dataPrep.a_originationLoadPrep import (
    d_outpath, d_filename
)


sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/Load"
)

import pandas as pd
import datetime as dt
import numpy as np
import cPickle

from sklearn.feature_selection import VarianceThreshold

# Methods
def VarianceThreshold_selector(exog):
    """
    Implementation of VarianceThreshold from sklearn with get_support() for 
    retaining proper column names for non-dropped columns
    
    
    PARAMETERS
    ----------
    data: DataFrame
        contains exogenous variables to be measured for homogeneity
    """

    # Save Date columns
    excl_cols = [x for x in exog.columns if 'Date' in x]
    excl_cols.extend([x for x in exog.columns if (
        (x not in excl_cols) and
        (exog[x].dtype in [object, '<M8[ns]']))
    ])

    # Initial columns set
    exog_columns = [x for x in exog.columns if x not in excl_cols]

    # Instantiate selector and fit_transform to data
    selector = VarianceThreshold(homogeneity_thresh)
    df = selector.fit_transform(exog[exog_columns])
    labels = []

    j = 0
    for x in selector.get_support(indices=False):
        if x:
            labels.append(exog_columns[j])
        j += 1

    return pd.DataFrame(data=df,
                        columns=labels)


def remove_homogeneity(exog, thresh):
    """
    
    :param exog: 
    :param thresh: 
    :return: 
    """

    # Set up dropped columns list
    dropped = []

    # Get ValueCounts out and turn in DataFrame()
    for c in exog.columns:

        vc = pd.DataFrame(exog[c].value_counts())

        # Reset index and assign column names
        vc.reset_index(inplace=True)
        vc.columns = ['val', 'occ']
        vc['occ'] /= vc['occ'].sum()
        if sum(vc['occ'] > thresh) > 0:
            print "Column {}: dropped".format(c)
            dropped.append(c)

    # Apply dropped columns
    return exog.loc[:, [x for x in exog.columns if x not in dropped]]


# Run Variables
homogeneity_thresh = .99


if __name__ == "__main__":

    # Read in
    df_origination = cPickle.load(open(d_outpath + d_filename + '.p', 'rb'))
    for c in df_origination.columns:
        if df_origination[c].dtype not in [object, '<M8[ns]']:
            df_origination.loc[:, c] = df_origination[c].astype(float)

    # creditScore - Standardize via mean()
    df_origination.loc[:, 'creditScore'] -= df_origination['creditScore'].mean()

    # Variance Thresholds
    df_origination = VarianceThreshold_selector(
        exog=df_origination
    )

    df_origination = remove_homogeneity(exog=df_origination,
                                        thresh=homogeneity_thresh)







