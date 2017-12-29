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
    d_outpath, d_sample_run, d_source, d_filename
)


sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/Load"
)
from load_loans import load_data, load_origination
from reference import (
    originationFileColList, monthlyFileColList, compress_columns
)

import pandas as pd
import datetime as dt
import numpy as np
import cPickle

if __name__ == "__main__":

    # Read in
    df_monthly = cPickle.load(open(d_outpath + d_filename + '.p', 'rb'))
    for c in df_monthly.columns:
        if df_monthly[c].dtype not in [object, '<M8[ns]']:
            df_monthly.loc[:, c] = df_monthly[c].astype(float)
