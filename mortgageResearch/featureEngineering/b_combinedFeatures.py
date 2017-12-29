import sys
sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/configs/"
)

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/"
)
from dataPrep.a_originationLoadPrep import d_outpath, origination_filename
from dataPrep.a_monthlyLoadPrep import mthly_filename, dmap


# Other
import pandas as pd

from sklearn.feature_selection import VarianceThreshold

# ---- ---- ----
# NOTES
"""
Origination features to be created from origination table only
Default info at the Loan Level (first default month, default flag) to be merged
    to this table first
Monthly features continue to be generated on full monthly table
Monthly-specific features to be merged fully following
"""


# ---- ---- ----
#
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


def create_default_flag(origination, monthly, delinq_days_min, dmap=dmap):
    """
    
    :param df: 
    :return: 
    """

    # Assert min number of days is workable
    assert delinq_days_min in [30, 60, 90, 120]

    # Flag
    days_to_val = {k: v for k, v in dmap.iteritems() if
                   str(delinq_days_min) in v}
    if len(days_to_val.keys()) > 1:
        raise Exception("'currLoanDelinqStatus' values have overlap")
    days_to_val = days_to_val.keys()[0]

    # Mask out
    msk = (
        monthly['currLoanDelinqStatus'] >= float(days_to_val)
    )
    monthly.loc[:,
        'delinquent_threshold_passed_{}'.format(
            str(delinq_days_min)
        )] = 0
    monthly.loc[msk, 'delinquent_threshold_passed_{}'.format(
        str(delinq_days_min)
    )] = 1

    # Create dictionary
    ret_d = monthly.loc[
        monthly['delinquent_threshold_passed_{}'.format(
            str(delinq_days_min)
        )] == 1, :].set_index('loanSeqNumber')[
        'delinquent_threshold_passed_{}'.format(
            str(delinq_days_min)
        )].to_dict()

    origination.loc[:, 'delinquent_threshold_passed_{}'.format(
        str(delinq_days_min)
    )] = 0
    origination.loc[origination['loanSeqNumber'].isin(ret_d.keys()),
                    'delinquent_threshold_passed_{}'.format(
        str(delinq_days_min)
    )] = origination['loanSeqNumber'].map(ret_d)
    return origination


def get_first_default_month(origination, monthly, delinq_days_min):
    """
    
    Answers:
    Is the customer delinquent this month
    
    
    PARAMETERS
    ----------
    df: DataFrame
        loan level dateframe containing (at least) monthly observations and 
        possible origination observations as well
    """

    # Split to those who have defaulted and those who have not
    monthlylt_loans = set(
        monthly.loc[monthly['delinquent_threshold_passed_{}'.format(
            str(delinq_days_min)
        )] == 1, :]['loanSeqNumber']
    )

    # Delinquencies Mask
    delinq_msk = (monthly['loanSeqNumber'].isin(list(monthlylt_loans)))

    # Split out the DataFrames
    monthly_delinq = monthly.loc[delinq_msk, :]

    # Get min default date
    monthly_delinq['firstDefaultMth'] = monthly_delinq.groupby(
        by=['loanSeqNumber'],
        as_index=False
    )['mthlyRepPeriod'].transform(min)
    monthly_delinq = monthly_delinq.loc[
                     monthly_delinq['firstDefaultMth'].notnull(), :]

    # Return dictionary
    ret_d = monthly_delinq.set_index('loanSeqNumber')\
        ['firstDefaultMth'].to_dict()

    # Create first default month
    origination.loc[:, 'first_default_month'] = \
        origination['loanSeqNumber'].map(ret_d)

    return origination


def get_default_month_prior_target(df):
    """
    
    :param df: 
    :return: 
    """

    return df


# ---- ---- ----
#
# Run Variables
homogeneity_thresh = .99
delinq_days_min = 60
default_month_target = False
default_month_prior = True
idx_cols = [
    'loanSeqNumber',
    'delinquent_threshold_passed_{}'.format(delinq_days_min),
    'first_default_month'
]


if __name__ == "__main__":

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    #
    # Read in Monthly
    df_monthly = pd.read_pickle(d_outpath + mthly_filename + '.p')
    for c in df_monthly.columns:
        if df_monthly[c].dtype not in [object, '<M8[ns]']:
            df_monthly.loc[:, c] = df_monthly[c].astype(float)

    # Read in Origination
    df_origination = pd.read_pickle(
        d_outpath + origination_filename + '.p'
    )
    for c in df_origination.columns:
        if df_origination[c].dtype not in [object, '<M8[ns]']:
            df_origination.loc[:, c] = df_origination[c].astype(float)

    # Drop records that are not in both
    match = list(
        set(df_origination.loanSeqNumber) &
        set(df_monthly.loanSeqNumber)
    )
    df_origination = df_origination.loc[
        df_origination['loanSeqNumber'].isin(match), :]
    df_monthly = df_monthly.loc[
        df_monthly['loanSeqNumber'].isin(match), :]

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    #
    # Homogeneity
    df_origination = remove_homogeneity(exog=df_origination,
                                        thresh=homogeneity_thresh)

    #
    # ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ---
    #
    # Flag Defaults above minimum threshold and attach the associated min date
    #   of that occurring
    df_origination = create_default_flag(origination=df_origination,
                                         monthly=df_monthly,
                                         delinq_days_min=delinq_days_min)
    df_origination = get_first_default_month(origination=df_origination,
                                              monthly=df_monthly,
                                              delinq_days_min=delinq_days_min)
    assert sum(df_origination['delinquent_threshold_passed_{}'.format(
        str(delinq_days_min)
    )]) == sum(df_origination['first_default_month'].notnull())

    # creditScore - Standardize via mean()
    df_origination.loc[:, 'creditScore'] -= \
        df_origination['creditScore'].mean()

    # Combine
    len_pre = len(df_origination)
    df_comb = pd.merge(
        df_origination,
        df_monthly,
        how='left',
        on='loanSeqNumber',
        suffixes=('_orig', '_mthly')
    )
    if len(df_comb) > len(df_origination):
        raise Exception("Merge of type \'left\' has added rows")

