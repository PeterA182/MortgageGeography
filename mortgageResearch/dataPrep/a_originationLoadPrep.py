import sys
import os
import cPickle

import pandas as pd
import datetime as dt
import numpy as np

from string import zfill
from sklearn.preprocessing import LabelEncoder

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/Load"
)
from load_loans import load_data, load_origination
from reference import (
    originationFileColList, monthlyFileColList, compress_columns
)

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/configs/"
)
from config import configs

# Run Settings
d_sample_run = True
d_outpath = sys.argv[1]
d_source = sys.argv[2]
model_name = sys.argv[3]
origination_filename = 'loadedPrepped_originationData'
if not os.path.exists(d_outpath):
    os.makedirs(d_outpath)


# Methods
def update_log(action, log={}, export=False):
    """
    
    :param timestamp: 
    :param action: 
    :param log: 
    :return: 
    """

    log.update({dt.datetime.now().strftime('%H:%M:%S'): action})

    if export:
        pd.DataFrame({'Timestamp': log.keys(),
                      'Action': log.values()}).to_csv(d_outpath + 'log.csv')


if __name__ == "__main__":

    # Cols List
    all_col_list = ['loanSeqNumber', 'creditScore']

    # Load Origination
    df_origination = load_origination(
        sample_run=d_sample_run,
        configs=configs,
        source=d_source,
        model_name=model_name,
        col_list=originationFileColList
    )
    update_log(action="Loaded data into DataFrame 'df_origination'")

    # Check for any entirely null column
    if any(sum(df_origination[col].isnull()) == len(df_origination) for col
           in df_origination.columns):
        raise Exception("Column has all null values")
    update_log(action='Checked for all-null columns in data')

    # Formatting
    # First Payment Date
    df_origination.loc[:, 'firstPaymentDate'] = pd.to_datetime(
        df_origination['firstPaymentDate'], format='%Y-%m-%d'
    )
    all_col_list.append('firstPaymentDate')
    update_log(action="Formatted 'firstPaymentDate' accordingly")

    # First Time Buy Flag
    df_origination['firstTimeBuyFlag'].fillna('N', inplace=True)
    le = LabelEncoder()
    le.fit(df_origination['firstTimeBuyFlag'])
    df_origination.loc[:, 'firstTimeBuyFlag'] = \
        le.transform(df_origination['firstTimeBuyFlag'])
    all_col_list.append('firstTimeBuyFlag')
    update_log(action="Encoded 'firstTimeBuyFlag'")

    # Maturity Date
    df_origination.loc[:, 'maturityDate'] = pd.to_datetime(
        df_origination['maturityDate'], format='%Y%m'
    )
    all_col_list.append('maturityDate')
    update_log(action="Formatted 'maturityDate'")

    # MSA
    df_origination.loc[:, 'MSA'] = df_origination['MSA'].astype(str)
    df_origination.loc[:, 'MSA'] = df_origination['MSA'].apply(
        lambda x: x.replace(".0", "")
    )
    all_col_list.append('MSA')
    update_log(action="Formatted 'MSA'")

    # TODO Mortgage Insurance Pct

    # numberUnits
    all_col_list.append('numberUnits')
    update_log(action="'numberUnits' needs no formatting")

    # occStatus
    for s in set(df_origination['occStatus']):
        df_origination.loc[:, 'occStatus_{}'.format(s)] = 0
        df_origination.loc[
            df_origination['occStatus'] == s, 'occStatus_{}'.format(s)] = 1
        all_col_list.append('occStatus_{}'.format(s))
    update_log(action="indicator columns created for 'occStatus'")

    # origCLTV
    df_origination.loc[:, 'origCLTV'] = \
        df_origination['origCLTV'].astype(float)
    df_origination.fillna(np.nanmedian(df_origination.origCLTV), inplace=True)
    all_col_list.append('origCLTV')
    update_log(action="fillna() completed for 'origCLTV'")

    # origDTI
    df_origination.loc[:, 'origDTI'] = \
        df_origination['origDTI'].astype(str)
    df_origination.loc[:, 'origDTI'] = df_origination['origDTI'].apply(
        lambda x: x.replace(' ', '0')
    )
    df_origination.loc[:, 'origDTI'] = df_origination['origDTI'].astype(float)
    df_origination.fillna(np.nanmedian(df_origination.origDTI))
    all_col_list.append('origDTI')
    update_log(action="fillna() completed for 'origDTI'")

    # propertyState
    for s in set(df_origination['propertyState']):
        df_origination.loc[:, 'propertyState_{}'.format(s)] = 0
        df_origination.loc[
            df_origination['propertyState'] == s,
            'propertyState_{}'.format(s)] = 1
        all_col_list.append('propertyState_{}'.format(s))
    update_log(action="indicator columns created for 'propertyState'")

    # propertyType
    for t in set(df_origination.propertyType):
        df_origination.loc[:, 'propertyType_{}'.format(t)] = 0
        df_origination.loc[df_origination['propertyType'] == t,
            'propertyType_{}'.format(t)] = 1
        all_col_list.append('propertyType_{}'.format(t))
    update_log(action="indicator variables created for 'propertyType'")

    # Postal Code -- Check
    df_origination.loc[:, 'postalCode'] = \
        df_origination['postalCode'].astype(str).apply(
            lambda x: x.replace(".0", "")
        )
    df_origination.loc[:, 'postalCode'] = \
        df_origination['postalCode'].apply(
            lambda x: zfill(x, width=5)
        )
    all_col_list.append('postalCode')

    # for pc in set(df_origination.postalCode):
    #     df_origination.loc[:, 'postalCode_{}'.format(pc)] = 0
    #     df_origination.loc[
    #         df_origination['postalCode'] == pc,
    #         'postalCode_{}'.format(pc)] = 1
    #     all_col_list.append(pc)
    update_log(action="indicator variables created for 'postalCode'")

    # Loan Purpose
    for lp in set(df_origination.loanPurpose):
        df_origination.loc[:, 'loanPurpose_{}'.format(lp)] = 0
        df_origination.loc[df_origination['loanPurpose'] == lp,
            'loanPurpose_{}'.format(lp)] = 1
        all_col_list.append('loanPurpose_{}'.format(lp))
    update_log(action="indicator variables created for 'loanPurpose'")

    # Original Loan Term
    assert sum(df_origination['origLoanTerm'].isnull()) == 0
    all_col_list.append('origLoanTerm')
    update_log(action='origLoanTerm added successfully - no formatting needed')

    # Borrowers
    assert sum(df_origination['borrowers'].isnull()) == 0
    all_col_list.append('borrowers')
    update_log(action="'borrowers' added successfully - no formatting needed")

    # Super Conforming Flag
    df_origination.loc[df_origination['superConformingFlag'] != 'Y',
        'superConformingFlag'] = 'N'
    df_origination.loc[:, 'superConformingFlag'] = \
        df_origination['superConformingFlag'].map({'Y': 1, 'N': 0})
    all_col_list.append('superConformingFlag')
    update_log(action="Replaced non-'Y' 'superConformingFlag' values with 'N'")

    # Mtg Insurance Pct
    df_origination.loc[:, 'mtgInsurancePct'] = \
        df_origination['mtgInsurancePct'].astype(str).apply(
            lambda x: x.replace('   ', '0')
        )
    df_origination.loc[:, 'mtgInsurancePct'] = \
        df_origination['mtgInsurancePct'].apply(
            lambda x: x.replace('000', '0')
        )
    df_origination.loc[:, 'mtgInsurancePct'] = \
        df_origination['mtgInsurancePct'].astype(float)
    all_col_list.append('mtgInsurancePct')
    update_log(action="Formatted 'mtgInsurancePct' field")

    # origUPB
    all_col_list.append('origUPB')
    update_log(action="'Formatted 'origUPB' field")

    # origLTV
    all_col_list.append('origLTV')
    update_log(action="Formatted 'origLTV' field")

    # origInterestRate
    all_col_list.append('origInterestRate')
    update_log(action="Formatted 'orgInterestRate' field")

    # channel
    for ch in set(df_origination['channel']):
        df_origination.loc[:, 'channel_{}'.format(ch)] = 0
        df_origination.loc[df_origination['channel'] == ch,
                           'channel_{}'.format(ch)] = 1
        all_col_list.append('channel_{}'.format(ch))
    update_log(action="Indicator variables created for 'channel'")

    #prepayPenaltymtgFlag
    df_origination.loc[df_origination['prepayPenaltyMtgFlag'] == 78.0,
                       'prepayPenaltyMtgFlag'] = 1
    df_origination.loc[df_origination['prepayPenaltyMtgFlag'] != 1,
                       'prepayPenaltyMtgFlag'] = 0
    df_origination.loc[:, 'prepayPenaltyMtgFlag'] = \
        df_origination['prepayPenaltyMtgFlag'].astype(float)
    all_col_list.append('prepayPenaltyMtgFlag')
    update_log(
        action="Formatted 'prepayPenaltyMtgFlag' mapping '78.0' to 'Yes'"
    )

    # productType
    pt_map = {
        'CO': 'Condo',
        'LH': 'Leasehold',
        'PU': 'PUD',
        'MH': 'Manufactured Housing',
        'SF': '1_4_feeSimple',
        'CP': 'co_op',
        ' ': 'Unknown'
    }
    for k, v in pt_map.iteritems():
        df_origination.loc[:, 'productType_{}'.format(v)] = 0
        df_origination.loc[
            df_origination['productType'] == k, 'productType_{}'.format(v)] = 1
        all_col_list.append('productType_{}'.format(v))
    update_log(action="Indicator variables created for 'productType'")

    # Send to CSV
    df_origination = df_origination.loc[:, all_col_list]
    df_origination.to_csv(
        d_outpath +
        configs[d_source][model_name]['origination_filenames']['prepped'] +
        '.csv'
    )
    df_origination.to_pickle(
        d_outpath +
        configs[d_source][model_name]['origination_filenames']['prepped'] +
        '.pkl'
    )
    update_log(action='Finished',
               export=True)


