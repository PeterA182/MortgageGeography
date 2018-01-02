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
from load_loans import load_data, load_origination, load_monthly
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
dmap = {
    '0': '0_29days_delinquent',
    '1': '30_59days delinquent',
    '2': '60_89days delinquent',
    '3': '90_119days delinquent',
    '4': '120_149days delinquent',
    '5': '150_179days_delinquent',
    '6': '180_209days_delinquent',
    '7': '210_plus'
}


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
    all_col_list = ['loanSeqNumber']
    all_nulls = [
        'modificationFlag',
        'mtgInsuranceRecovery',
        'netSalesProceeds',
        'nonMtgInsuranceRecovery',
        'expenses',
        'legalCosts',
        'maintenancePreservtnCosts',
        'taxesInsuranceOwed',
        'miscExpenses',
        'actualLossCalculation'
    ]

    # Load Origination
    df_monthly = load_monthly(
        sample_run=d_sample_run,
        configs=configs,
        source=d_source,
        model_name=model_name,
        col_list=monthlyFileColList,
        most_recent=True
    )
    update_log(action="Loaded data into DataFrame 'df_monthly'")

    # Check for entirely null columns
    # if any(sum(df_monthly[col].isnull()) == len(df_monthly) for col
    #        in df_monthly.columns):
    #     raise Exception("Column has all null values")
    # update_log(action='Checked for all-null columns in data')

    # Format Monthly Reporting Period
    df_monthly.loc[:, 'mthlyRepPeriod'] = \
        pd.to_datetime(df_monthly['mthlyRepPeriod'], format='%Y%m')
    df_monthly = df_monthly.loc[df_monthly['mthlyRepPeriod'].notnull(), :]
    all_col_list.append('mthlyRepPeriod')
    update_log(action="Formatted 'mthlyRepPeriod' to month and year datetime "
                      "objects")

    # origUPB
    df_monthly.loc[:, 'currUPB'] = df_monthly['currUPB'].astype(float)
    df_monthly.loc[:, 'currUPB'].fillna(np.nanmedian(df_monthly['currUPB']),
                                        inplace=True)
    all_col_list.append('currUPB')
    update_log(action="'Formatted 'origUPB' field")

    # currLoanDelinqStatus
    df_monthly.loc[:, 'currLoanDelinqStatus'] = \
        df_monthly['currLoanDelinqStatus'].astype(str).apply(
            lambda x: x.replace('R', '7')
        )
    df_monthly.loc[:, 'currLoanDelinqStatus'] = \
        df_monthly['currLoanDelinqStatus'].astype(float)
    df_monthly.loc[:, 'currLoanDelinqStatus'].fillna(0, inplace=True)
    all_col_list.append('currLoanDelinqStatus')
    update_log(action="Formatted 'currLoanDelinqStatus'")

    # loanAge
    df_monthly.loc[:, 'loanAge'] = df_monthly['loanAge'].astype(float)

    # remainingMonthsLegMaturity
    df_monthly.loc[:, 'remainingMonthsLegMaturity'] = \
        df_monthly['remainingMonthsLegMaturity'].astype(float)

    # repurchaseFlags
    rmap = {
        'Y': 1,
        'N': 0
    }
    df_monthly.loc[:, 'repurchaseFlag'] = df_monthly['repurchaseFlag'].map(rmap)
    df_monthly.loc[df_monthly['repurchaseFlag'].isnull(), 'repurchaseFlag'] = 0
    all_col_list.append('repurchaseFlag')
    update_log(action="Formatted and mapped 'repurchaseFlag'")

    # modificationFlag
    df_monthly.loc[:, 'modificationFlag'] = \
        df_monthly['modificationFlag'].astype(str)
    df_monthly.loc[df_monthly['modificationFlag'] != 'Y',
                   'modificationFlag'] = 'N'
    df_monthly.loc[df_monthly['modificationFlag'].isnull(),
                   'modificationFlag'] = 'N'
    df_monthly.loc[:, 'modificationFlag'] = \
        df_monthly['modificationFlag'].map({'Y': 1, 'N': 0})
    all_col_list.append('modificationFlag')
    update_log(action="Formatted 'modificationFlag'")

    # Zero Balance Code
    df_monthly.loc[:, 'zeroBalanceCode'] = \
        df_monthly['zeroBalanceCode'].astype(str).apply(
            lambda x: zfill(x, width=2)
        )
    df_monthly.loc[:, 'zeroBalanceCode'] = df_monthly['zeroBalanceCode'].map({
        '00': 'unknown',
        '01': 'prepaid_matured',
        '03': 'foreclosure_alternative',
        '06': 'repurchase',
        '09': 'REO_disposition'
    })
    df_monthly.loc[df_monthly['zeroBalanceCode'].isnull(),
        'zeroBalanceCode'] = '00'
    for s in list(set(df_monthly['zeroBalanceCode'])):
        df_monthly.loc[:, 'zeroBalCode_{}'.format(s)] = 0
        df_monthly.loc[df_monthly['zeroBalanceCode'] == s,
            'zeroBalCode_{}'.format(s)] = 1
        all_col_list.append('zeroBalCode_{}'.format(s))
    update_log("Indicator variables created for zeroBalanceCode")

    # zeroBalanceEffectiveDate
    df_monthly.loc[:, 'zeroBalanceEffectiveDate'] = \
        df_monthly['zeroBalanceEffectiveDate'].astype(str)
    df_monthly.loc[:, 'zeroBalanceEffectiveDate'] = \
        df_monthly['zeroBalanceEffectiveDate'].apply(
        lambda x: x.replace('.0', '')
    )
    df_monthly.loc[:, 'zeroBalanceEffectiveDate'] = \
        pd.to_datetime(df_monthly['zeroBalanceEffectiveDate'],
                       format='%Y%m')
    all_col_list.append('zeroBalanceEffectiveDate')
    update_log(action="Formatted 'zeroBalanceEffectiveDate")

    # currentInterestRate
    df_monthly.loc[:, 'currInterestRate'] = \
        df_monthly['currInterestRate'].astype(float)
    df_monthly['currInterestRate'].fillna(
        np.nanmedian(df_monthly['currInterestRate']),
        inplace=True
    )
    all_col_list.append('currInterestRate')
    update_log(action="Formatted 'currInterestRate'")

    # Current Deferred UPB
    df_monthly.loc[:, 'currDeferredUPB'] = \
        df_monthly['currDeferredUPB'].astype(float)
    df_monthly['currDeferredUPB'].fillna(
        np.nanmedian(df_monthly.currDeferredUPB),
        inplace=True
    )
    all_col_list.append(
        'currDeferredUPB'
    )
    update_log(action="Formatted 'currDeferredUPB'")

    # Due Date Last Paid Installment
    df_monthly['dueDateLastPaidInstallment'] = \
        df_monthly['dueDateLastPaidInstallment'].astype(str).apply(
            lambda x: x.replace('.0', '')
        )
    df_monthly.loc[:, 'dueDateLastPaidInstallment'] = \
        pd.to_datetime(df_monthly['dueDateLastPaidInstallment'],
                       format='%Y%m')
    all_col_list.append('dueDateLastPaidInstallment')
    update_log(action="Formatted 'dueDateLastPaidInstallment'")

    # mtgInsuranceRecovery
    df_monthly.loc[:, 'mtgInsuranceRecovery'] = \
        df_monthly['mtgInsuranceRecovery'].astype(float)
    df_monthly['mtgInsuranceRecovery'].fillna(0, inplace=True)
    all_col_list.append('mtgInsuranceRecovery')
    update_log(action="Formatted 'mtgInsuranceRecovery'")

    # netSalesProceeds
    df_monthly.loc[:, 'netSalesProceeds'] = \
        df_monthly['netSalesProceeds'].astype(str)
    # Assign out 'covered'
    df_monthly.loc[:, 'netSalesProceeds_covered'] = 0
    df_monthly.loc[df_monthly['netSalesProceeds'] == 'C',
                   'netSalesProceeds_covered'] = 1
    all_col_list.append('netSalesProceeds_covered')

    # Assign out 'unknown'
    df_monthly.loc[:, 'netSalesProceeds_unknown'] = 0
    df_monthly.loc[df_monthly['netSalesProceeds_unknown'].astype(str) == 'U',
                   'netSalesProceeds_unknown'] = 1
    all_col_list.append('netSalesProceeds_unknown')

    # nonMtgInsuranceRecovery
    df_monthly.loc[:, 'nonMtgInsuranceRecovery'] = \
        df_monthly['nonMtgInsuranceRecovery'].astype(float)
    df_monthly['nonMtgInsuranceRecovery'].fillna(0, inplace=True)
    all_col_list.append('nonMtgInsuranceRecovery')
    update_log(action="Formatted 'nonMtgInsuranceRecovery'")

    # Expenses
    df_monthly.loc[:, 'expenses'] = \
        df_monthly['expenses'].astype(float)
    df_monthly['expenses'].fillna(0, inplace=True)
    all_col_list.append('expenses')
    update_log(action="Formatted 'expenses'")

    # Legal Costs
    df_monthly.loc[:, 'legalCosts'] = \
        df_monthly['legalCosts'].astype(float)
    df_monthly['legalCosts'].fillna(0, inplace=True)
    all_col_list.append('legalCosts')
    update_log(action="Formatted 'legalCosts'")

    # maintenancePreservtnCosts
    df_monthly.loc[:, 'maintenancePreservtnCosts'] = \
        df_monthly['maintenancePreservtnCosts'].astype(float)
    df_monthly['maintenancePreservtnCosts'].fillna(0, inplace=True)
    all_col_list.append('maintenancePreservtnCosts')
    update_log(action="Formatted 'maintenancePreservtnCosts'")

    # taxesInsuranceOwed
    df_monthly.loc[:, 'taxesInsuranceOwed'] = \
        df_monthly['taxesInsuranceOwed'].astype(float)
    df_monthly['taxesInsuranceOwed'].fillna(0, inplace=True)
    all_col_list.append('taxesInsuranceOwed')
    update_log(action="Formatted 'taxesInsuranceOwed'")

    # miscExpenses
    df_monthly.loc[:, 'miscExpenses'] = \
        df_monthly['miscExpenses'].astype(float)
    df_monthly['miscExpenses'].fillna(0, inplace=True)
    all_col_list.append('miscExpenses')
    update_log(action="Formatted 'miscExpenses'")

    # actualLossCalculation
    df_monthly.loc[:, 'actualLossCalculation'] = \
        df_monthly['actualLossCalculation'].astype(float)
    df_monthly['actualLossCalculation'].fillna(0, inplace=True)
    all_col_list.append('actualLossCalculation')
    update_log(action="Formatted 'actualLossCalculation'")

    # modificationCost
    df_monthly.loc[:, 'modificationCost'] = \
        df_monthly['modificationCost'].astype(float)
    df_monthly['modificationCost'].fillna(0, inplace=True)
    all_col_list.append('modificationCost')
    update_log(action="Formatted 'modificationCost'")

    df_monthly.drop(labels=['Unknown'],
                    axis=1,
                    inplace=True)

    # Subset to appropriate columns
    df_monthly = df_monthly.loc[:, all_col_list]

    # Final Check
    for c in [x for x in df_monthly.columns]:
        if sum(df_monthly[c].isnull()) > 0:
            print c
            print "     {} null observations".format(
                str(sum(df_monthly[c].isnull()))
            )
    # Send out CSV and Pickle
    df_monthly.to_csv(
        d_outpath +
        configs[d_source][model_name]['monthly_filenames']['prepped'] + '.csv')
    df_monthly.to_pickle(
        d_outpath +
        configs[d_source][model_name]['monthly_filenames']['prepped'] + '.pkl'
    )
    update_log(action='Finished',
               export=True)

