# Column Lists for Datasets (Freddie)

originationFileColList = [
    'creditScore', 'firstPaymentDate', 'firstTimeBuyFlag', 'maturityDate',
    'MSA', 'mtgInsurancePct', 'numberUnits', 'occStatus', 'origCLTV',
    'origDTI', 'origUPB', 'origLTV', 'origInterestRate', 'channel',
    'prepayPenaltyMtgFlag', 'productType', 'propertyState', 'propertyType',
    'postalCode', 'loanSeqNumber', 'loanPurpose', 'origLoanTerm', 'borrowers',
    'sellerName', 'servicerName', 'superConformingFlag'
]
monthlyFileColList = [
    'loanSeqNumber', 'mthlyRepPeriod', 'currUPB', 'currLoanDelinqStatus',
    'loanAge', 'remainingMonthsLegMaturity', 'repurchaseFlag',
    'modificationFlag', 'zeroBalanceCode', 'zeroBalanceEffectiveDate',
    'currInterestRate', 'currDeferredUPB', 'dueDateLastPaidInstallment',
    'mtgInsuranceRecovery', 'netSalesProceeds', 'nonMtgInsuranceRecovery',
    'expenses', 'legalCosts', 'maintenancePreservtnCosts',
    'taxesInsuranceOwed', 'miscExpenses', 'actualLossCalculation',
    'modificationCost', 'Unknown'
]

def compress_columns(table, reset_index=False):

    # Index
    if reset_index:
        table.reset_index(inplace=True)

    get_col = lambda x: x[1] if x[1] != '' else x[0]
    columns = [get_col(x) for x in table.columns]
    table.columns = columns

    return table
