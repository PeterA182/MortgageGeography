configs = {
    'freddie': {

        # Combined origination and monthly features
        'comb_logisticRegression': {
            'figure_dir': '/Users/peteraltamura/Documents/GitHub/'
                          'mortgageResearch/Figures/',
            'sample_single_dir': "/Users/peteraltamura/Documents/GitHub/"
                        "mortgageResearch/Data/single/",
            'sample_single_file': "historical_data1_Q12016.txt",
            'sample_monthly_dir': '/Users/peteraltamura/Documents/GitHub/'
                         'mortgageResearch/Data/monthly/',
            'sample_monthly_file': "historical_data1_time_Q12016.txt",
            'binned_monthly_dir': "/Users/peteraltamura/Documents/GitHub/"
                                  "mortgageResearch/Data/monthly/bins/",
            'maxBinOccupancy': 10000,
            'monthly_filenames': {'prepped': 'loadedPrepped_monthlyData',
                                  'FE': 'FE_monthlyData',
                                  'FS': 'FS_monthlyData'},
            'origination_filenames': {'prepped': 'loadedPrepped_originationData',
                                      'FE': 'FE_originationData',
                                      'FS': 'FS_originationData'},
            'combined_filenames': {'prepped': 'loadedPrepped_combinedData',
                                   'FE': 'FE_combinedData',
                                   'FS': 'FS_combinedData'},
            'results_dir': 'modelResults/',
            'default_window_months': 3,
            'idx_cols': ['loanSeqNumber', 'currLoanDelinqStatus',
                         'first_default_month', 'MSA', 'mthlyRepPeriod',
                         'zeroBalanceEffectiveDate',
                         'dueDateLastPaidInstallment', 'maturityDate',
                         'firstPaymentDate', 'first_default_month',
                         'postalCode'],
            'vif_threshold': 20,
            'target': 'currLoanDelinqStatus'
        },

        # origination features only
        'orig_logisticRegression': {
            'figure_dir': '/Users/peteraltamura/Documents/GitHub/'
                          'mortgageResearch/Figures/',
            'sample_single_dir': "/Users/peteraltamura/Documents/GitHub/"
                        "mortgageResearch/Data/single/",
            'sample_single_file': "historical_data1_Q12016.txt",
            'sample_monthly_dir': '/Users/peteraltamura/Documents/GitHub/'
                         'mortgageResearch/Data/monthly/',
            'sample_monthly_file': "historical_data1_time_Q12016.txt",
            'binned_monthly_dir': "/Users/peteraltamura/Documents/GitHub/"
                                  "mortgageResearch/Data/monthly/bins/",
            'maxBinOccupancy': 10000,
            'monthly_filenames': {'prepped': 'loadedPrepped_monthlyData',
                                  'FE': 'FE_monthlyData'},
            'origination_filenames': {'prepped': 'loadedPrepped_originationData',
                                      'FE': 'FE_originationData'},
            'combined_filenames': {'prepped': 'loadedPrepped_combinedData',
                                   'FE': 'FE_combinedData'},
            'results_dir': 'modelResults/',
            'default_window_months': 3,
            'idx_cols': ['loanSeqNumber', 'currLoanDelinqStatus']
        }

    }
}

