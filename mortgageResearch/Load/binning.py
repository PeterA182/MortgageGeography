from __future__ import division

import sys
import glob

import pandas as pd
import numpy as np
import datetime as dt

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/Load"
)
from load_loans import load_data
from reference import (
    originationFileColList, monthlyFileColList, compress_columns
)

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
        "mortgageResearch/mortgageResearch/configs/"
)
from config import configs


# ---- ---- ----      ---- ---- ----      ---- ---- ----      ---- ---- ----
#

# Methods
def create_bins(dest_dir, source, duration, chunk=10000):
    """
    
    :return: 
    """

    # Duration / File Type
    if duration == 'single':
        fileType = 'sample_single_dir'
    elif duration == 'monthly':
        fileType = 'sample_monthly_dir'

    # Load in all files
    df_origination = pd.DataFrame()
    for path in glob.glob(configs[source][fileType] + '*.txt'):

        # Process file
        print "Loading File: {}".format(path)
        df_origination = pd.concat(
            [
                df_origination,
                load_data(
                    path=path,
                    columns=monthlyFileColList,
                    nrows=None,
                    date_col_fmt_dict={'firstPaymentDate': '%Y%m'},
                    error_bad_lines=True
                )
            ],
            axis=0
        )

    # Mapping
    try:
        df_map = pd.read_csv(dest_dir + 'bin_mapping.csv')
    except:
        df_map = pd.DataFrame(columns=['loanSeqNumber', 'bin_number'])

    # Iterate over all loanSegNumber found in files
    loanNumbers = list(set(df_origination['loanSeqNumber']))
    s = 0
    bin_number = 1
    for group in loanNumbers[s:s+chunk]:

        # Filter and save to bin CSV
        msk = (df_origination['loanSeqNumber'].isin(group))
        df_curr = df_origination.loc[msk, :]
        df_curr.to_csv(dest_dir + 'bin_{}.csv'.format(str(bin_number)))

        # Make append to mapping
        df_map = df_map.append(
            pd.DataFrame({
                'loanSeqNumber':
                    list(set(df_curr['loanSeqNumber'])),
                'bin':
                    [bin_number for i in
                     range(len(set(df_curr['loanSeqNumber'])))]}
            )
        )

        # Update iterable counts
        s += chunk
        bin_number += 1

    # Save map back to file
    df_map.to_csv(dest_dir + 'bin_mapping.csv')


def update_bins(dest_dir, source, duration, qtrYears):
    """
    
    :param dest_dir: 
    :param source: 
    :param duration: 
    :param chunk: 
    :return: 
    """

    # Duration / File Type
    if duration == 'single':
        fileType = 'sample_single_dir'
    elif duration == 'monthly':
        fileType = 'sample_monthly_dir'

    # Try reading in Map
    try:
        df_map = pd.read_csv(dest_dir + 'bin_mapping.csv')
    except:
        df_map = pd.DataFrame(
            columns=['loanSeqNumber', 'binNumber',
                     'binOccupancy', 'maxBinOccupancy']
        )

    # Load in all files
    for path in glob.glob(configs[source][fileType] + '*.txt'):

        # Check if this is a file desired for being read in
        if any([qtrYear in path for qtrYear in qtrYears]):

            # Process file
            print "Loading File: {}".format(path)
            currFile = load_data(path=path,
                               columns=monthlyFileColList,
                               nrows=None,
                               date_col_fmt_dict={'firstPaymentDate': '%Y%m'},
                               error_bad_lines=True)
            print "{} many loans in file".format(
                len(list(set(currFile['loanSeqNumber'])))
            )

            #
            #
            # ---- ---- ----      ---- ---- ----
            # ---- ---- ----      ---- ---- ----

            # Split file into loans that are continuing and loans that are new
            print "Updating bin structure: {}".format(path)

            # Update bins in which we can already find loans from new file
            old_new_msk = (currFile['loanSeqNumber'].isin(set(df_map['loanSeqNumber'])))
            df = currFile.loc[old_new_msk, :]

            # Iterate over bins we already have and append
            affected_bins = list(set(df_map.loc[df_map['loanSeqNumber'].isin(
                list(set(df['loanSeqNumber']))
            ), :]['binNumber']))
            for b_ in affected_bins:
                print "{} :: Updating bins with new loan data".format(
                    dt.datetime.now().strftime("%m_%d_%Y")
                )
                df_bin = pd.read_csv(dest_dir + 'bin_{}.csv'.format(str(b_)))
                binLoans = list(set(
                    df_map.loc[df_map['binNumber'] == b_, :]['loanSeqNumber']
                ))
                df_bin = pd.concat([
                    df_bin,
                    df.loc[df['loanSeqNumber'].isin(binLoans), :]
                ],
                    axis=0
                )
                df_bin.to_csv(dest_dir + 'bin_{}.csv'.format(str(b_)))

            #
            #
            # ---- ---- ----      ---- ---- ----
            # ---- ---- ----      ---- ---- ----

            # Save loan information for loans that cannot be found in
            # pre-existing bins
            df = currFile.loc[~old_new_msk, :]
            newLoans = list(set(df['loanSeqNumber']))

            # Check if there is a bin not yet full
            open_bin = df_map.loc[
                       df_map['binOccupancy'] < df_map['maxBinOccupancy'], :]
            open_bin = open_bin.loc[:, ['binNumber', 'binOccupancy']].\
                drop_duplicates(inplace=False)
            assert len(list(open_bin['binNumber'])) <= 1

            # ---- ---- -----
            # Fill up a non-full bin before making new ones

            if len(list(open_bin['binNumber'])) == 1:
                print "{} :: Updating any non-full bin with new loans".format(
                    dt.datetime.now().strftime("%m_%d_%Y")
                )
                curr_bin = list(open_bin['binNumber'])[0]
                open_spots = (configs[source]['maxBinOccupancy'] -
                              list(open_bin['binOccupancy'])[0])
                cut = df.loc[df['loanSeqNumber'].isin(newLoans[:open_spots]), :]

                # Read in bin
                df_bin = pd.read_csv(dest_dir + 'bin_{}.csv'.
                                     format(str(curr_bin)))
                df_bin = pd.concat([
                    df_bin,
                    cut
                ], axis=0)
                df_bin.to_csv(dest_dir + 'bin_{}.csv'.format(str(curr_bin)))
                df = df.loc[~(df['loanSeqNumber'].isin(newLoans[:open_spots])), :]

                # Update df_map to show bin as full
                df_map.loc[df_map['loanSeqNumber'].isin(
                    list(cut['loanSeqNumber'])), 'binOccupancy'] = \
                    configs[source]['maxBinOccupancy']

            # Send the rest of the loans to a new bin
            try:
                curr_bin = max(df_map['binNumber']) + 1
            except ValueError as VE:
                curr_bin = 0

            # ---- ---- ----
            # Create new bin, update map and save

            open_spots = configs[source]['maxBinOccupancy']

            remaining_loans = list(set(df['loanSeqNumber']))
            while len(remaining_loans) > 0:
                print "{} :: Created new bin {} for remaining loans".format(
                    dt.datetime.now().strftime("%m_%d_%Y"),
                    str(curr_bin)
                )
                loan_chunk = remaining_loans[:open_spots]
                cut = df.loc[df['loanSeqNumber'].isin(loan_chunk), :]
                cut.to_csv(dest_dir + 'bin_{}.csv'.format(str(curr_bin)))

                # Update df_map
                df_map = df_map.append(
                    pd.DataFrame({
                        'loanSeqNumber': list(set(cut['loanSeqNumber'])),
                        'binNumber': [curr_bin for i in list(set(cut['loanSeqNumber']))],
                        'binOccupancy': [configs[source]['maxBinOccupancy'] for i in list(set(cut['loanSeqNumber']))],
                        'maxBinOccupancy': [configs[source]['maxBinOccupancy'] for i in list(set(cut['loanSeqNumber']))]
                    })
                )

                # Update Current Bin
                curr_bin += 1

                # Get new list of remaining loans
                remaining_loans = list(set([
                    x for x in remaining_loans if x not in loan_chunk
                ]))
                print (" "*15) + "{} many loans remaining".format(
                    str(len(remaining_loans))
                )
        df_map.to_csv(dest_dir + 'bin_mapping.csv')
















if __name__ == "__main__":

    pass

