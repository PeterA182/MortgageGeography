from __future__ import division

import sys

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/configs/"
)
from config import configs

import os
import pandas as pd
import numpy as np

from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from matplotlib import pyplot

# Variables
# d_outpath = sys.argv[1]
# d_source = sys.argv[2]
# model_name = sys.argv[3]

d_outpath = '/Users/peteraltamura/Documents/GitHub/mortgageResearch/output/' \
            'combinedDefaultPred_logit/'
d_source = 'freddie'
model_name = 'comb_logisticRegression'
model_outpath = d_outpath + 'combLogReg_results/'
if not os.path.exists(model_outpath):
    os.makedirs(model_outpath)
results_outpath = d_outpath + configs[d_source][model_name]['results_dir']
if not os.path.exists(results_outpath):
    os.makedirs(results_outpath)

idx_cols = configs[d_source][model_name]['idx_cols']
test_size = .33
seed = 7


# Methods
def get_results_table(X_test, Y_test, Y_hat):
    """

    :param X_test: 
    :param Y_test: 
    :return: 
    """

    # Reconstruct Table
    df_result = pd.concat([X_test, Y_test], axis=1)
    df_result.loc[:, 'Y_hat'] = Y_hat

    # False Positive Prediction
    df_result.loc[:, 'falsePosPred'] = 0
    df_result.loc[(
                      (df_result['Y_hat'] == 1) &
                      (df_result[configs[d_source][model_name]['target']] == 0)
                  ), 'falsePosPred'] = 1

    # False Negative Prediction
    df_result.loc[:, 'falseNegPred'] = 0
    df_result.loc[(
                      (df_result['Y_hat'] == 0) &
                      (df_result[configs[d_source][model_name]['target']] == 1)
                  ), 'falseNegPred'] = 1
    df_result.loc[:, 'match'] = (
        df_result['Y_hat'] == df_result[configs[d_source][model_name]['target']]
    )

    return df_result


def xgb_model(X_train, Y_train, X_test, Y_test, idx_cols=idx_cols):
    """

    :param X: 
    :param Y: 
    :return: 
    """

    X_train_sub = X_train.loc[:, [
        x for x in X_train.columns if x not in idx_cols
    ]]
    for c in X_train_sub.columns:
        print c

        if sum(X_train_sub[c].isnull()) != 0:
            print "     {} null observations".format(
                str(sum(X_train_sub[c].isnull()))
            )

    for c in Y_train.columns:
        print c
        assert sum(Y[c].isnull()) == 0

    model = XGBClassifier()
    model.fit(
        X_train_sub,
        Y_train
    )
    print model

    # Y-hat Series
    Y_hat = model.predict(
        X_test.loc[:, [x for x in X_test.columns if x not in idx_cols]]
    )

    # Create Results table with Type I and Type I Errors
    df_result = get_results_table(X_test=X_test,
                                  Y_test=Y_test,
                                  Y_hat=Y_hat)

    return model, df_result


def calculate_vif_(X, thresh=20):
    variables = range(X.shape[1])
    dropped=True
    while dropped:
        dropped=False
        vif = [variance_inflation_factor(X[variables].values, ix) for ix in range(X[variables].shape[1])]

        maxloc = vif.index(max(vif))
        if max(vif) > thresh:
            print('dropping \'' + X[variables].columns[maxloc] + '\' at index: ' + str(maxloc))
            del variables[maxloc]
            dropped=True

    print('Remaining variables:')
    print(X.columns[variables])
    return X[variables]


if __name__ == "__main__":

    # Read in Data
    df_comb = pd.read_pickle(
        d_outpath + configs[d_source][model_name]['combined_filenames']['FE'] + '.pkl'
    )
    for c in df_comb.columns:
        if sum(df_comb[c].isnull()) > 0:
            df_comb = df_comb.loc[df_comb[c].notnull(), :]

    # Split Exogenous and Endogenous
    Y = df_comb.loc[:, [
        configs[d_source][model_name]['target']
    ]]

    X = df_comb.loc[:, [
        x for x in df_comb.columns if x not in Y.columns
    ]]
    for c in X.columns:
        if sum(X[c].isnull()) > 0:
            raise Exception("Null value in exogenous variables")

    # X index
    X_idx = X.loc[:, idx_cols]
    X = X.loc[:, [x for x in X.columns if x not in idx_cols]]
    print X.columns


    # X = calculate_vif_(X=X, thresh=configs[d_source][model_name]['vif_thresh'])
    # X.to_pickle(d_outpath + configs[d_source][model_name]['FS'] + '.pkl')
    # X = X.loc[:, [
    #     'creditScore', 'origUPB', 'origLTV', 'origCLTV',
    #     'loan_months_total', 'prev_defaults', 'UPB_max_diff', 'UPB_var'
    # ]]

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size, random_state=seed
    )
    for c in X_train.columns:
        if sum(X_train[c].isnull()) > 0:
            print c

    # Train and Test Model
    print "::Instantiating and Training model ::"
    model, df_result = xgb_model(X_train=X_train,
                                 Y_train=Y_train,
                                 X_test=X_test,
                                 Y_test=Y_test)

    # Send out
    pyplot.savefig(results_outpath + 'logRegr_coefficientViz.png')
    df_result.to_csv(results_outpath + 'logRegr_df_residuals.csv')

    # Print Stats
    # Match
    print "Total Observations:"
    print "     {}".format(str(len(df_result)))
    print "Total Matches"
    total_matches = len(
        df_result.loc[(df_result[configs[d_source][model_name]['target']] ==
                       df_result['Y_hat']), :]
    )
    print total_matches

    print "Default Matches"
    def_matches = len(df_result.loc[(
        (df_result[configs[d_source][model_name]['target']] == 1)
        &
        (df_result['Y_hat'] == 1)
    ), :])
    print def_matches

    print "No Default Matches"
    no_def_matches = len(df_result.loc[(
        (df_result[configs[d_source][model_name]['target']] == 0)
        &
        (df_result['Y_hat'] == 0)
    ), :])
    print no_def_matches

    print "False Positives"
    false_pos = len(df_result.loc[(
        (df_result['Y_hat'] == 1)
        &
        (df_result[configs[d_source][model_name]['target']] == 0)
    ), :])
    print false_pos
    print "False Negatives"
    false_neg = len(df_result.loc[(
        (df_result['Y_hat'] == 0)
        &
        (df_result[configs[d_source][model_name]['target']] == 1)
    ), :])
    print false_neg