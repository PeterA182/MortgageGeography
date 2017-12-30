from __future__ import division

import sys

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/configs/"
)
from config import configs

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/dataPrep/"
)
from a_originationLoadPrep import d_outpath

sys.path.append(
    "/Users/peteraltamura/Documents/GitHub/"
    "mortgageResearch/mortgageResearch/featureEngineering/"
)
from b_combinedFeatures import combined_filename, origFE_filename,\
    idx_cols, delinq_days_min

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from matplotlib import pyplot

# Variables
test_size = .33
seed = 7
model_outpath = d_outpath + 'xgb1_results/'
if not os.path.exists(model_outpath):
    os.makedirs(model_outpath)


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
                      (df_result['delinquent_threshold_passed_60'] == 0)
                  ), 'falsePosPred'] = 1

    # False Negative Prediction
    df_result.loc[:, 'falseNegPred'] = 0
    df_result.loc[(
                      (df_result['Y_hat'] == 0) &
                      (df_result['delinquent_threshold_passed_60'] == 1)
                  ), 'falseNegPred'] = 1
    df_result.loc[:, 'match'] = (
        df_result['Y_hat'] == df_result['delinquent_threshold_passed_60']
    )

    return df_result


def logisticRegression_model(X_train, Y_train, X_test, Y_test, idx_cols=idx_cols):
    """

    :param X: 
    :param Y: 
    :return: 
    """

    # Instantiate and Fit Model
    model = LogisticRegression()
    model.fit(X_train.loc[:, [x for x in X_train.columns if x not in idx_cols]],
              Y_train)
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


if __name__ == "__main__":

    # Read in Data
    df_comb = pd.read_pickle(
        d_outpath + origFE_filename + '.p'
    )

    # Split Exogenous and Endogenous
    Y = df_comb.loc[:, [
        'delinquent_threshold_passed_{}'.format(str(delinq_days_min))
    ]]
    X = df_comb.loc[:,
        [x for x in df_comb.columns if x not in Y.columns]
        ]
    for c in X.columns:
        if sum(X[c].isnull()) > 0:
            print c

    # X index
    print type(idx_cols)
    print X.columns
    X_idx = X.loc[:, idx_cols]
    X = X.loc[:, [x for x in X.columns if x not in idx_cols]]

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y,
        test_size=test_size, random_state=seed
    )
    for c in X_train.columns:
        if sum(X_train[c].isnull()) > 0:
            print c

    # Train and Test Model
    print ":: Instantiating and training model ::"
    model, df_result = logisticRegression_model(X_train=X_train,
                                                Y_train=Y_train,
                                                X_test=X_test,
                                                Y_test=Y_test)

    # # Extract Feature Importances DataFrame
    coeffTable = pd.DataFrame(model.coef_.T)
    print coeffTable
    coeffTable.columns = ['Coefficient']
    coeffTable['Feature'] = X_test.columns

    # Send out
    pyplot.savefig(model_outpath + 'logRegr_coefficientViz.png')
    df_result.to_csv(model_outpath + 'logRegr_df_residuals.csv')
    coeffTable.to_csv(model_outpath + 'logRegr_coefTbl.csv')

    # Print Stats
    print "Percent Match"
    print len(df_result.loc[(
        df_result['delinquent_threshold_passed_60'] == df_result['Y_hat']
    ),  :])/len(df_result)
    print "Percent Positive Match"
    print len(df_result.loc[(
        (df_result['delinquent_threshold_passed_60'] == 1)
        &
        (df_result['Y_hat'] == 1)
    ), :])/len(df_result)
    print "Percent False Positive"
    print str(np.mean(df_result['falsePosPred']))
    print "Percent False Negative"
    print str(np.mean(df_result['falseNegPred']))

