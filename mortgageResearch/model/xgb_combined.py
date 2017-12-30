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
from b_combinedFeatures import combined_filename, idx_cols, \
    origFE_filename, delinq_days_min

import os
import pandas as pd
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from matplotlib import pyplot


"""
Using observations from loanOrigination to determine future default
"""


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
    print "::Instantiating and training model ::"
    model, df_result = xgb_model(X_train=X_train,
                                 Y_train=Y_train,
                                 X_test=X_test,
                                 Y_test=Y_test)

    # Extract Feature Importances DataFrame
    feat_imp = pd.DataFrame(model.feature_importances_)
    feat_imp.columns = ['importance_value']
    feat_imp['Feature'] = X_test.columns

    # Plot Feature Importances
    pyplot.bar(range(len(model.feature_importances_)),
               model.feature_importances_)

    # Send out
    pyplot.savefig(model_outpath + 'featureImportanceViz.png')
    df_result.to_csv(model_outpath + 'df_residuals.csv')
    feat_imp.to_csv(model_outpath + 'featureImportanceTbl.csv')

    print df_result.describe()
