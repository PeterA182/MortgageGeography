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
from b_combinedFeatures import combined_filename, idx_cols, delinq_days_min

import pandas as pd

from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Variables
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
        d_outpath + combined_filename + '.p'
    )

    # Split Exogenous and Endogenous
    Y = df_comb.loc[:, [
        'delinquent_threshold_passed_{}'.format(str(delinq_days_min))
    ]]
    X = df_comb.loc[:,
        [x for x in df_comb.columns if x not in Y.columns]
    ]
    for c in X.columns:
        print c
        assert sum(X[c].isnull()) == 0
    for c in Y.columns:
        print c
        assert sum(Y[c].isnull()) == 0

    # Split
    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=test_size, random_state=seed
    )

    # Train and Test Model
    print ":: Instantiating and training model ::"
    model, df_result = logisticRegression_model(X_train=X_train,
                                                Y_train=Y_train,
                                                X_test=X_test,
                                                Y_test=Y_test)