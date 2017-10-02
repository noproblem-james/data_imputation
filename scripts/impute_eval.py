import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error

import data_munging_tools as dmt
import model_fitting_tools as mft

from fancyimpute import BiScaler, KNN, NuclearNormMinimization, SoftImpute, SimpleFill, MICE, MatrixFactorization

import data_munging_tools as dmt
import model_fitting_tools as mft
import eda_tools as et

def pandas_imputer_eval(df, drop_col, impute_percent=10, seed=1984):
    np.random.seed(seed)
    drop_col_idx = df.columns.get_loc(drop_col)

    drop_index = np.random.rand(df.shape[0]) < impute_percent/100.

    drops_df = df.copy()
    drops_df.loc[drop_index, drop_col] = None

    actual_values = df.loc[drop_index, drop_col].values
    # print drops_df.isnull().sum()
    X_filled_mean = drops_df.fillna(drops_df.mean())
    X_filled_median = drops_df.fillna(drops_df.median())
    # print X_filled[drop_index]
    mean_imputer_predictions = X_filled_mean.iloc[drop_index, drop_col_idx]
    median_imputer_predictions = X_filled_median.iloc[drop_index, drop_col_idx]

    mean_mae = mean_absolute_error(actual_values, mean_imputer_predictions)
    mean_rmse = mean_squared_error(actual_values, mean_imputer_predictions) ** 0.5

    median_mae = mean_absolute_error(actual_values, median_imputer_predictions)
    median_rmse = mean_squared_error(actual_values, median_imputer_predictions) ** 0.5

    print "pandas fill_na(mean) mae, rmse: ",  mean_mae, mean_rmse
    print "pandas fill_na(median) mae, rmse: ",  median_mae, median_rmse


def fancy_imputer_eval(df, drop_col, imputer, impute_percent=10, seed=1984):
    '''
    Tests a fancy imputer on an all-numeric dataframe, with values for a single feature changed at random to Null.
    Parameters: dataframe, a feature on which to test, an imputer object, a percent of values to set to null and impute.
    Returns: mean absolute error score, root mean squared error score
    '''
    np.random.seed(seed)
    drop_index = np.random.rand(df.shape[0]) < impute_percent/100.

    drop_col_idx = df.columns.get_loc(drop_col)

    drops_df = df.copy()
    drops_df.loc[drop_index, drop_col] = None
    actual_values = df.loc[drop_index, drop_col].values

    if type(imputer) != SimpleFill:

        biscaler = BiScaler(verbose=0)
        drops_normed = biscaler.fit_transform(drops_df.as_matrix())

        X_filled = imputer.complete(drops_normed)
        X_filled_unnormed = biscaler.inverse_transform(X_filled)
        imputer_predictions = X_filled_unnormed[drop_index, drop_col_idx]

    else:
        X_filled = imputer.complete(drops_df)
        imputer_predictions = X_filled[drop_index, drop_col_idx]


    mae = mean_absolute_error(actual_values, imputer_predictions)
    rmse = mean_squared_error(actual_values, imputer_predictions) ** 0.5

    return {"mae": mae, "rmse": rmse}
    # return pd.DataFrame(X_filled, columns=df.columns)


def imputers_eval(df, imputers_dict, impute_percent=10):
    '''
    tests imputers one at a time for each feature in a df with only numerical features
    params: a test dataframe, a dict of imputers where keys are names of imputers and values are
     imputer objects, a percent of data to set to null for testing purposes
    output: a results data frame with each feature as a row, each imputer type as a column,
        and an rmse score for each feature/imputer type combination
    '''
    df_results_dict = {}
    for col in df.columns:
        col_results_dict = {}
        for k,v in imputers_dict.iteritems():
            print "now imputing values for column {} with imputer {}".format(col, k)
            dict_1 = fancy_imputer_eval(df, col, v, impute_percent=impute_percent)
            col_results_dict[k] = dict_1["mae"]
            # col_results_dict[(k, "rmse")] = rmse
        df_results_dict[col] = col_results_dict
    return pd.DataFrame(df_results_dict).T

def imputers_percent_eval(df, imputer, impute_percents = range(10, 91, 10)):
    '''
    uses fancy impute eval to test masked values at different percent Null
    '''
    df_results_dict = {}
    for col in df.columns:
        col_results_dict = {}
        print "computing error for filling column: {}".format(col)
        for percent_filled in impute_percents:
            percent_result_dict = fancy_imputer_eval(df, col, imputer, impute_percent=percent_filled)
            col_results_dict[percent_filled] = percent_result_dict["mae"]

#             col_results_dict [(percent_filled, "rmse")] = rmse
        df_results_dict[col] = col_results_dict
    return pd.DataFrame(df_results_dict).T


def fancy_impute_test_pipe(train_df, test_df, target, imputer):
    """
    Parameters: training dataframe, testing dataframe, target variable name (as a string), imputer object
    Returns: filled and binarized training dataframe, filled and binarized training dataframe
    """
    test_df = test_df.copy()
    train_df = train_df.copy()

    # Drop rows with missing target values
    test_df.dropna(subset=[target], inplace=True)
    train_df.dropna(subset=[target], inplace=True)
    test_df.reset_index(inplace=True)
    train_df.reset_index(inplace=True)

    #create flags for test and train
    mft.flag_test_train(train_df, test_df)

    ### Split into X and y
    X_train, y_train = mft.X_y_split(train_df, target)
    X_test, y_test = mft.X_y_split(test_df, target)

    #Merge train and test for binarization of train and test and imputation of test
    merged_df = pd.concat([X_train, X_test])

    #split into numeric and nonnumeric
    numeric_df, nonnumeric_df = dmt.split_numerical_features(merged_df, verbose=0)

    #Binarize nonnumeric features
    binarized_df = pd.get_dummies(nonnumeric_df)

    #resplit into train and test
    numerics_train_df = numeric_df[numeric_df["flag"] == 0]
    numerics_test_df = numeric_df[numeric_df["flag"] == 1]
    binarized_train_df = binarized_df[binarized_df["flag_str_train"] == 1]
    binarized_test_df = binarized_df[binarized_df["flag_str_test"] == 1]

    #perform imputations
    filled_train_df = dmt.fancy_impute(numerics_train_df, imputer)
    filled_df = dmt.fancy_impute(numeric_df, imputer)

    #scaling and/or imputing creates rounding error
    filled_df["flag"] = filled_df["flag"].round(0)

    #separate imputed test set from imputed train set
    filled_test_df = filled_df[filled_df["flag"] == 1]

    #rejoin test and train
    rejoined_train_df = filled_train_df.join(binarized_train_df)
    filled_test_df.reset_index(inplace=True)
    rejoined_test_df = filled_test_df.join(binarized_test_df)
    rejoined_test_df.drop("index", axis=1, inplace=True)

    return rejoined_train_df, rejoined_test_df, y_train, y_test

def fancy_impute_test_pipe_2(train_df, test_df, target, imputer):
    """
    Alternative method for performing imputation, in which categorical features are binarized first and
        continuous variables are imputed on the entire dataframe (including binarized variables).
    Parameters: training dataframe, testing dataframe, target variable name (as a string), imputer object
    Returns: filled and binarized training dataframe, filled and binarized training dataframe

    """
    test_df = test_df.copy()
    train_df = train_df.copy()

    # Drop rows with missing target values
    test_df.dropna(subset=[target], inplace=True)
    train_df.dropna(subset=[target], inplace=True)
    test_df.reset_index(inplace=True)
    train_df.reset_index(inplace=True)

    #create flags for test and train
    mft.flag_test_train(train_df, test_df)

    ### Split into X and y
    X_train, y_train = mft.X_y_split(train_df, target)
    X_test, y_test = mft.X_y_split(test_df, target)

    #Merge train and test for binarization of train and test and imputation of test
    merged_df = pd.concat([X_train, X_test])

    #Binarize nonnumeric features
    binarized_df = pd.get_dummies(merged_df)

    #split off test
    binarized_train_df = binarized_df[binarized_df["flag"] == 0]

    #perform imputations
    filled_train_df = dmt.fancy_impute(binarized_train_df, imputer)
    filled_df = dmt.fancy_impute(binarized_df, imputer)

    #scaling and/or imputing creates rounding error
    filled_df["flag"] = filled_df["flag"].round(0)

    #separate imputed test set from imputed train set
    filled_test_df = filled_df[filled_df["flag"] == 1]

    return filled_train_df, filled_test_df, y_train, y_test

    # return pd.DataFrame(results_dict).T

def fit_and_score_imputers(train_df, test_df, target, imputers_dict, model, multiscores=True, verbose=1):
    '''
    Binarizes and imputes values in a train and a test dataframe, fits a model and scores it.
    '''

    results_dict ={}
    for imputer_name, imputer in imputers_dict.iteritems():
        rejoined_train_df, rejoined_test_df, y_train, y_test = fancy_impute_test_pipe(train_df, test_df, target, imputer)
        if verbose == 1:
            print "fitting model to dataframe imputed with: {}".format(imputer_name)
        model.fit(rejoined_train_df, y_train)
        r2 = model.score(rejoined_test_df, y_test)
        mse, mae, rmse, rrse = mft.eval_model(model, rejoined_test_df, y_test, y_train)
        if multiscores == True:
            results_dict[imputer_name] = {"r^2": r2, "rmse": rmse, "mae": mae}
        else:
            results_dict[imputer_name] = mae
    return results_dict

def imputers_multifit_test(train_df, test_df, target, imputers_dict, model, epochs=10, verbose=1):
    '''
    Parameters: a training dataframe, a testing dataframe, a target feature name, a machine learning model to use for making
    predictions, the number of times to run the test, and verbosity (whether to print progress at each round)
    Returns: a dictionary where keys are the round of the test which is run, and the model score for that round.
    '''
    epochs_dict = {}
    counter = 0
    for epoch in range(epochs):
        counter += 1
        if verbose == 1:
            print  "*" * 5, "starting round {} of {}".format(counter, epochs), "*" * 5, "\n"
        scores = fit_and_score_imputers(train_df, test_df, target, imputers_dict, model, multiscores=False, verbose=0)
        epochs_dict[counter] = scores
    if verbose == 1:
        print "***** DONE ******"
    return epochs_dict


def add_observations_impute(not_missing_val_df, missing_val_df, test_df, target, imputers_dict, model, num_partitions=26, epochs=50):
    '''
    takes in two dataframes: one which is missing values, imputes that dataframe multiple times with multiple different imputation methods
    '''
    all_results_dict = {}
    total_num_rows = missing_val_df.shape[0]
    print "total rows with missing data: ", total_num_rows
    rows_per_partition = total_num_rows/num_partitions
    print "rows per partition: ", rows_per_partition
    partitions = range(1, num_partitions+1)
    print " partitions: ", partitions

    print "fitting model with only rows where num_clusters exists."
    result_dict = imputers_multifit_test(not_missing_val_df, test_df, target, imputers_dict, model, epochs, verbose=0)
    all_results_dict[0] = result_dict

    for partition in partitions:

        stop_row = rows_per_partition * partition

        print "fitting models for first {} rows where num_clusters does not exist".format(stop_row)

        if stop_row > missing_val_df.shape[0]:
            stop_row = missing_val_df.shape[0]

        new_rows = missing_val_df.iloc[0 : stop_row, :]


        df_training = pd.concat([not_missing_val_df, new_rows])
        result_dict = imputers_multifit_test(df_training, test_df, target, imputers_dict, model, epochs, verbose=0)

        all_results_dict[stop_row] = result_dict

    print "*** END ***"

    results_list = flatten_3D_dict(all_results_dict)

    return results_list



def flatten_3D_dict(outer_dict):
    new_list = []
    for num_rows, inner_dict_1 in outer_dict.iteritems():
        for epoch_num, inner_dict_2 in inner_dict_1.iteritems():
            for imputer_type, mae in inner_dict_2.iteritems():
                entry = [num_rows, imputer_type, mae]
                new_list.append(entry)
    return new_list
