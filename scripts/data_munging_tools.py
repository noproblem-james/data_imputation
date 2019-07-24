import pandas as pd
import numpy as np
import re
# from fancyimpute import BiScaler, SimpleFill
# import model_fitting_tools as mft


def drop_blacklist(df, exceptions={}, blacklist_patterns=[]):
    '''
    Removes columns based on regex-detected patterns
    parameters: df, set of exceptions to pattern, list of blacklist patterns
    '''
    df = df.copy()
    blacklisted_cols = []
    before_shape = df.shape

    for col in df.columns:
        if col in exceptions:
            continue

        else:
            for pattern in blacklist_patterns:
                if re.match(pattern, col):
                    blacklisted_cols.append(col)

    df.drop(blacklisted_cols, inplace=True, axis=1)
    after_shape = df.shape
    num_dropped = before_shape[0] - after_shape[0]
    print(f"Number of columns dropped for blacklist_pattern: {num_dropped}")
    return df


def drop_hi_lo_card(df, exceptions={}, id_col=""):
    '''
    Drop cardinality == 0, cardinality == 1, cardinality == n,
    or (type='categorical and cardinality > 0.2 * n)
    '''
    print(f"Shape before cardinality removal: {df.shape}")

    df = df.copy()

    num_rows = df.shape[0]

    for col in df.columns:
        if col in exceptions:
            continue
        else:
            nuniques = df[col].nunique()

            if nuniques == 0:
                # drop cardinality = 0 (empty columns)
                df.drop(col, inplace=True, axis=1)
            elif nuniques == 1:
                # drop cardinality = 1 (val same for every row)
                df.drop(col, inplace=True, axis=1)
            elif nuniques == df.shape[0]:
                # drop unique id... different for every column
                df.drop(col, inplace=True, axis=1)
#                 print 'Dropped {} since it was always unique'.format(col)
            elif col != id_col and df[col].dtype == 'object' and len(df[col].value_counts()) > num_rows * 0.2:
                df.drop(col, inplace=True, axis=1)
#                 print 'Dropped {} since it was categorical and had a high cardinality'.format(col)

    print(f"Shape after cardinality removal: {df.shape}")
    return df

def drop_high_nulls(df, exceptions={}, cutoff=0.5):
    print(f"Shape before high null removal: {df.shape}")
    df = df.copy()

    for col in df.columns:
        if col in exceptions:
            continue
        else:
            prop_missing = df[col].isnull().sum() / float(df[col].shape[0])
            if prop_missing > cutoff:
                df.drop(col, inplace=True, axis=1)

    print(f"Shape before high null removal: {df.shape}")
    return df


def reduce_cardinality(df, cols=None, cardinality=5, threshold=15):
    '''
    Parameters
    ----------
    df: a pandas DataFrame
    cols: a list of categorical columns
    cardinality: a maximum cardinality (i.e., a maximum number of categories)
    threshold:  a minimum value count threshold (a category must have, at minimum, this number of observations)

    Returns
    -------
    df: a pandas DataFrame, indentical to the original, except that the categorical columns specified have at most
    number of categories specified by `max cardinality` (including the 'OTHER' category), AND every category has at least
    as many observations as specified by value count threshold
    '''

    if cols == None:
        cols = df.select_dtypes(include="O").columns.values.tolist()

    df = df.copy()

    def create_rollup(col):
        rollup_dict = {}
        for idx, row in col.value_counts().reset_index().iterrows():
            if (idx < cardinality) and (row[col.name] >= threshold):
                rollup_dict[row['index']] = row['index']
            else:
                rollup_dict[row['index']]= "OTHER"
        new_series = col.map(rollup_dict)
        return new_series

    df[cols] = df[cols].apply(create_rollup)

    return df

def haversine_distance(s_lat, s_lng, e_lat, e_lng):
   '''
   https://stackoverflow.com/questions/4913349/haversine-formula-in-python-bearing-and-distance-between-two-gps-points
   '''
   R = 3959.87433  # approximate radius of earth in mi

   s_lat = s_lat*np.pi/180.0
   s_lng = np.deg2rad(s_lng)
   e_lat = np.deg2rad(e_lat)
   e_lng = np.deg2rad(e_lng)

   d = np.sin((e_lat - s_lat)/2)**2 + np.cos(s_lat)*np.cos(e_lat) * np.sin((e_lng - s_lng)/2)**2

   return 2 * R * np.arcsin(np.sqrt(d)) * 5280


#                 print 'Dropped {} since it had a high proportion of missing values. {}'.format(col, prop_missing)
#     print ("Shape before high null removal: {}").format(df.shape)

# def drop_categorical_features (df):
#     print "Shape before removal: {}".format(df.shape)
#     columns_removed= []
#     for col in df.columns:
#         if df[col].dtypes == object:
#             df.drop(col, inplace=True, axis=1)
#             columns_removed.append(col)
#     print "Categorical olumns dropped: {}".format(columns_removed)
#     print "Shape after removal: {}".format(df.shape)
#
# def drop_nonnumeric_features (df):
#     df = df.copy()
#     print "Shape before removal: {}".format(df.shape)
#     columns_removed= []
#     for col in df.columns:
#         if df[col].dtypes != float and df[col].dtypes != int:
#             df.drop(col, inplace=True, axis=1)
#             columns_removed.append(col)
#     print "Columns dropped: {}".format(columns_removed)
#     print "Shape after removal: {}".format(df.shape)
#     return df
#
# def split_numerical_features(df, verbose=1):
#     numeric_cols = []
#     nonnumeric_cols = []
#     for col in df.columns:
#         if df[col].dtypes == float or df[col].dtypes == int:
#             numeric_cols.append(col)
#         else:
#             nonnumeric_cols.append(col)
#     numeric_df = df[numeric_cols]
#     nonnumeric_df = df[nonnumeric_cols]
#     if verbose == 1:
#         print "numeric columns: {}".format(numeric_cols)
#         print "non-numeric columns: {}".format(nonnumeric_cols)
#     return numeric_df, nonnumeric_df
#
# def fancy_impute(df, imputer):
#     '''
#     fills numerical dataframe with fancy imputer and returns completed dataframe
#     '''
#     if type(imputer) != SimpleFill:
#
#         biscaler = BiScaler(verbose=0)
#         normed = biscaler.fit_transform(df.as_matrix())
#
#         filled_mat = imputer.complete(normed)
#         filled_mat = biscaler.inverse_transform(filled_mat)
#
#     else:
#         filled_mat = imputer.complete(df)
#
#     filled_df = pd.DataFrame(filled_mat, columns= df.columns)
#
#     return filled_df
#
# def extra_fancy_impute(df, simple_imputer, fancy_imputer, important_features):
#     '''
#     first, fill all nulls on most features with a simple imputation method, like median().
#     second, fill remaining nulls on important features with fancy imputer.
#     '''
#     first_pass_df = df[df.columns.difference(important_features)]
#     first_pass_filled = simple_imputer.complete(first_pass_df)
#     second_pass = np.concatenate((first_pass_filled, df[important_features].as_matrix()), axis=1)
#     print first_pass_filled.shape, df[important_features].as_matrix().shape
#     print second_pass.shape
#     biscaler = BiScaler(verbose=0)
#     normed = biscaler.fit_transform(second_pass)
#     filled_mat = fancy_imputer.complete(normed)
#     filled_mat = biscaler.inverse_transform(filled_mat)
#     filled_df = pd.DataFrame(filled_mat, columns= df.columns)
#     return filled_df
#
# def munge_pipe(df, blacklist_patterns=[], exceptions={}, null_cutoff=.05):
#     '''
#     parameters: dataframe, blacklist patterns (as list), exceptions to blacklist patterns
#         (as set)
#     returns: copy of munged dataframe
#     '''
#     df = df.copy()
#     print "df shape before removals {}".format(df.shape)
#     drop_blacklist(df, blacklist_patterns=blacklist_patterns, exceptions=exceptions)
#     print "*" * 50
#     drop_high_cardinality(df, exceptions=exceptions)
#     print "*" * 50
#     drop_high_nulls(df, exceptions=exceptions, cutoff=null_cutoff)
#     print "df shape after removals {}".format(df.shape)
#     return df
#
# def fancy_impute_pipe(train_df, test_df, target, imputer):
#     """
#     Parameters: training dataframe, testing dataframe, target variable name (as a string), imputer object
#     Returns: filled and binarized training dataframe, filled and binarized training dataframe
#     """
#     test_df = test_df.copy()
#     train_df = train_df.copy()
#
#     # Drop rows with missing target values
#     test_df.dropna(subset=[target], inplace=True)
#     train_df.dropna(subset=[target], inplace=True)
#     test_df.reset_index(inplace=True)
#     train_df.reset_index(inplace=True)
#
#     #create flags for test and train
#     flag_test_train(train_df, test_df)
#
#     ### Split into X and y
#     X_train, y_train = mft.X_y_split(train_df, target)
#     X_test, y_test = mft.X_y_split(test_df, target)
#
#     #Merge train and test for binarization of train and test and imputation of test
#     merged_df = pd.concat([X_train, X_test])
#
#     #split into numeric and nonnumeric
#     numeric_df, nonnumeric_df = split_numerical_features(merged_df, verbose=0)
#
#     #Binarize nonnumeric features
#     binarized_df = pd.get_dummies(nonnumeric_df)
#
#     #resplit into train and test
#     numerics_train_df = numeric_df[numeric_df["flag"] == 0]
#     numerics_test_df = numeric_df[numeric_df["flag"] == 1]
#     binarized_train_df = binarized_df[binarized_df["flag_str_train"] == 1]
#     binarized_test_df = binarized_df[binarized_df["flag_str_test"] == 1]
#
#     #perform imputations
#     filled_train_df = fancy_impute(numerics_train_df, imputer)
#     filled_df = fancy_impute(numeric_df, imputer)
#
#     #scaling and/or imputing creates rounding error
#     filled_df["flag"] = filled_df["flag"].round(0)
#
#     #separate imputed test set from imputed train set
#     filled_test_df = filled_df[filled_df["flag"] == 1]
#
#     #rejoin test and train
#     rejoined_train_df = filled_train_df.join(binarized_train_df)
#     filled_test_df.reset_index(inplace=True)
#     rejoined_test_df = filled_test_df.join(binarized_test_df)
#     rejoined_test_df.drop("index", axis=1, inplace=True)
#
#     return rejoined_train_df, rejoined_test_df, y_train, y_test
#
# def flag_test_train(df_train, df_test, string_flag=True):
#     '''
#     #create two flags for test and train, where one flag is a string, the other is a binary
#     '''
#     df_train["flag"] = 0
#     df_test["flag"] = 1
#     if string_flag == True:
#         df_train["flag_str"] = "train"
#         df_test["flag_str"] = "test"
