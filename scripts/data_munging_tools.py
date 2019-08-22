import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import entropy

def calc_entropy(df):
    # select categorical columns, use scipy stats to calculate entropy of
    # each column's value counts
    cat_df = df.copy().select_dtypes("O")
    result = []
    for col in cat_df.columns:
        series = cat_df[col]
        counts = series.value_counts().values
        col_ent = {"col": col, "entropy": entropy(counts)}
        result.append(col_ent)
    df_result = pd.DataFrame(result).set_index("col")
    return df_result

def make_df_summary(df, target_col=None):
    '''
    create summary statistics for dataframe
    '''
    perc_missing = df.isnull().sum() / df.shape[0] * 100
    dtype = df.dtypes
    nunique = df.nunique()
    coeff_var = df.std() / df.mean()
    corr_target = df.corr()[target_col]

    df_describe = df.describe().T
    df_entropy = calc_entropy(df)

    df_summary = (pd.DataFrame([dtype, perc_missing, nunique, coeff_var, corr_target],
                    index=["dtype", "perc_missing", "num_unique", "coeff_var", "corr_target"])
                    .T
                    .reset_index()
                    .rename(columns={"index": "column"})
                    .sort_values(["dtype", "column"])
                    .set_index("column")
                    .join(df_describe)
                    .join(df_entropy)
                    .filter(["dtype", "num_unique", "perc_missing",
                             "mean", "std", "coeff_var", "entropy",
                             "min", "25%", "50%", "75%", "max",
                             "corr_target"
                            ])
                 )

    return df_summary


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

def remove_outiers(df, lims_dict):
    df = df.copy()
    for feature, limits in lims_dict.items():
        print(feature)
        up_lim = limits["max"]
        lo_lim = limits["min"]

        max_mask = df[feature] > up_lim
        min_mask = df[feature] < lo_lim

        both_mask = min_mask | max_mask

        df.loc[both_mask, feature] = np.nan
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

def inspect_cat_plots(df, cat_col, target_col):
    """
    Pass a dataframe, a categorical feature, and the (continuous) target

    Returns a two barcharts for each categorical feature:
     1. a count of the number of unique values in that column
     2. the median score of the target column fo each unique value in that column.
     """
    sub_df = df.filter([cat_col, target_col]).assign(**{cat_col: lambda x: x[cat_col].fillna("NaN")})
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    sns.countplot(data=sub_df, x=cat_col, ax=ax[0])

    for item in ax[0].get_xticklabels() + ax[1].get_xticklabels():
        item.set_rotation(45)

    sns.boxplot(data=sub_df, x=cat_col, y=target_col, ax=ax[1])
    plt.show()


def make_strip_plots(df, col, lims_dict):
    lims = lims_dict[col]
    sub_df = df.filter([col])
    fig, ax = plt.subplots(1, 2, figsize=(18, 7))
    sns.stripplot(data=sub_df, x=col, ax=ax[0])

    print(lims)

    if lims["max"] < float("inf"):
        upper = lims["max"]
        ax[0].axvline(upper, color="r")
        before_filter = sub_df.shape[0]
        queried_df = sub_df.query(f"{col} < @upper")
        after_filter = queried_df.shape[0]
        print(before_filter - after_filter)
        sns.stripplot(data=queried_df, x=col, ax=ax[1])

    if lims["min"] > 0:
        lower = lims["min"]
        ax[0].axvline(lower, color="r")
        before_filter = sub_df.shape[0]
        queried_df = sub_df.query(f"{col} > @lower")
        after_filter = queried_df.shape[0]
        print(before_filter - after_filter)
        sns.stripplot(data=queried_df, x=col, ax=ax[1])

    plt.show()
