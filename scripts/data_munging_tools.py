import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
from scipy.stats import entropy
from sklearn.metrics.pairwise import euclidean_distances



def calc_entropy(df):
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



def calc_midpoint(df, lat_1_col, lng_1_col, lat_2_col, lng_2_col):
    df = df.copy().filter([lat_1_col, lng_1_col, lat_2_col, lng_2_col])
    dist_x = df[lng_2_col] - df[lng_1_col]
    dist_y = df[lat_2_col] - df[lat_1_col]
    half_x = dist_x / 2
    half_y = dist_y / 2
    mid_x = df[lng_1_col] + half_x
    mid_y = df[lat_1_col] + half_y
    df["midpoint_lat"] = mid_y
    df["midpoint_lng"] = mid_x
    return df

## TODO Update this function to user Haversine distnace
def find_distance_to_nearest_neighbor(df, lat_col_1, lng_1_col, lat_2_col, lng_2_col):
    midpoint_df = (df.copy()
                       .filter([lat_col_1, lng_1_col, lat_2_col, lng_2_col])
                       .pipe(calc_midpoint, lat_col_1, lng_1_col, lat_2_col, lng_2_col)
                       .filter(["midpoint_lat", "midpoint_lng"])
                       .dropna()
                  )
    euclid_dist_df = pd.DataFrame(euclidean_distances(midpoint_df, midpoint_df), index=midpoint_df.index,
                                  columns=midpoint_df.index)
    no_self = euclid_dist_df[euclid_dist_df != 0].copy()
    df["shortest_dist"] = no_self.min(axis=1)
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


def parse_choke_size(x):
    replace_dict = {"a": "6", "s": "5", "i": "1", "0pen": "1", "b": "6", "g": "9"}

    if x == "na":
        # presumably it has no choke, ie it is open, therefore 1
        return 1

    if type(x) == float and np.isnan(x):
        return x

    if type(x) != str:
        print(x, type(x))
        return np.nan

    if "/" not in x:
        print(x)
        return np.nan

    for k, v in replace_dict.items():
        x = x.replace(k, v)

    x = x.strip("'").strip("_").strip(" ").strip("'")

    fraction = x.split("/")
    if len(fraction) != 2:
        return np.nan
    num, denom = fraction[0], fraction[1]

    try:
        num = int(num)
        denom = int(denom)

    except:
#         print(f"couldnt_convert {num} / {denom}")
        return np.nan


    if num == denom:
        # 64/64 would be open ratio, but assume it is open
        return 1

    elif num > denom:
        # something went wrong. cant be more open than open.
        return np.nan

    else:
        decimal = num / denom
        return decimal


def normalize_formation_helper(x):
    if x in {"bakken", "three forks"}:
        return x
    else:
        return np.nan

def normalize_formation(df, primary_col, secondary_col):
    df = df.copy()
    fill_values = df[secondary_col].str.lower().apply(lambda x: normalize_formation_helper(x))
    df["target_formation"] = df[primary_col].apply(lambda x: normalize_formation_helper(x)).fillna(fill_values)
    df = pd.get_dummies(data=df, columns=["target_formation"], prefix="formation")
    df.drop([primary_col, secondary_col], axis=1, inplace=True)
    return df
