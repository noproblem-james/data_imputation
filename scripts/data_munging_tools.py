import pandas as pd
import numpy as np
import re


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



from sklearn.metrics.pairwise import euclidean_distances

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