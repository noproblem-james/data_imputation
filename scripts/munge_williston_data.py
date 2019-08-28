import pandas as pd
import numpy as np
import math
from sklearn.metrics.pairwise import haversine_distances

def midpoint_helper(lat1, lon1, lat2, lon2):
    '''
    credit: https://github.com/samuelbosch/blogbits/blob/master/geosrc/GreatCircle.py
    '''
    ## little shortcut
    if lon1 == lon2: return (lat1 + lat2) / 2 , lon1
    if lat1 == lat2: return lat1, (lon1 + lon2) / 2

    lon1, lat1 = math.radians(lon1), math.radians(lat1)
    lon2, lat2 = math.radians(lon2), math.radians(lat2)
    dLon = lon2-lon1

    Bx = math.cos(lat2) * math.cos(dLon)
    By = math.cos(lat2) * math.sin(dLon)
    lat3 = math.atan2(math.sin(lat1) + math.sin(lat2),
                      math.sqrt((math.cos(lat1) + Bx)*(math.cos(lat1) + Bx) \
                                + By * By)
                     )
    lon3 = lon1 + math.atan2(By, math.cos(lat1) + Bx)
    return math.degrees(lat3), math.degrees(lon3)

def append_midpoints(df, latcol_1, lngcol_1, latcol_2, lngcol_2, index_col):
    df = df.copy()
    vfunc = vfunc = np.vectorize(midpoint_helper)

    mid_lat, mid_lng = vfunc(df[latcol_1],
                         df[lngcol_1],
                         df[latcol_2],
                         df[lngcol_2])

    df = pd.concat([df.reset_index(),
                              pd.Series(mid_lat, name="mid_lat"),
                              pd.Series(mid_lng, name="mid_lng")], axis=1).set_index(index_col)

    return df

def get_pairwise_dists(df, lat_col, lng_col):
    lat = df[lat_col].apply(math.radians)
    lng = df[lng_col].apply(math.radians)
    R = 3959.87433 * 5280 # approximate radius of earth in ft (mi * ft/mi)
    pairwise_dists_df = pd.DataFrame(haversine_distances(pd.DataFrame([lat, lng]).T),
                                     index=df.index,
                                     columns=df.index)
    return pairwise_dists_df * R # (converting radians to feet)

def get_dist_to_nn(pairwise_dists):
    non_zeros = pairwise_dists[pairwise_dists != 0].copy()
    min_dists = non_zeros.min()
    return min_dists

def append_min_dist_col(df, lat_col, lng_col):
    df = df.copy()
    pairwise_dists_df = get_pairwise_dists(df, lat_col, lng_col)
    min_dists = get_dist_to_nn(pairwise_dists_df)
    df["min_dist"] = min_dists
    return df

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
        # something went wrong. cant be more open than fully open.
        return np.nan

    else:
        decimal = num / denom
        return decimal


def normalize_formation_helper(x):
    if x == "bakken":
        return 1
    elif x == "three forks":
        return 0
    else:
        return np.nan

def normalize_formation(df, primary_col, secondary_col):
    df = df.copy()
    fill_values = df[secondary_col].str.lower().apply(lambda x: normalize_formation_helper(x))
    df["target_formation"] = df[primary_col].apply(lambda x: normalize_formation_helper(x)).fillna(fill_values)
    df.drop([primary_col, secondary_col], axis=1, inplace=True)
    return df
