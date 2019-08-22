import pandas as pd
import numpy as np
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

## TODO Update this function to use Haversine distnace
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
    df["shortest_dist"] = no_self.min(axis=1) * 1000
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
#     df = pd.get_dummies(data=df, columns=["target_formation"], prefix="formation")
    df.drop([primary_col, secondary_col], axis=1, inplace=True)
    return df
