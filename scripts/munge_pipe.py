
import pandas as pd
import data_munging_tools as dmt
from math import inf

def munge_df(df, coord_cols, target_col, model_features, thresh_dict):
    df = (df.copy()
            .rename(columns=str.lower)
            .dropna(subset=coord_cols + [target_col])
            .assign(api = lambda x: x["api"].str.strip("US"))
            .set_index("api")
            .sort_index(axis=1)
            .assign(length=lambda x: dmt.haversine_distance(x["surface_lat"],
                                                          x["surface_lng"],
                                                          x["bh_lat"],
                                                          x["bh_lng"]),
                  prop_per_ft=lambda x: x["total_lbs_proppant"] / x["length"],
                  fluid_per_ft=lambda x: x["total_volume_bbls"]/ x["length"],
                  stage_spacing=lambda x: x["total_num_stages"] / x["length"],
                  range_tvd=lambda x: x["max_tvd"] - x["min_tvd"],
                  choke_size= lambda x: x["choke_size"].apply(dmt.parse_choke_size)
                 )
            .pipe(dmt.find_distance_to_nearest_neighbor, *coord_cols)
            .pipe(dmt.remove_outiers, thresh_dict)
            .pipe(dmt.normalize_formation, "stimulated_formation", "producedpools")
            .filter(model_features + ["data_group", target_col])
             )
    return df

if __name__ == '__main__':

    # Load the dataframes and concat them together
    test_df = pd.read_csv('../data/cleaned-input.test.tsv', sep='\t', low_memory=False)
    train_df = pd.read_csv('../data/cleaned-input.training.tsv', sep='\t', low_memory=False)

    concat_df = pd.concat([test_df.assign(data_group="TEST"),
                           train_df.assign(data_group="TRAIN")
                          ])

    # specify key features for munging and feature engineering purposes
    coord_cols = ["surface_lat", "surface_lng", "bh_lat", "bh_lng"]

    target_col = "production_liquid_180"

    thresh_dict = {'total_lbs_proppant': {'min': 0, 'max': 20000000.0},
                    'total_volume_bbls': {'min': 0, 'max': 500000.0},
                    'length': {'min': 2000, 'max': float("inf")},
                    'total_num_stages': {'min': 5, 'max': float("inf")}}

    model_features = ['total_num_stages',
                     'bakken_isopach_ft',
                     'fluid_per_ft',
                     'length',
                     'prop_per_ft',
                     'shortest_dist',
                     'stage_spacing',
                     'std_tvd',
                     'total_lbs_proppant',
                     'total_volume_bbls',
                     'tvd',
                     'choke_size']

    # munge the df
    full_df = munge_df(concat_df, coord_cols, target_col, model_features, thresh_dict)

    #resplit into test and train and save
    train_df = (full_df.copy()
                       .query("data_group == 'TRAIN'")
                       .drop("data_group", axis=1)
               )

    test_df =  (full_df.copy()
                      .query("data_group == 'TEST'")
                      .drop("data_group", axis=1)
               )

    full_df.to_csv("../data/full_df.tsv", sep="\t")
    train_df.to_csv("../data/train_df.tsv", sep="\t")
    test_df.to_csv("../data/test_df.tsv", sep="\t")
