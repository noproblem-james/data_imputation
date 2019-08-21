
import pandas as pd
from math import inf
import data_munging_tools as dmt
import munge_williston_data as mwd

def munge_df(df, coord_cols=[], target_col="", blacklist_cols=[], thresh_dict={}):
    df = (df.copy()
            .rename(columns=str.lower)
            .dropna(subset=coord_cols)
            .assign(api = lambda x: x["api"].str.strip("US"))
            .set_index("api")
            .sort_index(axis=1)
            .assign(length=lambda x: mwd.haversine_distance(x["surface_lat"],
                                                          x["surface_lng"],
                                                          x["bh_lat"],
                                                          x["bh_lng"])
                       )
            .pipe(dmt.remove_outiers, thresh_dict)
            .assign(
                  # prop_per_ft=lambda x: x["total_lbs_proppant"] / x["length"],
                  # fluid_per_ft=lambda x: x["total_volume_bbls"]/ x["length"],
                  stage_spacing=lambda x: x["total_num_stages"] / x["length"],
                  spud_year=lambda x: x["spuddate"].apply(lambda x: float(str(x).split("-")[0])),
                  choke_size= lambda x: x["choke_size"].apply(mwd.parse_choke_size)
                 )
            .pipe(mwd.find_distance_to_nearest_neighbor, *coord_cols)
            .pipe(mwd.normalize_formation, "stimulated_formation", "producedpools")
            # .query("spud_year > 2009")
            # .query("data_group == 'TEST' | spud_year > 2009")
            .drop(blacklist_cols, axis=1)
            .dropna(subset=[target_col])
            .sort_index(axis=1)
             )

    return df

if __name__ == '__main__':

    # Load the dataframes and concat them together
    test_df = pd.read_csv('../data/cleaned-input.test.tsv', sep='\t', low_memory=False)
    train_df = pd.read_csv('../data/cleaned-input.training.tsv', sep='\t', low_memory=False)

    print("train_df columns: ", train_df.columns.tolist())
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

    blacklist_cols = \
                    ['fileno',
                     'num_pools_produced',
                     'section',
                     'dfelev',
                     'grelev',
                     'kbelev',
                     'max_tvd',
                     'min_tvd',
                     'td',
                     'bh_lng',
                     'bh_lat',
                     'production_liquid_120',
                     'production_liquid_150',
                     'production_liquid_1825',
                     'production_liquid_270',
                     'production_liquid_30',
                     'production_liquid_365',
                     'production_liquid_60',
                     'production_liquid_730',
                     'production_liquid_90',
                     'countyname',
                     'currentoperator',
                     'currentwellname',
                     'fieldname',
                     'footages',
                     'leasename',
                     'leasenumber',
                     'originaloperator',
                     'originalwellname',
                     # 'producedpools',
                     'qq',
                     'range',
                     'spud_date',
                     # 'spud_year',
                     'legs',
                     # 'surface_lat',
                     # 'surface_lng',
                     'choke_size',
                     'township',
                     'type_treatment',
                     'well_status_date',
                     'wellbore',
                     'wellstatus',
                     'welltype']

    # munge the df
    full_df = munge_df(concat_df, coord_cols, target_col, blacklist_cols, thresh_dict)
    full_df.to_csv("../data/full_df.tsv", sep="\t")
    print("columns in newly created dataframe: ", full_df.columns.tolist())

    #resplit into test and train and save
    train_df = (full_df.copy()
                       .query("data_group == 'TRAIN'")
                       .drop("data_group", axis=1)
                       # .query("spud_year > 2009")
               )

    test_df =  (full_df.copy()
                      .query("data_group == 'TEST'")
                      .drop("data_group", axis=1)
               )


    train_df.to_csv("../data/train_df.tsv", sep="\t")
    test_df.to_csv("../data/test_df.tsv", sep="\t")
