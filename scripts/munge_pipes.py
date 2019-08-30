
import pandas as pd
import numpy as np
import json
from math import inf
import data_munging_tools as dmt
import munge_williston_data as mwd


def make_data_dict(fp='../data/AttributeDescriptions.html'):
    data_dict_df = (pd.read_html(fp)[0]
                 .copy()
                 .rename(columns=lambda x: x.lower().replace(" ", "_"))
                 .assign(field_name=lambda x: x["field_name"].apply(lambda x: x.lower()),
                         using=False,
                         notes=np.nan
                         )
                 .set_index("field_name")
                 .sort_index()
                 .pipe(fill_data_dict)
                 )

    return data_dict_df


def fill_data_dict(data_dict_df):
    data_dict_df =  data_dict_df.copy()
    # Metadata
    meta_cols = ["api",
                 "fileno",
                 "currentoperator",
                 "currentwellname",
                 "originaloperator",
                 "originalwellname",
                 "spud_date",
                 "well_status_date",
                 "wellbore",
                 "wellstatus",
                 "welltype"
                 ]

    data_dict_df.loc[meta_cols, "category"] = "metadata"
    data_dict_df.loc[meta_cols, "using"] = False
    data_dict_df.loc["spud_date", "using"] = False # Not using string version
    data_dict_df.loc["spud_year", ["using", "category", "description", "notes"]] = \
        [True, "metadata", "year well was completed", "engineered feature, using for imputation purposes"]
    data_dict_df.loc["data_group", ["using", "category", "description", "notes"]] = \
        ["False", "metadata", "train or test", "train-test flag"]


    # Target
    # For now, we're only predicting production at a single IP day (180)
    data_dict_df.loc[data_dict_df.category == 'Target', ["using"]] = False
    data_dict_df.loc["production_liquid_180", ["using"]] = True

    # Location
    data_dict_df.loc[["footages", "qq"], "category"] = "Location"

    new_location_rows = []
    for col in ['mid_lat', 'mid_lng']:
        row ={
            "field_name": col,
            "category": "Location",
            "description": "midpoint between tophole and bottomhole",
            "using": False,
            "notes": "engineered feature"}
        new_location_rows.append(row)

    data_dict_df = data_dict_df.append(pd.DataFrame(pd.DataFrame(new_location_rows).set_index("field_name")))

    data_dict_df.loc[data_dict_df.category == 'Location', ["using"]] = False
    data_dict_df.loc[["surface_lat", "surface_lng"], ["using", "notes"]] = [True, "using for imputation purposes"]

    # Depth measurements
    tvd_cols = data_dict_df.T.filter(regex="tvd").columns.tolist()
    tvd_cols.append("td")
    data_dict_df.loc[tvd_cols, ["category", "using"]] = "Location", False
    data_dict_df.loc[["tvd", "std_tvd"], "using"] = True
    data_dict_df.loc["max_tvd", "notes"] = "redundant to `tvd`"

    # Elevation measurements
    elev_cols = data_dict_df.T.filter(regex="elev").columns.tolist()
    data_dict_df.loc[elev_cols, ["category", "using"]] = "Location", False

    # Geology
    data_dict_df.loc[['bakken_isopach_ft', 'stimulated_formation'], "using"] = True
    data_dict_df.loc['producedpools', "using"] = False

    ### Completion Design Parameters
    data_dict_df.loc[data_dict_df.category == 'CompletionDesign', "notes"] = "total completion"
    data_dict_df.loc[["legs", "choke_size", "num_pools_produced"], "category"] = "CompletionDesign"
    data_dict_df.loc[data_dict_df.category == 'CompletionDesign', "using"] = True
    data_dict_df.loc[["type_treatment", "legs", "num_pools_produced"], ["using", "notes"]] = \
        False, "no meaningful variation"

    # Add engineered features
    new_rows = []
    engineered_features = ["prop_per_ft", "length", "fluid_per_ft", "min_dist", "stage_spacing"]
    for feature in engineered_features:
        new_record = {
            "field_name": feature,
            "category": "CompletionDesign",
            "description": None,
            "using": True,
            "notes": "engineered feature"
        }
        new_rows.append(new_record)

    data_dict_df = pd.concat([data_dict_df,
                              pd.DataFrame(new_rows).set_index("field_name")],
                              axis=0, sort=False).sort_index()


    return data_dict_df

def make_model_df(df, index_col, target_col, blacklist_cols, thresh_dict, coord_cols):

    df = (df.copy()
            .rename(columns=str.lower)
            .dropna(subset=coord_cols)
            .assign(api = lambda x: x["api"].str.strip("US"))
            .set_index(index_col)
            .sort_index(axis=1)
            .pipe(mwd.append_midpoints, *coord_cols, index_col=index_col)
            .pipe(mwd.append_min_dist_col, "mid_lat", "mid_lng")
            .pipe(mwd.append_length_col, *coord_cols, index_col=index_col)
            .pipe(dmt.remove_outiers, thresh_dict)
            .assign(
                  prop_per_ft=lambda x: x["total_lbs_proppant"] / x["length"],
                  fluid_per_ft=lambda x: x["total_volume_bbls"]/ x["length"],
                  stage_spacing=lambda x: x["total_num_stages"] / x["length"],
                  spud_year=lambda x: x["spud_date"].apply(lambda x: float(str(x).split("-")[0])),
                  choke_size= lambda x: x["choke_size"].apply(mwd.parse_choke_size)
                 )
            .pipe(mwd.normalize_formation, "stimulated_formation", "producedpools")
            # .query("spud_year > 2009")
            # .query("data_group == 'TEST' | spud_year > 2009")
             .drop(blacklist_cols, axis=1)
            .dropna(subset=[target_col])
            .sort_index(axis=1)
             )

    return df

def make_inspect_df(df=None, coord_cols=None):
    if df == None:
        df = make_full_df()

    df = (df.copy()
          .rename(columns=str.lower)
          .assign(api = lambda x: x["api"].str.strip("US"))
          .set_index("api")
          .sort_index(axis=1)
          .pipe(mwd.append_midpoints, *coord_cols, index_col="api")
          .pipe(mwd.append_min_dist_col, "mid_lat", "mid_lng")
          .pipe(mwd.append_length_col, *coord_cols, index_col="api")
          .assign(prop_per_ft=lambda x: x["total_lbs_proppant"] / x["length"],
                  fluid_per_ft=lambda x: x["total_volume_bbls"] / x["length"],
                  stage_spacing=lambda x: x["total_num_stages"] / x["length"],
                  spud_year=lambda x: x["spud_date"].apply(lambda x: float(str(x).split("-")[0])),
                  choke_size=lambda x: x["choke_size"].apply(mwd.parse_choke_size)
                 )
        )

    return df


def make_full_df():
     test_df = pd.read_csv('../data/cleaned-input.test.tsv', sep='\t', low_memory=False)
     train_df = pd.read_csv('../data/cleaned-input.training.tsv', sep='\t', low_memory=False)
     full_df = pd.concat([test_df.assign(data_group="TEST"),
                           train_df.assign(data_group="TRAIN")
                          ])
     return full_df


if __name__ == '__main__':

    full_df = make_full_df()

    with open('../scripts/instructions.json') as fp:
        instructions_dict = json.load(fp)

    model_df = make_model_df(full_df, **instructions_dict)

    test_df = model_df.copy().query("data_group == 'TEST'").drop("data_group", axis=1)
    test_df.to_csv("../data/test_df.tsv", sep="\t")

    train_df = model_df.copy().query("data_group == 'TRAIN'").drop("data_group", axis=1)
    test_df.to_csv("../data/train_df.tsv", sep="\t")