
import pandas as pd, numpy as np, scipy.stats as st


def get_bad_vals_summaries(df, features_subset):
    num_rows = df.shape[0]
    missingness_df = (df.copy()
                          .filter(features_subset)
                          .isnull()
                          .sum()
                          .astype(float)
                          .to_frame()
                          .rename(columns={0: "num_missing"})
                          .assign(perc_missing = lambda x: (x["num_missing"] / num_rows * 100).round(2))
                      )
    zero_df = (df.copy()
                  .filter(features_subset)
                  .select_dtypes("number")
                  .where(lambda x: x == 0)
                  .count()
                  .to_frame()
                  .rename(columns={0: "num_zero"})
                  .assign(perc_zero = lambda x: (x["num_zero"] / num_rows * 100).round(2))
                  )

    neg_df = (df.copy()
                  .filter(features_subset)
                  .select_dtypes("number")
                  .where(lambda x: x < 0)
                  .count()
                  .astype(int)
                  .to_frame()
                  .rename(columns={0: "num_neg"})
                  .assign(perc_neg = lambda x: (x["num_neg"] / num_rows * 100).round(2))
              )


    bad_vals_summary_df = (missingness_df.copy()
                                         .join(zero_df)
                                         .join(neg_df)
                                         .round(2)
                          )
    return bad_vals_summary_df


def missingness_crosstab(df, missing_col, cat_col):
    crosstab_df = (pd.crosstab(df[cat_col], df[missing_col].isnull().map({True: "missing", False: "not_missing"}))
                       .assign(perc_missing = lambda x: round(x["missing"] / (x["missing"] + x["not_missing"]) * 100, 3))
                  )
    return crosstab_df


def create_normd_prod_cols(df, prod_cols, length_col):
    df = df.copy()
    for prod_col in prod_cols:
        df[prod_col + "_per_foot"] = df[prod_col] / df[length_col]
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

#### Create markdown
def pandas_to_markdown_table(df):
    cols = df.columns
    out = '   |   '.join(cols) + '\n'
    out += ('  |   '.join([' -- '] * len(cols)) + '\n')
    for idx, row in df.iterrows():
        out += ('  |   '.join([str(v) for v in row.values]) + '\n')
    return print(out)


### Plotting

def corrfunc(x, y, **kws):
    r, p = st.pearsonr(x, y)
    ax = plt.gca()
    ax.annotate("pearsonr = {:.2f} ; p = {:.2f}".format(r, p),
                xy=(.05, .01), xycoords=ax.transAxes)

def corrfunc_groups(x, y, **kws):
    '''
    annotates axes with pearsonr and pval when `hue` is defined, and groups have been created.
    '''
    r, p = st.pearsonr(x, y)
    ax = plt.gca()
    # count how many annotations are already present
    n = len([c for c in ax.get_children() if isinstance(c, mpl.text.Annotation)])
    pos = (.05, .95 - .05 * n)
    ax.annotate("{}: r={:.2f}; p={:.2f}".format(kws['label'][:8].lower(), r, p),
                xy=pos, xycoords=ax.transAxes, color=kws['color'])

def normalize_axes_limits(grid):
    for axes in grid.axes:
        for ax in axes:
            if len(ax.collections) > 0:
                data_xmin = ax.collections[0].get_offsets()[:, 0].min()
                new_xmin = data_xmin - abs(data_xmin) * .05

                data_xmax = ax.collections[0].get_offsets()[:, 0].max()
                new_xmax = data_xmax + abs(data_xmax) * .05
                ax.set_xlim(new_xmin, new_xmax)

                data_ymin = ax.collections[0].get_offsets()[:, 1].min()
                new_ymin = data_ymin - abs(data_ymin) * .05

                data_ymax = ax.collections[0].get_offsets()[:, 1].max()
                new_ymax = data_ymax + abs(data_ymax) * .05
                ax.set_ylim(new_ymin, new_ymax)
