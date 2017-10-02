from imports import *

def short_info(df):
    # name =[x for x in globals() if globals()[x] is df][0]
    print "*"*50
    # print "dataframe name: {}".format(name)
    print "shape: {}".format(df.shape)
    print "index: {}".format(df.index)
    print "Nulls exist: {}".format(np.any(df.isnull()))

def eda_basics(df):
    print df.info()
    print "\n"
    return df.describe()

    # scatter = pd.scatter_matrix(df, diagonal="kde", figsize = (10, 10))
    # return df.head()

def column_summaries (df):
    nan_percents = pd.DataFrame((1- (df.count()/df.shape[0])) * 100, columns=["percent_nans"])
    dtypes = pd.DataFrame(df.dtypes, columns=["dtype"])
    unique_vals = unique_value_counts(df)
    return dtypes.join(unique_vals).join(nan_percents)

def unique_value_counts(df):
    dict1 = OrderedDict()
    for col in df:
        dict1[col] = df[col].unique().size
    uq_count = pd.DataFrame(dict1, index=[0]).transpose()
    uq_count.columns = ["unq_val_ct"]
    return uq_count

def nan_counts(df, plot=False):
    from collections import OrderedDict
    # dict2 = OrderedDict({col: df[col].isnull().sum()/float(df.shape[0]) for col in df})
    dict1 = OrderedDict()
    for col in df:
        dict1[col]=df[col].isnull().sum()/float(df.shape[0])
    nan_count = pd.DataFrame(dict1, index=[0]).transpose()
    nan_count.columns = ["prop NaNs of total"]
    if plot==True:
        nan_count.plot.bar(figsize=(18,10))
    return nan_count

def vc_bars(df, cols_to_eval):
    """
    takes arguments: DataFrame and list of columns to evaluate
    Returns bar charts of count of all unique values in DataFrame
    """
    for column in cols_to_eval:
        vc = pd.DataFrame(df[column].value_counts())
        vc.plot.bar(figsize=(15,5), fontsize=14)

def groupby_vs_target(df,cols_to_eval,target_col):
    '''
    takes arguments (df, cols_to_eval, and target_col)
    returns barplots of median target val for that category
    '''
    for column in cols_to_eval:
        new_df = df[[column, target_col]].groupby(column).median()
        new_df.plot.bar(figsize=(15,5), fontsize=14)


def bar_plots(df, feature_cols, target_col, figsize=(10,10)):
    """
    Pass a dataframe, a list of columns that are categorical features, and a
    continuous target variable.

    Returns a two barcharts for each categorical feature:
     1. a count of the number of unique values in that column
     2. the median score of the target column fo each unique value in that column.
     """

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(len(feature_cols) ,2, figsize=(20,22))
    ax = ax[:,np.newaxis]
    for column, ax in zip(df[feature_cols], ax):
        row_idx = 0

        vc = df[column].value_counts().sort_index()
        ax[row_idx, 0].bar(range(vc.size),vc)
        ax[row_idx, 0].set_xticks(range(vc.size))
        ax[row_idx, 0].set_xticklabels(vc.index, fontsize=9)
        ax[row_idx, 0].set_title("Value counts of {}".format( column))


        gb = df.groupby(column).median()[target_col].sort_index()
        ax[row_idx, 1].bar(range(gb.size),gb)
        ax[row_idx, 1].set_xticks(range(gb.size))
        ax[row_idx, 1].set_xticklabels(gb.index, fontsize=9)

        ax[row_idx, 1].set_title("Median {} by {}".format(target_col, column))

        row_idx += 1

    plt.tight_layout()

def eda_plots(df, figsize=(10,10)):
    """
    This needs work
    """

    plt.rcParams.update({'font.size': 14})
    fig, ax = plt.subplots(df.shape[1]/3,3, figsize=figsize)
    ax = ax.flatten()
    for column, ax in zip(df.columns, ax):
        # if(df[column].dtype == np.float64 or df[column].dtype == np.int64):
        #     ax.boxplot(df[column])
        #     ax.grid(True)
        #     ax.set_xticklabels("")
        #     ax.set_xlabel(str(column))
        # else:
        vc = df[column].value_counts()
        ax.bar(range(vc.size),vc)
        ax.set_xticks(range(vc.size))
        ax.set_xticklabels(vc.index, fontsize=9)
        ax.set_xlabel(str(column))
    plt.tight_layout()

def remove_char_from_string(df, col, text):
    '''
    Strip charatters from values in a column in a DataFrame
    '''
    df[col] = [x.strip(text) if type(x)==str else x for x in df[col]]

def cols_by_dtype(df):
    pass

def model_summary(X, y, label='scatter'):
    """
    Uses stats models to fit an OLS linear regression, add an intercept,
    and return a model summary.
    """
    X = sm.add_constant(X)
    model = sm.OLS(y, X).fit()
    summary = model.summary()
    return summary

def parse_dates (df, col):
    df[col] = pd.to_datetime(df[col])
    df[year] = pd.DatetimeIndex(df[col]).year
    # df['salemonth'] = pd.DatetimeIndex(df['saledate']).month
    # df['saleday'] = pd.DatetimeIndex(df['saledate']).day
    # df['sale_dow'] = pd.DatetimeIndex(df['saledate']).dayofweek
