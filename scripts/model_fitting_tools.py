import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

def X_y_split(df, target):
    '''
    params: df, target variable (as string),
    returns: df_X, df_y
    '''
    df_X = df.copy()
    df_y = df_X.pop(target)
    return df_X, df_y


def eval_model(model, X_test, y_test, y_train):
    # generate error metrics for a model
    naive_preds = np.ones(y_test.shape)*np.mean(y_train.values)
    training_mean_mse = mean_squared_error(y_test, naive_preds)
    training_mean_rmse = training_mean_mse ** 0.5

    test_predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, test_predictions)
    mae = mean_absolute_error(y_test, test_predictions)
    rmse = mse ** 0.5
    rrse = rmse / training_mean_rmse * 100

    return mse, mae, rmse, rrse


def get_rig_df(scikit_model, x_cols):
    # create a dataframe of feature importances from a scikit model
    feat_imp_df = (pd.DataFrame.from_dict({"feature": np.array(x_cols),
                                           "importance": scikit_model.feature_importances_})
               .set_index("feature")
               .sort_values("importance", ascending=False)
              )
    return feat_imp_df

def make_preds(X_test, y_test, target_col, fitted_model):
    # create a dataframe of predictions vs actuals, using a holdout set
    y_hat = fitted_model.predict(X_test)

    eval_df = (pd.concat([y_test.reset_index(),
                          pd.Series(y_hat, name="pred").round(2)], axis=1)
               .set_index("api")
               .rename(columns={target_col: "actual"})
               .assign(resid=lambda x: x["pred"] - x["actual"])
               .assign(perc_resid=lambda x: x["resid"] / x["actual"] * 100)
               .assign(abs_resid=lambda x: x["resid"].abs())
               .assign(abs_perc_resid=lambda x: x["abs_resid"] / x["actual"] * 100)
                )

    mape = eval_df.abs_perc_resid.mean()
    mae = eval_df.abs_resid.mean()
    mape_adj = eval_df.abs_resid.mean() / eval_df.actual.mean() * 100
    print("MAE: ", mae, "\nMAPE: ", mape, "\nadj MAPE: ", mape_adj)

    return eval_df
