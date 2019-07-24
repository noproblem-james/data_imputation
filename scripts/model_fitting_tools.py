import numpy as np
import pandas as pd

from sklearn.metrics import mean_squared_error, mean_absolute_error

def X_y_split(df, target):
    '''
    params: df, target variable (as string),
    returns: df_X, df_y
    '''
    df_y = df[target]
    df_X = df.drop(target, axis=1)
    return df_X, df_y


def flag_test_train(df_train, df_test, string_flag=True):
    '''
    #create two flags for test and train, where one flag is a string, the other is a binary
    '''
    df_train["flag"] = 0
    df_test["flag"] = 1
    if string_flag == True:
        df_train["flag_str"] = "train"
        df_test["flag_str"] = "test"


def eval_model(model, X_test, y_test, y_train):
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
    feat_imp_df = (pd.DataFrame.from_dict({"feature": np.array(x_cols), 
                                           "importance": scikit_model.feature_importances_})
               .set_index("feature")
               .sort_values("importance", ascending=True)
              )
    return feat_imp_df