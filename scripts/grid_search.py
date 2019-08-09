import pandas as pd
import pickle
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV

import model_fitting_tools as mft


def fit_model(train_X, y_train, X_test, y_test, random_state=1984):

    scaler = StandardScaler()

    gbr = GradientBoostingRegressor(random_state=random_state)

    mice = IterativeImputer(
                            random_state=random_state,
                            initial_strategy="median"
                            )

    gbr_grid_params = {
                      'n_estimators': [100, 200, 300],
                      'learning_rate': [0.05, 0.01],
                      'max_depth': [8, 12, 16],
                      'min_samples_leaf': [5, 10, 20],
                      }

    pipe = make_pipeline(scaler, mice, gbr)

    pipe_grid = {'gradientboostingregressor__' + k : v for  k, v in gbr_grid_params.items()}

    gscv = GridSearchCV(pipe,
                       pipe_grid,
                       cv=5,
                       scoring='neg_median_absolute_error',
                       verbose=1,
                       refit=True,
                       n_jobs=-1)


    gscv.fit(X_train, y_train)

    best_params = gscv.best_params_
    print("Best Params: ", best_params)

    model = gscv.best_estimator_

    filename = '../results/finalized_model.pickle'
    pickle.dump(model, open(filename, 'wb'))

    return model

def make_preds(X_test, target_col, model):
    y_hat = model.predict(X_test)

    eval_df = (pd.concat([y_test.reset_index(),
                          pd.Series(y_hat, name="pred").round(2)], axis=1)
               .set_index("api")
               .rename(columns={target_col: "actual"})
               .assign(resid=lambda x: x["pred"] - x["actual"])
               .assign(perc_resid=lambda x: x["resid"] / x["actual"] * 100)
               .assign(abs_resid=lambda x: x["resid"].abs())
               .assign(abs_perc_resid=lambda x: x["abs_resid"] / x["actual"] * 100)
                )

    eval_df.to_csv("../results/eval_df.tsv", sep="\t")

    mape = eval_df.abs_perc_resid.mean()
    mae = eval_df.abs_resid.mean()
    mape_adj = eval_df.abs_resid.mean() / eval_df.actual.mean() * 100
    print("MAE: ", mae, "\nMAPE: ", mape, "\nadj MAPE: ", mape_adj)

    return eval_df



if __name__ == '__main__':
    train_df = pd.read_csv("../data/train_df.tsv", sep="\t", index_col="api")
    test_df = pd.read_csv("../data/test_df.tsv", sep="\t", index_col="api")
    target_col = "production_liquid_180"
    # print(train_df.head())
    X_train, y_train = mft.X_y_split(train_df, target=target_col)
    X_test, y_test = mft.X_y_split(test_df, target=target_col)


    model = fit_model(X_train, y_train, X_test, y_test)

    eval_df = make_preds(X_test, target_col, model)
