import pandas as pd
import numpy as np
import pickle
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.compose import ColumnTransformer
import model_fitting_tools as mft

def _track_aux_cols(df, aux_cols):
    # helper function, because cannot operate on column names after imputation step
    all_cols = df.columns
    aux_idx = [df.columns.get_loc(col) for col in aux_cols]
    training_cols = df.drop(aux_cols, axis=1).columns
    return all_cols, aux_idx, training_cols

def write_results(gscv, X_test, y_test, training_cols, target_col, save_model=False):
    # helper function to write results of gridsearchcv to disk
    model = gscv.best_estimator_

    eval_df = mft.make_preds(X_test, y_test, target_col, model)
    eval_df.to_csv("../results/eval_df.tsv", sep="\t")

    feat_imp_df = mft.get_rig_df(model['gbr'],
                                 training_cols)

    feat_imp_df.to_csv("../results/feat_imp_df.tsv", sep="\t")

    gscv_results_df = pd.DataFrame(gscv.cv_results_)

    gscv_results_df.to_csv("../results/gscv_results_df.tsv", sep="\t")

    if save_model == True:
        filename = '../results/finalized_model.pickle'
        pickle.dump(model, open(filename, 'wb'))


def fit_model(train_df, test_df, target_col, aux_cols, random_state=1984, test_only=False):
    '''
    train a model using gridsearchcv, using iterative imputation with auxillary features.
    use best hyperparamters to test model against holdout set.
    write results to disk.
    '''

    X_train, y_train = mft.X_y_split(train_df, target=target_col)
    X_test, y_test = mft.X_y_split(test_df, target=target_col)


    all_cols, aux_idx, training_cols = _track_aux_cols(X_train, aux_cols)

    scaler = StandardScaler()

    gbr = GradientBoostingRegressor(random_state=random_state)

    imputation_estimator = ExtraTreesRegressor(n_estimators=10,
                                    random_state=random_state,
                                    )
    mice = IterativeImputer(
                            # tol=0.1,
                            estimator=imputation_estimator,
                            initial_strategy="median",
                            random_state=random_state
                            )

    col_dropper = ColumnTransformer(remainder='passthrough',
                                    transformers=[('remove', 'drop', aux_idx)]
                                )

    pipe = Pipeline(steps=[
                           ('imputer', mice),
                           ('dropper', col_dropper),
                           ('gbr', gbr)
                          ]
                    )


    gbr_grid_params = {
                  'n_estimators': [300, 500],
                  'learning_rate': [0.05, 0.01],
                  'max_depth': [8, 12, 16],
                  'min_samples_leaf': [5, 10, 20],
                  }


    if test_only == True:
        gbr_grid_params = {
                          'n_estimators': [5, 10],
                          'learning_rate': [0.1],
                          'max_depth': [4],
                          'min_samples_leaf': [20],
                          }

    pipe_grid = {'gbr__' + k : v for  k, v in gbr_grid_params.items()}

    gscv = GridSearchCV(pipe,
                       pipe_grid,
                       cv=10,
                       scoring='neg_median_absolute_error',
                       verbose=1,
                       refit=True,
                       n_jobs=-1)


    gscv.fit(X_train, y_train)

    best_params = gscv.best_params_
    print("Best Params: ", best_params)

    if test_only == False:
        write_results(gscv, X_test, y_test, training_cols, target_col)



if __name__ == '__main__':
    train_df = pd.read_csv("../data/train_df.tsv", sep="\t", index_col="api")
    test_df = pd.read_csv("../data/test_df.tsv", sep="\t", index_col="api")
    target_col = "production_liquid_180"
    aux_cols =  ['surface_lat', 'surface_lng', 'spud_year']

    print(train_df.shape, train_df.columns)
    print(test_df.shape)
    # fit_model(train_df, test_df, target_col, aux_cols, random_state=1984)

    print("DONE")
