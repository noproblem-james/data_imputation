import pandas as pd
import pickle
<<<<<<< HEAD
from sklearn.ensemble import GradientBoostingRegressor, ExtraTreesRegressor
=======
from sklearn.ensemble import GradientBoostingRegressor
>>>>>>> 08c300d1bea6bb230179bcd532ecffb75a91ba4e
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
<<<<<<< HEAD
                            # tol=0.1,
                            estimator =ExtraTreesRegressor(n_estimators=10, random_state=0),
=======
>>>>>>> 08c300d1bea6bb230179bcd532ecffb75a91ba4e
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


if __name__ == '__main__':
    train_df = pd.read_csv("../data/train_df.tsv", sep="\t", index_col="api")
    test_df = pd.read_csv("../data/test_df.tsv", sep="\t", index_col="api")
    target_col = "production_liquid_180"
    # print(train_df.head())
    X_train, y_train = mft.X_y_split(train_df, target=target_col)
    X_test, y_test = mft.X_y_split(test_df, target=target_col)

    model = fit_model(X_train, y_train, X_test, y_test)

    eval_df = mft.make_preds(X_test, y_test, target_col, model)
    eval_df.to_csv("../results/eval_df.tsv", sep="\t")
    
    feat_imp_df = mft.get_rig_df(model['gradientboostingregressor'], 
                                 X_train.columns.values)
<<<<<<< HEAD
    feat_imp_df.to_csv("../results/feat_imp_df.tsv", sep="\t")
=======
    feat_imp_df.to_csv("../results/feat_imp_df.tsv", sep="\t")
>>>>>>> 08c300d1bea6bb230179bcd532ecffb75a91ba4e
