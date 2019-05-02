import json
import numpy as np
from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split

from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor


# Splits data `df` into 3 sets: training, validation and test. Validation set is
# `train_prop`, `val_prop` and `test_prop` are floats and should sum together to 1
def train_val_test_split(df, train_prop, val_prop, test_prop, seed=None):
    train, val_test = train_test_split(df,
                                       shuffle=True,
                                       test_size=(val_prop + test_prop),
                                       random_state=seed)
    val, test = train_test_split(val_test,
                                 shuffle=False,
                                 test_size=(test_prop / (val_prop + test_prop)),
                                 random_state=seed)

    return train, val, test


# Computes root mean squared logarithmic error
def rmsle(predicted, actual):
    # if the predictions are negative, invert the sign of prediction
    mask = predicted < 0
    predicted[mask] = -predicted[mask]

    return np.sqrt(mean_squared_log_error(y_true=actual, y_pred=predicted))


# Makes submission file for Kaggle - assumes DataFrame `df` contains columns "id" and "revenue"
def write_submission(df, path):
    df.to_csv(path, index=False)
    print("Wrote submission to '{}'...".format(path))


# Fixes budgets and/or revenues for some movies in training set, for which the original data
# is invalid - returns fixed DataFrame
def fix_train_budget_revenue(train_df):
    with open("data/fixes_train_budget_revenue.json") as fix:
        fix_mappings = json.load(fix)

    for i in range(train_df.shape[0]):
        imdb_id = train_df.at[i, "imdb_id"]
        fix = fix_mappings.get(imdb_id, None)
        if fix:
            train_df.at[i, "budget"] = fix.get("budget", train_df.at[i, "budget"])
            train_df.at[i, "revenue"] = fix.get("revenue", train_df.at[i, "revenue"])

    return train_df


# Adds genre information for the examples in training/test set that currently don't have it
def fix_genres(df):
    with open("data/fixes_genre.json") as fix:
        fix_mappings = json.load(fix)

    for i in range(df.shape[0]):
        imdb_id = df.at[i, "imdb_id"]
        fix = fix_mappings.get(imdb_id, None)
        if fix:
            df.at[i, "genres"] = fix

    return df


# Adds runtime information for the examples in training/test set that currently don't have it
def fix_runtime(df):
    with open("data/fixes_runtime.json") as fix:
        fix_mappings = json.load(fix)

    for i in range(df.shape[0]):
        imdb_id = df.at[i, "imdb_id"]
        fix = fix_mappings.get(imdb_id, None)
        if fix:
            df.at[i, "runtime"] = fix

    return df


def tune_knn(train_X, train_y, val_X, val_y, neigh_params, weight_params=None):
    if weight_params is None:
        weight_params = ["uniform"]
    best_weights, best_k, best_err = None, None, float("inf")
    for curr_weights in weight_params:
        for curr_neighs in neigh_params:
            print("Testing KNN regression for params: k={}, weights='{}'...".format(curr_neighs,
                                                                                    curr_weights))
            knn = KNeighborsRegressor(n_neighbors=curr_neighs,
                                      weights=curr_weights,
                                      n_jobs=-1)
            knn.fit(train_X, train_y)
            curr_preds = knn.predict(val_X)
            curr_error = rmsle(curr_preds, val_y)
            print("Error: {:.5f}...".format(curr_error))

            if curr_error < best_err:
                best_weights, best_k, best_err = curr_weights, curr_neighs, curr_error

    return best_k, best_weights, best_err


def tune_ridge(train_X, train_y, val_X, val_y, alpha_params, seed=None):
    best_alpha, best_err = None, float("inf")
    for curr_alpha in alpha_params:
        print("Testing ridge regression for params: alpha={:.3f}...".format(curr_alpha))
        curr_ridge = Ridge(alpha=curr_alpha, random_state=seed)
        curr_ridge.fit(train_X, train_y)
        curr_preds = curr_ridge.predict(val_X)
        curr_error = rmsle(curr_preds, val_y)
        print("Error: {:.5f}...".format(curr_error))

        if curr_error < best_err:
            best_alpha, best_err = curr_alpha, curr_error

    return best_alpha, best_err


def tune_svr(train_X, train_y, val_X, val_y, c_params, eps_params):
    best_c, best_eps, best_err = None, None, float("inf")
    # lower C = stricter regularization
    for curr_c in c_params:
        for curr_eps in eps_params:
            print("Testing SVM regression for params: c={:.3f}, eps={:.2f}...".format(curr_c,
                                                                                      curr_eps))
            curr_svr = SVR(C=curr_c,
                           epsilon=curr_eps,
                           gamma="scale")
            curr_svr.fit(train_X, train_y)
            curr_preds = curr_svr.predict(val_X)
            curr_error = rmsle(curr_preds, val_y)
            print("Error: {:.5f}...".format(curr_error))
            if curr_error < best_err:
                best_c = curr_c
                best_eps = curr_eps
                best_err = curr_error

    return best_c, best_eps, best_err


def tune_rf(train_X, train_y, val_X, val_y, n_estimators_param, seed=None):
    best_n_trees, best_err = None, float("inf")
    for curr_n_trees in n_estimators_param:
        print("Testing RF regression for params: n_estimators={}...".format(curr_n_trees))
        # use MAE criterion -> MSE squaring the already huge errors might cause numerical issues
        curr_rf = RandomForestRegressor(n_estimators=curr_n_trees,
                                        criterion="mae",
                                        random_state=seed,
                                        n_jobs=-1)
        curr_rf.fit(train_X, train_y)
        curr_preds = curr_rf.predict(val_X)
        curr_error = rmsle(curr_preds, val_y)
        print("Error: {:.5f}...".format(curr_error))
        if curr_error < best_err:
            best_n_trees = curr_n_trees
            best_err = curr_error

    return best_n_trees, best_err
