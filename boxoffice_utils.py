import warnings
import json
import pandas as pd
import numpy as np
import xgboost as xgb

from sklearn.metrics import mean_squared_log_error
from sklearn.model_selection import train_test_split, KFold
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
from sklearn.utils.validation import DataConversionWarning

# suppress annoying warning that shows up when training some classifiers with sklearn
warnings.filterwarnings("ignore", category=DataConversionWarning)


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
    # if the predictions are negative, invert the sign of prediction
    mask = df["revenue"] < 0
    df["revenue"][mask] = -df["revenue"][mask]

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

# Fixes broken json column values from the dataset (single and double quotes issue)
# e.g. {'name': "Donners' Company", ... -> {"name": "Donners\' Company", ...
def fix_broken_json_values(train_df, col_name):
    for i in range(train_df.shape[0]):
        broken_json = train_df.at[i, col_name]
        if isinstance(broken_json, str):
            train_df.at[i, col_name] = json.dumps(eval(broken_json))
        else:
            # TODO: find a more universal thing for when broken_json = nan or something else that's weird
            train_df.at[i, col_name] = "[]"
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


def tune_knn(train_X, train_y, neigh_params, weight_params=None, k=5):
    if weight_params is None:
        weight_params = ["uniform"]

    kf = KFold(n_splits=k)
    best_weights, best_k, best_err = None, None, float("inf")
    for curr_weights in weight_params:
        for curr_neighs in neigh_params:
            print("Testing KNN regression for params: k={}, weights='{}'...".format(curr_neighs,
                                                                                    curr_weights))
            errors_folds = []
            for train_idx, test_idx in kf.split(train_X):
                curr_train_X, curr_train_y = train_X.iloc[train_idx, :], train_y.iloc[train_idx, :]
                curr_test_X, curr_test_y = train_X.iloc[test_idx, :], train_y.iloc[test_idx, :]
                knn = KNeighborsRegressor(n_neighbors=curr_neighs,
                                          weights=curr_weights,
                                          n_jobs=-1)
                knn.fit(curr_train_X, curr_train_y)
                curr_preds = knn.predict(curr_test_X)
                errors_folds.append(rmsle(curr_preds, curr_test_y))

            curr_error = np.mean(errors_folds)
            print("... Obtained error: {:.5f}".format(curr_error))
            if curr_error < best_err:
                best_weights, best_k, best_err = curr_weights, curr_neighs, curr_error

    return best_k, best_weights, best_err


def tune_ridge(train_X, train_y, alpha_params, seed=None, k=5):
    kf = KFold(n_splits=k)
    best_alpha, best_err = None, float("inf")
    for curr_alpha in alpha_params:
        print("Testing ridge regression for params: alpha={:.3f}...".format(curr_alpha))
        errors_folds = []
        for train_idx, test_idx in kf.split(train_X):
            curr_train_X, curr_train_y = train_X.iloc[train_idx, :], train_y.iloc[train_idx, :]
            curr_test_X, curr_test_y = train_X.iloc[test_idx, :], train_y.iloc[test_idx, :]

            curr_ridge = Ridge(alpha=curr_alpha, random_state=seed)
            curr_ridge.fit(curr_train_X, curr_train_y)
            curr_preds = curr_ridge.predict(curr_test_X)

            errors_folds.append(rmsle(curr_preds, curr_test_y))

        curr_error = np.mean(errors_folds)
        print("... Obtained error: {:.5f}".format(curr_error))
        if curr_error < best_err:
            best_alpha, best_err = curr_alpha, curr_error

    return best_alpha, best_err


def tune_svr(train_X, train_y, c_params, eps_params, k=5):
    kf = KFold(n_splits=k)
    best_c, best_eps, best_err = None, None, float("inf")
    # lower C = stricter regularization
    for curr_c in c_params:
        for curr_eps in eps_params:
            print("Testing SVM regression for params: c={:.3f}, eps={:.2f}...".format(curr_c,
                                                                                      curr_eps))
            errors_folds = []
            for train_idx, test_idx in kf.split(train_X):
                curr_train_X, curr_train_y = train_X.iloc[train_idx, :], train_y.iloc[train_idx, :]
                curr_test_X, curr_test_y = train_X.iloc[test_idx, :], train_y.iloc[test_idx, :]

                curr_svr = SVR(C=curr_c, epsilon=curr_eps, gamma="scale")
                curr_svr.fit(curr_train_X, curr_train_y)
                curr_preds = curr_svr.predict(curr_test_X)

                errors_folds.append(rmsle(curr_preds, curr_test_y))

            curr_error = np.mean(errors_folds)
            print("... Obtained error: {:.5f}".format(curr_error))
            if curr_error < best_err:
                best_c, best_eps, best_err = curr_c, curr_eps, curr_error

    return best_c, best_eps, best_err


def tune_rf(train_X, train_y, n_estimators_params, seed=None, k=5):
    kf = KFold(n_splits=k)
    best_n_trees, best_err = None, float("inf")
    for curr_n_trees in n_estimators_params:
        print("Testing RF regression for params: n_estimators={}...".format(curr_n_trees))

        errors_folds = []
        for train_idx, test_idx in kf.split(train_X):
            curr_train_X, curr_train_y = train_X.iloc[train_idx, :], train_y.iloc[train_idx, :]
            curr_test_X, curr_test_y = train_X.iloc[test_idx, :], train_y.iloc[test_idx, :]

            # use MAE criterion -> MSE squaring the already huge errors might cause numerical issues
            curr_rf = RandomForestRegressor(n_estimators=curr_n_trees,
                                            criterion="mae",
                                            random_state=seed,
                                            n_jobs=-1)
            curr_rf.fit(curr_train_X, curr_train_y)
            curr_preds = curr_rf.predict(curr_test_X)
            errors_folds.append(rmsle(curr_preds, curr_test_y))

        curr_error = np.mean(errors_folds)
        print("... Obtained error: {:.5f}".format(curr_error))
        if curr_error < best_err:
            best_n_trees, best_err = curr_n_trees, curr_error

    return best_n_trees, best_err


def tune_xgb(train_X, train_y, lr_params, lambda_params, num_rounds_params, k=5):
    kf = KFold(n_splits=k)

    best_lr, best_lambda, best_rounds, best_err = None, None, None, float("inf")
    param = {"n_gpus": 0, "nthread": -1, "learning_rate": 0.001, "reg_lambda": 1.0}
    for curr_lr in lr_params:
        for curr_reg_lambda in lambda_params:
            for curr_num_rounds in num_rounds_params:
                print("Testing XGB regression for params: learning_rate={:.3f}, l2_lambda={:.2f}, "
                      "num_boosting_rounds={:d}...".format(curr_lr,
                                                           curr_reg_lambda,
                                                           curr_num_rounds))
                param["learning_rate"] = curr_lr
                param["reg_lambda"] = curr_reg_lambda

                errors_folds = []
                for train_idx, test_idx in kf.split(train_X):
                    curr_train_X, curr_train_y = train_X.iloc[train_idx, :], \
                                                 train_y.iloc[train_idx, :]
                    curr_test_X, curr_test_y = train_X.iloc[test_idx, :], \
                                               train_y.iloc[test_idx, :]

                    curr_dtrain = xgb.DMatrix(curr_train_X, label=curr_train_y)
                    curr_dtest = xgb.DMatrix(curr_test_X)

                    curr_bst = xgb.train(param, curr_dtrain, num_boost_round=curr_num_rounds)
                    curr_preds = curr_bst.predict(curr_dtest)
                    errors_folds.append(rmsle(curr_preds, curr_test_y))

                curr_error = np.mean(errors_folds)
                print("... Obtained error: {:.5f}".format(curr_error))
                if curr_error < best_err:
                    best_lr, best_lambda = curr_lr, curr_reg_lambda
                    best_rounds, best_err = curr_num_rounds, curr_error

    return best_lr, best_lambda, best_rounds, best_err


def display_results(res):
    print("--------------------------------")
    print("Summary of obtained results...")

    for model, res in res.items():
        print("Model '{}':".format(model))
        for prop, value in res.items():
            print("'{}': {}".format(prop, value))


def run_models(df_offline, feats, label, test_prop, models, seed=None, k=5):
    """
    Perform (1) parameter tuning and (2) prediction on offline test set using the models,
    specified in the `models` dictionary. K-fold cross validation is used for parameter
    tuning.

    Supported models and their parameters:

    `"knn"` ("neigh_params", "weight_params"),

    "ridge" ("alpha_params"),

    "svr" ("c_params", "eps_params"),

    "rf" ("n_estimators_params"),

    "xgb" ("lr_params", "lambda_params", "num_rounds_params").

    Example of the model properties (KNN experiments will use reasonable defaults for parameter
    options, gradient boosting experiments will use provided lambda parameters and resonable
    defaults for every other parameter):

    >>>
    model_props = {
        "knn": {},
        "xgb": {"lr_params": [0.01, 0.02]}
    }

    Parameters
    ----------
    df_offline: pd.DataFrame
        Offline dataset (in our case the DataFrame, containing data from 'data/train.csv')

    feats: list of str
        Column names in df_offline to use as features

    label: list of str
        Label name in df_offline to use as label

    test_prop: float
        Proportion of data in df_offline to use as (offline) test set

    models: dict
        Models to run and a list of parameters to use when tuning them

    seed: int, default: None
        Random state to ensure reproducibility of experiment. Leave None if you
        want to run random experiment

    k: int, default: 5
        Number of groups in K-fold cross validation

    Returns
    -------
    res: dict of dict
        Results (in form of a dict) for each model, specified in models
    """
    results = {}
    train, test = train_test_split(df_offline,
                                   shuffle=True,
                                   test_size=test_prop,
                                   random_state=seed)
    print("{} examples in training and {} in test set...".format(train.shape[0], test.shape[0]))
    print("Using k = {}...".format(k))

    # only use the provided features and label
    train_X, train_y = train[feats], train[label]
    test_X, test_y = test[feats], test[label]

    knn_properties = models.get("knn", None)
    if knn_properties is not None:
        # user-provided or "reasonable" default parameter options
        neigh_params = knn_properties.get("neigh_params", [1, 3, 5, 10, 20])
        weight_params = knn_properties.get("weight_params", ["uniform", "distance"])

        best_k, best_weights, best_err = tune_knn(train_X, train_y,
                                                  neigh_params=neigh_params,
                                                  weight_params=weight_params,
                                                  k=k)

        print("Winning parameters for KNN: k={:d}, weights='{:s}'... "
              "Obtained error: {:.5f}".format(best_k, best_weights, best_err))
        knn = KNeighborsRegressor(n_neighbors=best_k,
                                  weights=best_weights,
                                  n_jobs=-1)
        knn.fit(train_X, train_y)
        offline_test_preds = knn.predict(test_X)
        offline_error = rmsle(offline_test_preds, test_y)

        print("[KNN] Offline RMSLE: {:.5f}".format(offline_error))
        results["knn"] = {"best_neigh": best_k,
                          "best_weight": best_weights,
                          "offline_error": offline_error}

    ridge_properies = models.get("ridge", None)
    if ridge_properies is not None:
        # user-provided or "reasonable" default parameter options
        alpha_params = ridge_properies.get("alpha_params", [0.001, 0.002, 0.005, 0.01, 0.02, 0.05,
                                                            0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0])

        best_alpha, best_err = tune_ridge(train_X, train_y,
                                          alpha_params=alpha_params,
                                          seed=seed,
                                          k=k)

        print("Winning parameters for Ridge regression: alpha={:.3f}... "
              "Obtained error: {:.5f}".format(best_alpha, best_err))
        ridge = Ridge(alpha=best_alpha,
                      random_state=seed)
        ridge.fit(train_X, train_y)
        offline_test_preds = ridge.predict(test_X)
        offline_error = rmsle(offline_test_preds, test_y)

        print("[Ridge] Offline RMSLE: {:.5f}".format(offline_error))
        results["ridge"] = {"best_alpha": best_alpha,
                            "offline_error": offline_error}

    svr_properties = models.get("svr", None)
    if svr_properties is not None:
        # user-provided or "reasonable" default parameter options
        c_params = svr_properties.get("c_params", [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5, 1.0,
                                                   2.0, 5.0, 10.0])
        eps_params = svr_properties.get("eps_params", [0.0, 0.01, 0.02, 0.1, 0.2, 0.5, 1.0,
                                                       2.0, 5.0])

        best_c, best_eps, best_err = tune_svr(train_X, train_y,
                                              c_params=c_params,
                                              eps_params=eps_params,
                                              k=k)

        print("Winning parameters for SVM regression: c={:.3f}, eps={:.2f}... "
              "Obtained error: {:.5f}".format(best_c, best_eps, best_err))
        svr = SVR(C=best_c,
                  epsilon=best_eps,
                  gamma="scale")
        svr.fit(train_X, train_y)
        offline_test_preds = svr.predict(test_X)
        offline_error = rmsle(offline_test_preds, test_y)

        print("[SVR] Offline RMSLE: {:.5f}".format(offline_error))
        results["svr"] = {"best_c": best_c,
                          "best_eps": best_eps,
                          "offline_error": offline_error}

    rf_properties = models.get("rf", None)
    if rf_properties is not None:
        # user-provided or "reasonable" default parameter options
        n_estimators_params = rf_properties.get("n_estimators_params", [50, 100, 200, 500, 1000,
                                                                        2000, 5000])

        best_n_trees, best_err = tune_rf(train_X, train_y,
                                         n_estimators_params=n_estimators_params,
                                         seed=seed,
                                         k=k)

        print("Winning parameters for RF regression: n_estimators={}... "
              "Obtained error: {:.5f}".format(best_n_trees, best_err))
        rf = RandomForestRegressor(n_estimators=best_n_trees,
                                   criterion="mae",
                                   random_state=seed,
                                   n_jobs=-1)
        rf.fit(train_X, train_y)
        offline_test_preds = rf.predict(test_X)
        offline_error = rmsle(offline_test_preds, test_y)

        print("[RF regression] Offline RMSLE: {:.5f}".format(offline_error))
        results["rf"] = {"best_n_estimators": best_n_trees,
                         "offline_error": offline_error}

    xgb_properties = models.get("xgb", None)
    if xgb_properties is not None:
        # user-provided or "reasonable" default parameter options
        lr_params = xgb_properties.get("lr_params", [0.001, 0.005, 0.1, 0.2, 0.5])
        lambda_params = xgb_properties.get("lambda_params", [0.01, 0.05, 0.1, 0.2, 0.5, 1.0])
        num_rounds_params = xgb_properties.get("num_rounds_params",
                                               [10, 50, 100, 200, 500, 1000])

        best_lr, best_lambda, best_rounds, best_err = tune_xgb(train_X, train_y,
                                                               lr_params=lr_params,
                                                               lambda_params=lambda_params,
                                                               num_rounds_params=num_rounds_params,
                                                               k=k)

        print("Winning parameters for XGB regression: learning_rate={:.3f}, l2_lambda={:.2f}, "
              "num_boosting_rounds={:d}... Obtained error: {:.5f}".format(best_lr,
                                                                          best_lambda,
                                                                          best_rounds,
                                                                          best_err))
        param = {"n_gpus": 0, "nthread": -1, "learning_rate": best_lr, "reg_lambda": best_lambda}
        bst = xgb.train(param, xgb.DMatrix(train_X, label=train_y),
                        num_boost_round=best_rounds)
        offline_test_preds = bst.predict(xgb.DMatrix(test_X))
        offline_error = rmsle(offline_test_preds, test_y)

        print("[XGBoost regression] Offline RMSLE: {:.5f}".format(offline_error))
        results["xgb"] = {"best_lr": best_lr,
                          "best_lambda": best_lambda,
                          "best_num_rounds": best_rounds,
                          "offline_error": offline_error}

    display_results(results)
    return results


def onehot_genres(df, genre_encoder):
    """
    Parameters
    ----------

    df: pd.DataFrame
        Data

    genre_encoder: dict
        Maps genre names (str) to column indices (int) in newly created one-hot attributes

    Returns
    -------
    extended_df, genre_cols:
        Augmented DataFrame ([0]) and newly generated column (attribute) names ([1])
    """
    # Generate column names for the genre one-hot encoded attributes
    genre_cols = [None for _ in range(len(genre_encoder) + 1)]
    for genre, idx in genre_encoder.items():
        genre_cols[idx] = "{}".format(genre)
    genre_cols[-1] = "genre_count"

    genre_onehot = np.zeros((df.shape[0], len(genre_encoder) + 1))
    for i, row in enumerate(df["genres"]):
        # one dummy feature for each genre + count of genres specified for current movie
        genre_count = 0
        json_row = json.loads(row.replace("'", "\""))
        for entry in json_row:
            curr_genre = entry["name"]
            genre_onehot[i, genre_encoder[curr_genre]] = 1
            genre_count += 1
        genre_onehot[i, -1] = genre_count

    extended_df = df.join(pd.DataFrame(genre_onehot, columns=genre_cols))
    return extended_df, genre_cols


def onehot_original_language(df, lang_encoder):
    """
    Parameters
    ----------

    df: pd.DataFrame
        Data

    lang_encoder: dict
        Maps language names (str) to column indices (int) in newly created one-hot attributes

    Returns
    -------
    extended_df, lang_cols:
        Augmented DataFrame ([0]) and newly generated column (attribute) names ([1])
    """
    # Generate column names for the original_language one-hot encoded attributes
    lang_cols = [None for _ in lang_encoder]
    for lang, idx in lang_encoder.items():
        lang_cols[idx] = "{}".format(lang)

    original_lang_onehot = np.zeros((df.shape[0], len(lang_encoder)))
    for i, row in enumerate(df["original_language"]):
        # one dummy feature for each original language (with enough examples);
        # use "other_lang" attribute for less-represented languages
        idx_lang = lang_encoder.get(row, lang_encoder["other_lang"])
        original_lang_onehot[i, idx_lang] = 1

    #print(lang_cols)
    
    extended_df = df.join(pd.DataFrame(original_lang_onehot, columns=lang_cols))
    return extended_df, lang_cols


def add_important_cast_count(df):
    """
    Parameters
    ----------

    df: pd.DataFrame
        Data

    Returns
    -------
    extended_df:
        Augmented DataFrame ([0]) with the new cast_count column
    """
    with open("data/top100actors.json") as actors:
        top_100_actors = json.load(actors)

    cast_nums = np.zeros(df.shape[0])
    for i, row in enumerate(df["cast"]):
        json_row = json.loads(row)
        important_actors = 0
        for entry in json_row:
            if entry["name"] in top_100_actors:
                important_actors += 1

        cast_nums[i] = important_actors

    extended_df = df.join(pd.DataFrame(cast_nums, columns=["important_cast_count"]))
    
    return extended_df


def prod_count_comp_lang_count(df, feature):
    """
    Parameters
    ----------

    df: pd.DataFrame
        Data

    Returns
    -------
    extended_df:
        Augmented DataFrame ([0]) with the new <feature>_count column
    """
    feature_applied = df[feature].apply(lambda x: len(x))
    feature_cols = np.zeros(df.shape[0])
    for idx, count in enumerate(feature_applied):
        feature_cols[idx] = count

    extended_df = df.join(pd.DataFrame(feature_cols, columns=[feature+"_count"]))
    
    return extended_df


def top_producer_director_writer(df, file, feature):
    """
    Parameters
    ----------

    df: pd.DataFrame
        Data

    Returns
    -------
    extended_df:
        Augmented DataFrame ([0]) with the new top_crew_<job> column
    """
    with open(file) as crew_mem:
        top_crew_mems = json.load(crew_mem)

    crew = np.zeros(df.shape[0])
    for i, row in enumerate(df["crew"]):
        json_row = json.loads(row)
        top_crew = 0
        for entry in json_row:
            job = entry["job"]
            if job == feature:
                crew_name = entry["name"]
                if crew_name in top_crew_mems:
                    top_crew = 1

        crew[i] = top_crew

    extended_df = df.join(pd.DataFrame(crew, columns=["top_crew_"+feature.lower()]))
    
    return extended_df