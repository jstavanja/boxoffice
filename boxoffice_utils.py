import json
import numpy as np
from sklearn.metrics import mean_squared_log_error


# Computes root mean squared logarithmic error
def rmsle(predicted, actual):
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

