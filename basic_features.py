import numpy as np
import pandas as pd
import xgboost as xgb
from boxoffice_utils import fix_train_budget_revenue, fix_genres, fix_runtime, run_models, \
    write_submission, onehot_genres, onehot_original_language
import json

if __name__ == "__main__":
    df_offline = pd.read_csv("data/train.csv", header=0)
    df_test = pd.read_csv("data/test.csv", header=0)
    df_offline = fix_train_budget_revenue(df_offline)  # type: pd.DataFrame
    df_offline = fix_genres(df_offline)
    df_offline = fix_runtime(df_offline)
    df_test = fix_genres(df_test)
    df_test = fix_runtime(df_test)

    FEATS = ["runtime"]
    LABEL = ["revenue"]

    # Find unique genres and assign indices to them
    idx_genre = 0
    genre_encoder = {}
    for i, row in enumerate(df_offline["genres"]):
        json_row = json.loads(row.replace("'", "\""))
        for entry in json_row:
            curr_genre = entry["name"]
            existing_idx = genre_encoder.get(curr_genre, None)
            if existing_idx is None:
                genre_encoder[curr_genre] = idx_genre
                idx_genre += 1

    # Turn variable-length genre information into fixed-size one-hot encoded attributes
    df_offline, genre_cols = onehot_genres(df_offline, genre_encoder)
    df_test, _ = onehot_genres(df_test, genre_encoder)
    FEATS.extend(genre_cols)

    # Select languages with more than 5 examples in training set
    # (group other languages under "other_lang")
    lang_counts = df_offline["original_language"].value_counts()
    original_langs = lang_counts.index.values[lang_counts >= 5]
    original_langs = np.append(original_langs, "other_lang")
    lang_encoder = dict(zip(original_langs, range(original_langs.shape[0])))

    df_offline, lang_cols = onehot_original_language(df_offline, lang_encoder)
    df_test, _ = onehot_original_language(df_test, lang_encoder)
    FEATS.extend(lang_cols)

    model_props = {"xgb": {}}
    # model_props = {"knn": {},
    #                "ridge": {},
    #                "svr": {},
    #                "rf": {},
    #                "xgb": {}}
    run_models(df_offline, FEATS, LABEL,
               models=model_props,
               test_prop=0.2,
               seed=1337,
               k=5)

    # -------------------------
    # PREPARE A SUBMISSION FILE
    # -------------------------
    # df_offline_X = df_offline[FEATS]
    # df_offline_y = df_offline[LABEL]
    # df_test_X = df_test[FEATS]
    # parameters previously tuned
    # bst = xgb.train({"n_gpus": 0,
    #                  "nthread": -1,
    #                  "learning_rate": 0.005,
    #                  "reg_lambda": 0.5},
    #                 xgb.DMatrix(df_offline_X, label=df_offline_y),
    #                 num_boost_round=50)
    # df_test["revenue"] = bst.predict(xgb.DMatrix(df_test_X))
    # df_test = df_test[["id", "revenue"]]
    # write_submission(df_test, "basic_feats_submission.csv")









