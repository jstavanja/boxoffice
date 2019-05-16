import numpy as np
import pandas as pd
import xgboost as xgb
from boxoffice_utils import fix_train_budget_revenue, fix_genres, fix_runtime, run_models, \
    write_submission, onehot_genres, onehot_original_language, fix_broken_json_values, add_important_cast_count, \
    prod_count_comp_lang_count, top_producer_director_writer

import json

if __name__ == "__main__":
    df_offline = pd.read_csv("data/train.csv", header=0)
    df_test = pd.read_csv("data/test.csv", header=0)
    df_offline = fix_train_budget_revenue(df_offline)  # type: pd.DataFrame
    df_offline = fix_genres(df_offline)
    df_offline = fix_runtime(df_offline)
    df_test = fix_genres(df_test)
    df_test = fix_runtime(df_test)

    FEATS = ["runtime", "budget", "popularity"]
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

    df_offline = fix_broken_json_values(df_offline, 'cast')

    # Turn variable-length genre information into fixed-size one-hot encoded attributes
    df_offline, genre_cols = onehot_genres(df_offline, genre_encoder)
    df_test, _ = onehot_genres(df_test, genre_encoder)
    FEATS.extend(genre_cols)

    df_offline = add_important_cast_count(df_offline)
    FEATS.extend(["important_cast_count"])

    """
    Features: 
        crew
        - top director
        - top screenplay writer
        - top producer
        spoken languages count
        production companies count
        production countries count
    """

    df_offline = fix_broken_json_values(df_offline, "crew")
    df_test = fix_broken_json_values(df_test, "crew")
    
    df_offline = top_producer_director_writer(df_offline, "data/top100directors.json", "Director")
    df_test = top_producer_director_writer(df_test, "data/top100directors.json", "Director")
    FEATS.extend(["top_crew_director"])
    
    df_offline = top_producer_director_writer(df_offline, "data/top100writers.json", "Screenplay")
    df_test = top_producer_director_writer(df_test, "data/top100writers.json", "Screenplay")
    FEATS.extend(["top_crew_screenplay"])
    
    df_offline = top_producer_director_writer(df_offline, "data/top_producers.json", "Producer")
    df_test = top_producer_director_writer(df_test, "data/top_producers.json", "Producer")
    FEATS.extend(["top_crew_producer"]) 

    df_offline = fix_broken_json_values(df_offline, "spoken_languages")
    df_test = fix_broken_json_values(df_test, "spoken_languages")
    df_offline = prod_count_comp_lang_count(df_offline, "spoken_languages")
    df_test = prod_count_comp_lang_count(df_test, "spoken_languages")
    FEATS.extend(["spoken_languages_count"]) 

    df_offline = fix_broken_json_values(df_offline, "production_companies")
    df_test = fix_broken_json_values(df_test, "production_companies")
    df_offline = prod_count_comp_lang_count(df_offline, "production_companies")
    df_test = prod_count_comp_lang_count(df_test, "production_companies")
    FEATS.extend(["production_companies_count"])

    df_offline = fix_broken_json_values(df_offline, "production_countries")
    df_test = fix_broken_json_values(df_test, "production_countries")
    df_offline = prod_count_comp_lang_count(df_offline, "production_countries")
    df_test = prod_count_comp_lang_count(df_test, "production_countries")
    FEATS.extend(["production_countries_count"]) 

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









