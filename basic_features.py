import pandas as pd
from boxoffice_utils import fix_train_budget_revenue, fix_genres, fix_runtime, run_models

if __name__ == "__main__":
    df_offline = pd.read_csv("data/train.csv", header=0)
    df_test = pd.read_csv("data/test.csv", header=0)
    df_offline = fix_train_budget_revenue(df_offline)  # type: pd.DataFrame
    df_offline = fix_genres(df_offline)
    df_offline = fix_runtime(df_offline)

    # TODO: turn some other basic features into numbers (e.g. using dummy variables) and use them
    FEATS = ["runtime"]
    LABEL = ["revenue"]

    model_props = {"knn": {},
                   "ridge": {},
                   "svr": {},
                   "rf": {},
                   "xgb": {}}
    run_models(df_offline, FEATS, LABEL,
               models=model_props,
               train_prop=0.7,
               val_prop=0.1,
               test_prop=0.2,
               seed=1337)









