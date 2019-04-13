import pandas as pd
from boxoffice_utils import write_submission, rmsle
from sklearn.model_selection import train_test_split


if __name__ == "__main__":
    df_offline = pd.read_csv("data/train.csv", header=0)
    mean_train_revenue = df_offline["revenue"].mean()

    df_offline_train, df_offline_test = train_test_split(df_offline,
                                                         test_size=0.3,
                                                         random_state=1337)

    df_offline_test["pred"] = df_offline_train["revenue"].mean()
    print("Offline RMSLE: %.5f" % rmsle(df_offline_test["pred"], df_offline_test["revenue"]))

    df_test = pd.read_csv("data/test.csv", header=0)
    # use entire training set for "training"
    df_test["revenue"] = mean_train_revenue
    df_test = df_test[["id", "revenue"]]
    write_submission(df_test, "baseline_submission.csv")
