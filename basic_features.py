import pandas as pd
from boxoffice_utils import fix_train_budget_revenue, fix_genres, fix_runtime, \
    train_val_test_split, rmsle, tune_knn
from sklearn.neighbors import KNeighborsRegressor

if __name__ == "__main__":
    df_offline = pd.read_csv("data/train.csv", header=0)
    df_test = pd.read_csv("data/test.csv", header=0)
    df_offline = fix_train_budget_revenue(df_offline)  # type: pd.DataFrame
    df_offline = fix_genres(df_offline)
    df_offline = fix_runtime(df_offline)

    # TODO: turn some other basic features into numbers (e.g. using dummy variables) and use them
    FEATS = ["runtime"]
    LABEL = ["revenue"]
    train, val, test = train_val_test_split(df_offline, train_prop=0.7, val_prop=0.1,
                                            test_prop=0.2, seed=1337)
    train_X, train_y = train[FEATS], train[LABEL]
    val_X, val_y = val[FEATS], val[LABEL]
    test_X, test_y = test[FEATS], test[LABEL]
    print("{} examples in training, {} in validation and {} in test set...".format(train.shape[0],
                                                                                   val.shape[0],
                                                                                   test.shape[0]))

    best_k, best_weights, best_err = tune_knn(train_X, train_y, val_X, val_y,
                                              neigh_params=[1, 3, 5, 10, 20],
                                              weight_params=["uniform", "distance"])

    print("Winning parameters for KNN: k={:d}, weights='{:s}'... "
          "Obtained error: {:.5f}".format(best_k, best_weights, best_err))
    knn = KNeighborsRegressor(n_neighbors=best_k,
                              weights=best_weights,
                              n_jobs=-1)
    knn.fit(pd.concat([train_X, val_X], ignore_index=True),
            pd.concat([train_y, val_y], ignore_index=True))
    offline_test_preds = knn.predict(test_X)
    offline_error = rmsle(offline_test_preds, test_y)
    print("[KNN] Offline RMSLE: {:.5f}".format(offline_error))















