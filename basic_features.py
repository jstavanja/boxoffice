import pandas as pd
from boxoffice_utils import fix_train_budget_revenue, fix_genres, fix_runtime, \
    train_val_test_split, rmsle, tune_knn, tune_ridge, tune_svr, tune_rf
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import Ridge
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor

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

    # ------------------------------------------
    # k-nearest neighbors regression
    # ------------------------------------------
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

    # ------------------------------------------
    # ridge regression
    # ------------------------------------------
    best_alpha, best_err = tune_ridge(train_X, train_y, val_X, val_y,
                                      alpha_params=[0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1,
                                                    0.2, 0.5, 1.0, 2.0, 5.0, 10.0],
                                      seed=1337)

    print("Winning parameters for Ridge regression: alpha={:.3f}... "
          "Obtained error: {:.5f}".format(best_alpha, best_err))
    ridge = Ridge(alpha=best_alpha,
                  random_state=1337)
    ridge.fit(pd.concat([train_X, val_X], ignore_index=True),
              pd.concat([train_y, val_y], ignore_index=True))
    offline_test_preds = ridge.predict(test_X)
    offline_error = rmsle(offline_test_preds, test_y)
    print("[Ridge] Offline RMSLE: {:.5f}".format(offline_error))

    # ------------------------------------------
    # support vector regression
    # ------------------------------------------
    best_c, best_eps, best_err = tune_svr(train_X, train_y, val_X, val_y,
                                          c_params=[0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5,
                                                    1.0, 2.0, 5.0, 10.0],
                                          eps_params=[0.0, 0.01, 0.02, 0.1, 0.2, 0.5,
                                                      1.0, 2.0, 5.0])

    print("Winning parameters for SVM regression: c={:.3f}, eps={:.2f}... "
          "Obtained error: {:.5f}".format(best_c, best_eps, best_err))
    svr = SVR(C=best_c,
              epsilon=best_eps,
              gamma="scale")
    svr.fit(pd.concat([train_X, val_X], ignore_index=True),
            pd.concat([train_y, val_y], ignore_index=True))
    offline_test_preds = svr.predict(test_X)
    offline_error = rmsle(offline_test_preds, test_y)
    print("[SVR] Offline RMSLE: {:.5f}".format(offline_error))

    # ------------------------------------------
    # random forest regression
    # ------------------------------------------
    best_n_trees, best_err = tune_rf(train_X, train_y, val_X, val_y,
                                     n_estimators_param=[50, 100, 200, 500, 1000, 2000, 5000],
                                     seed=1337)

    print("Winning parameters for RF regression: n_estimators={}... "
          "Obtained error: {:.5f}".format(best_n_trees, best_err))
    rf = RandomForestRegressor(n_estimators=best_n_trees,
                               criterion="mae",
                               random_state=1337,
                               n_jobs=-1)
    rf.fit(pd.concat([train_X, val_X], ignore_index=True),
           pd.concat([train_y, val_y], ignore_index=True))
    offline_test_preds = rf.predict(test_X)
    offline_error = rmsle(offline_test_preds, test_y)
    print("[RF regression] Offline RMSLE: {:.5f}".format(offline_error))











