import numpy as np
from sklearn.metrics import mean_squared_log_error


def rmsle(predicted, actual):
    return np.sqrt(mean_squared_log_error(y_true=actual, y_pred=predicted))


# makes submission file for Kaggle - assumes DataFrame `df` contains columns "id" and "revenue"
def write_submission(df, path):
    df.to_csv(path, index=False)
    print("Wrote submission to '{}'...".format(path))
