import pandas as pd
from boxoffice_utils import fix_train_budget_revenue
import numpy as np
import matplotlib.pyplot as plt

# Plot in millions of dollars
STEP = 1_000_000


def landmark_movie(original_name, display_name, offset):
    norm_revenue = df_offline[df_offline["original_title"] == original_name]["revenue"].values / STEP

    plt.plot([norm_revenue, norm_revenue], [0.0, 0.05],
             '--', color="midnightblue")
    plt.plot(norm_revenue, 0.05, "^", color="midnightblue")
    plt.text(norm_revenue - offset, 0.055, display_name)


if __name__ == "__main__":
    df_offline = pd.read_csv("data/train.csv", header=0)
    df_offline = fix_train_budget_revenue(df_offline)  # type: pd.DataFrame

    revenues = df_offline["revenue"].values
    print("Mean: {}".format(np.mean(revenues)))
    print("Median: {}".format(np.median(revenues)))
    print("75th percentile: {}".format(np.percentile(revenues, 75)))
    # revenues = np.log10(revenues)
    bins = np.arange(0, revenues.max() + STEP, step=STEP)
    inds = np.digitize(revenues, bins)

    uniqs, cnts = np.unique(inds, return_counts=True)
    cnts = cnts / np.sum(cnts)

    plt.plot(uniqs, cnts, color="darkviolet")
    plt.xlabel("Revenue (in millions of \$)")
    plt.ylabel("Proportion of movies")

    tmp = df_offline[(1100000000 < df_offline["revenue"]) & (df_offline["revenue"] < 1400000000)]
    print(tmp["original_title"])

    landmark_movie("Space Jam", "Space Jam", 100)
    landmark_movie("Cars 2", "Cars 2", 75)
    landmark_movie("Deadpool", "Deadpool", 100)
    landmark_movie("The Dark Knight", "The Dark Knight", 100)
    landmark_movie("The Avengers", "The\nAvengers", 100)

    plt.show()

