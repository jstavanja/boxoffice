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
    plt.text(norm_revenue - offset, 0.055, display_name, fontdict={"fontsize": 11})


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

    plt.figure(figsize=(8, 5))
    plt.title("Revenue distribution in offline dataset", fontdict={"fontsize": 18})
    plt.plot(uniqs, cnts, color="darkviolet")
    plt.xlabel("Revenue (in millions of \$)", fontdict={"fontsize": 14})
    plt.ylabel("Proportion of movies", fontdict={"fontsize": 14})
    plt.xticks(np.arange(0, np.max(inds) + 1, step=200), fontsize=14)
    plt.yticks(fontsize=14)

    landmark_movie("Space Jam", "Space Jam", 100)
    landmark_movie("Cars 2", "Cars 2", 75)
    landmark_movie("Deadpool", "Deadpool", 110)
    landmark_movie("The Dark Knight", " The\n Dark\nKnight", 70)
    landmark_movie("The Avengers", "???", 30)

    plt.savefig("img/revenue_distribution.png")
    # plt.show()

