import pandas as pd
from boxoffice_utils import fix_train_budget_revenue, fix_genres, fix_runtime, onehot_genres
import matplotlib.pyplot as plt
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

    # Turn variable-length genre information into fixed-size one-hot encoded attributes
    df_offline, genre_cols = onehot_genres(df_offline, genre_encoder)

    tmp = ["Documentary", "Animation", "Romance", "Adventure", "Science Fiction", "Western"]
    median_revenue = []

    for col in tmp:
        valid_examples = df_offline[df_offline[col] == 1]
        m_rev = valid_examples["revenue"].median()
        print("Median revenue for {}: {}, obtained on {} examples".format(col,
                                                                          m_rev,
                                                                          len(valid_examples)))
        median_revenue.append(m_rev / 1_000_000)

    sort_indices = sorted(range(len(tmp)), key=lambda idx: median_revenue[idx], reverse=True)

    tmp = ["Documentary", "Animation", "Romance", "Adventure", "Science\nFiction", "Western"]
    tmp_sorted = [tmp[idx] for idx in sort_indices]
    median_sorted = [median_revenue[idx] for idx in sort_indices]

    plt.figure(figsize=(10, 6))
    plt.title("Median revenue for different genres", fontdict={"fontsize": 18})
    plt.bar(tmp_sorted, median_sorted, color="midnightblue")
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.ylabel("Median revenue (in millions of $)", fontdict={"fontsize": 18})
    # plt.savefig("img/median_revenue_genres.png")
    plt.show()



