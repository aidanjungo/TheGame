import csv
import os
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from thegame import set_TheGame, play_TheGame


def test_strategy(strategy, n_player_list, acceptable_diff_list, n_game):

    results = {}
    cnt = 0
    tot = len(n_player_list) * len(acceptable_diff_list) * n_game

    for n_player in n_player_list:
        for acceptable_diff in acceptable_diff_list:
            for _ in range(n_game):

                # Set TheGame
                players, table = set_TheGame(n_player)

                # Play TheGame
                score = play_TheGame(players, table, strategy, acceptable_diff, False)

                results[cnt] = {
                    "n_player": n_player,
                    "acceptable_diff": acceptable_diff,
                    "score": score,
                }
                cnt += 1
                if cnt % 20 == 0:
                    print(f"{cnt/tot*100:.2f}% done")

    df = pd.DataFrame.from_dict(results, orient="index")

    # Check if result file exist
    if not os.path.exists("results.csv"):
        with open("results.csv", "w") as f:
            writer = csv.writer(f)
            writer.writerow(["n_player", "acceptable_diff", "score"])

    df.to_csv("results.csv", mode="a", index=False, header=False)


def show_results():

    df = pd.read_csv("results.csv")

    #  Plot heatmap Average scores vs n_player and acceptable_diff
    df_gb_player = df.groupby(["n_player", "acceptable_diff"]).mean()
    df2 = df_gb_player.reset_index().pivot(
        columns="n_player", index="acceptable_diff", values="score"
    )
    sns.heatmap(df2, annot=True, fmt=".2f", linewidths=0.5, cmap="RdYlGn")
    plt.title("Average score vs n_player and acceptable_diff")
    plt.show()

    # Plot heatmap winning probability vs n_player and acceptable_diff
    df_gb_win = df[df["score"] == 0].groupby(["n_player", "acceptable_diff"]).count()
    df_gb_win_tot = df.groupby(["n_player", "acceptable_diff"]).count()
    df_gb_win["perc"] = df_gb_win["score"].div(df_gb_win_tot["score"])

    print(df_gb_win)
    df3 = df_gb_win.reset_index().pivot(
        columns="n_player", index="acceptable_diff", values="perc"
    )
    sns.heatmap(df3, annot=True, fmt=".2%", linewidths=0.5, cmap="RdYlGn")
    plt.title("Winning probability vs n_player and acceptable_diff")
    plt.show()

    # Cumulative average
    # df_conv = df[(df["n_player"] == 3) & (df["acceptable_diff"] == 4)].reset_index()
    # df_conv["cumu_avg"] = df_conv["score"].expanding().mean()
    # df_conv[["cumu_avg"]].plot()
    # plt.ylim(22.3, 23.3)
    # plt.show()


if __name__ == "__main__":

    # Options
    n_player = [1, 2, 3, 4, 5]
    strategy = "strategy_n_diff"
    acceptable_diff = [2, 4, 6, 8, 10]
    n_game = 1000

    test_strategy(strategy, n_player, acceptable_diff, n_game)

    show_results()
