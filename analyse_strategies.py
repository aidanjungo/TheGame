import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from strategies.strategy_simple import strategy_simple
from strategies.strategy_n_diff import strategy_n_diff
from thegame import set_TheGame,play_TheGame


def test_strategy(strategy, n_player_list, accetable_diff_list, n_game, display_output=True):
    
    results = {}
    cnt = 0
    tot = len(n_player_list) * len(accetable_diff_list) * n_game
    
    for n_player in n_player_list:
        for accetable_diff in accetable_diff_list:
            for i in range(n_game):
                # Set TheGame
                players, table = set_TheGame(n_player)

                # Play TheGame
                score = play_TheGame(players, table, strategy, accetable_diff, False)
                
                results[cnt] = {'n_player': n_player, 'accetable_diff': accetable_diff, 'score': score}
                cnt += 1
                if cnt % 20 == 0:
                    print(f'{cnt/tot*100:.2f}% done')
    
    df = pd.DataFrame.from_dict(results, orient='index')
    
    df.to_csv('results.csv', mode='a', index=False, header=False)
    


def show_results():
    
    n_player = 3
    
    df = pd.read_csv('results.csv')
    
    # df_gb_win = df[(df['n_player']==n_player) & (df['score']==0)].groupby(['accetable_diff']).count()
    # df_gb_win_tot = df[(df['n_player']==n_player)].groupby(['accetable_diff']).count()
    
    # df_gb_win['perc'] = df_gb_win['score'].div(df_gb_win_tot['score'])
    # print(df_gb_win)
    
    # df_gb_win['perc'].plot(kind='bar')
    # plt.ylabel('Win percentage')
    # plt.show()
    
    
    df_gb_diff = df[df['n_player']==n_player].groupby(['accetable_diff']).mean()
    print(df_gb_diff)
    # # Plot bars of mean scores vs accetable_diff
    # df_gb_diff['score'].plot(kind='bar')
    # plt.ylabel('Average score')
    # plt.ylim(20,25)
    # plt.show()
    
    print(df[(df['n_player']==n_player)].size)
    
    
    # df_gb_player = df[df['accetable_diff']==4].groupby(['n_player']).mean()
    # print(df_gb_player)
    # # Plot bars of mean scores vs accetable_diff
    # df_gb_player['score'].plot(kind='bar')
    # plt.show()
    
    # df.loc[(df['n_player']==2) & (df['accetable_diff']<6)].hist(column='score',bins=200)
    # plt.show()
    
    # df.heatmap(cmap='RdYlGn', annot=True)
    # plt.show()
    
    # Plot heatmap n_player vs accetable_diff
    df_gb_player = df.groupby(['n_player', 'accetable_diff']).mean()
    print(df_gb_player)
    df2 = df_gb_player.reset_index().pivot(columns='n_player',index='accetable_diff',values='score')
    # sns.heatmap(df2, annot=True, fmt=".2f", linewidths=.5, cmap='RdYlGn')
    # plt.title('Average score vs n_player and accetable_diff')
    # plt.show()
    
    # Plot heatmap n_player vs accetable_diff percentage of wins
    df_gb_win = df[df['score']==0].groupby(['n_player', 'accetable_diff']).count()
    df_gb_win_tot = df.groupby(['n_player', 'accetable_diff']).count()
    df_gb_win['perc'] = df_gb_win['score'].div(df_gb_win_tot['score'])
  
    print(df_gb_win)
    df3 = df_gb_win.reset_index().pivot(columns='n_player',index='accetable_diff',values='perc')
    # sns.heatmap(df3, annot=True, fmt='.2%', linewidths=.5, cmap='RdYlGn')
    # plt.title('Winning probability vs n_player and accetable_diff')
    # plt.show()

    print(df.describe())
    
    df_conv = df[(df['n_player']==3) & (df['accetable_diff']==4)].reset_index()
    df_conv['cumu_avg'] = df_conv['score'].expanding().mean()
    df_conv[['cumu_avg']].plot()
    plt.ylim(22.3,23.3)
    plt.show()
    

if __name__ == "__main__":
    
    
    # Options
    display_output = True
    n_player = [3]
    #strategy = 'strategy_simple'
    strategy = 'strategy_n_diff'
    accetable_diff = [4]
    n_game = 100000
    
    test_strategy(strategy, n_player, accetable_diff, n_game)
    
    show_results()
    
    
    
                   
