from urllib.request import urlopen
from bs4 import BeautifulSoup
import requests
import pandas as pd


stats_for_each_year = []
years = list(range(1998, 2021+1))
for year in years:
    print("Year:", year)

    url_totals = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_totals.html"
    url_per_game = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_game.html"
    url_per_36_min = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_minute.html"
    url_per_100_poss = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_per_poss.html"
    url_advanced = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_advanced.html"
    url_shooting = "https://www.basketball-reference.com/leagues/NBA_" + str(year) + "_shooting.html"
    # urls = [url_totals, url_per_game, url_per_36_min, url_per_100_poss, url_advanced]
    urls = [url_totals, url_per_game, url_per_36_min, url_per_100_poss, url_advanced, url_shooting]

    url_index = 0
    for url in urls:
        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')

        all_tr = soup.findAll("tr")
        all_tr = list(filter(lambda ele: not ele.find(class_='over_header'), all_tr))

        header_data = all_tr[0]
        players_data = []
        player_ids = []
        for i in range(0, len(all_tr)):
            # if all_tr[i].get("class") == ['full_table'] or all_tr[i].get("class") == ['italic_text', 'partial_table']:
            # ^ The above if statement can be used if we care about partial seasons
            if all_tr[i].get("class") == ['full_table']:
                players_data.append(all_tr[i])
                player_id = all_tr[i].findAll('td')[0].get("data-append-csv")
                player_ids.append(player_id)

        # I'm getting rid of the Rk column below with [1:]:
        headers = ["playerID"] + [th.getText() for th in header_data.findAll('th')][1:]
        players_stats = [[player_ids[i]] + [td.getText() for td in players_data[i].findAll('td')] \
                         for i in range(len(players_data))]

        if url_index == 0:
            stats_totals = pd.DataFrame(players_stats, columns=headers)
        elif url_index == 1:
            stats_per_game = pd.DataFrame(players_stats, columns=headers)
        elif url_index == 2:
            stats_per_36_min = pd.DataFrame(players_stats, columns=headers)
        elif url_index == 3:
            stats_per_100_poss = pd.DataFrame(players_stats, columns=headers)
        elif url_index == 4:
            stats_advanced = pd.DataFrame(players_stats, columns=headers)
        else:
            stats_shooting = pd.DataFrame(players_stats, columns=headers)

            # Join the dataframes here
            all_player_stats = stats_totals.join(stats_per_game, lsuffix='_totals')\
                .join(stats_per_36_min, lsuffix='_per_game')\
                .join(stats_per_100_poss, lsuffix='_per_36_min', rsuffix='_per_100_poss')\
                .join(stats_advanced, rsuffix='_advanced')\
                .join(stats_shooting, rsuffix='_shooting')
            # Drop extraneous columns:
            all_player_stats = all_player_stats \
                .drop(columns=['playerID_totals', 'playerID_per_game', 'playerID_per_36_min', 'playerID_per_100_poss',
                               'playerID_shooting']) \
                .drop(columns=['Player_totals', 'Player_per_game', 'Player_per_36_min', 'Player_per_100_poss',
                               'Player_shooting']) \
                .drop(columns=['Pos_totals', 'Pos_per_game', 'Pos_per_36_min', 'Pos_per_100_poss', 'Pos_shooting']) \
                .drop(columns=['Age_totals', 'Age_per_game', 'Age_per_36_min', 'Age_per_100_poss', 'Age_shooting']) \
                .drop(columns=['Tm_totals', 'Tm_per_game', 'Tm_per_36_min', 'Tm_per_100_poss', 'Tm_shooting']) \
                .drop(columns=['G_totals', 'G_per_game', 'G_per_36_min', 'G_per_100_poss', 'G_shooting']) \
                .drop(columns=['GS_totals', 'GS_per_game', 'GS_per_36_min', 'GS_per_100_poss'])\
                .drop(columns=['\xa0_shooting','','\xa0'])
                # .drop(columns=['playerID_totals', 'playerID_per_game', 'playerID_per_36_min', 'playerID_per_100_poss']) \
                # .drop(columns=['Player_totals', 'Player_per_game', 'Player_per_36_min', 'Player_per_100_poss']) \
                # .drop(columns=['Pos_totals', 'Pos_per_game', 'Pos_per_36_min', 'Pos_per_100_poss']) \
                # .drop(columns=['Age_totals', 'Age_per_game', 'Age_per_36_min', 'Age_per_100_poss']) \
                # .drop(columns=['Tm_totals', 'Tm_per_game', 'Tm_per_36_min', 'Tm_per_100_poss']) \
                # .drop(columns=['G_totals', 'G_per_game', 'G_per_36_min', 'G_per_100_poss']) \
                # .drop(columns=['GS_totals', 'GS_per_game', 'GS_per_36_min', 'GS_per_100_poss'])

        url_index += 1

    pd.set_option("display.max_rows", 10, "display.max_columns", None)
    stats_for_each_year.append(all_player_stats)
    # print(all_player_stats.head(10))

years_stats_dict = {}
for year_index in range(0, len(years)):
    years_stats_dict[years[year_index]] = stats_for_each_year[year_index]
    stats_for_each_year[year_index].to_pickle("./data/player_stats_" + str(years[year_index]) + ".pkl")

print(years_stats_dict)

