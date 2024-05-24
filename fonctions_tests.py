import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from datetime import datetime

def add_manager_win_percentage_before(df:pd.DataFrame):
    for index, row in df.iterrows():
        home_club_manager_name = row['home_club_manager_name']
        date = row['date']
        matches_won_at_home = df[(df['home_club_manager_name'] == home_club_manager_name) & (df['date'] < date)]['results'].value_counts().get(1, 0)
        total_matches_at_home = df[df['home_club_manager_name'] == home_club_manager_name].shape[0]

        ratio = matches_won_at_home / total_matches_at_home if total_matches_at_home != 0 else np.nan
        row['home_club_manager_win_percentage_before'] = ratio

        away_club_manager_name = row['away_club_manager_name']
        matches_won_away = df[(df['away_club_manager_name'] == away_club_manager_name) & (df['date'] < date)]['results'].value_counts().get(-1, 0)
        total_matches_away = df[df['away_club_manager_name'] == away_club_manager_name].shape[0]

        ratio = matches_won_away / total_matches_away if total_matches_at_home != 0 else np.nan
        row['away_club_manager_win_percentage_before'] = ratio

        # Update the row
        df.at[index, 'home_club_manager_win_percentage_before'] = row['home_club_manager_win_percentage_before']
        df.at[index, 'away_club_manager_win_percentage_before'] = row['away_club_manager_win_percentage_before']

def add_club_win_percentage_with_referee_before(df:pd.DataFrame):
    for index, row in df.iterrows():
        home_club_name = row['home_club_name']
        away_club_name = row['away_club_name']
        referee = row['referee']
        date = row['date']

        # Home club
        matches_won_with_referee = df[(df['home_club_name'] == home_club_name) & (df['referee'] == referee) & (df['date'] < date)]['results'].value_counts().get(1, 0)
        total_matches_with_referee = df[(df['home_club_name'] == home_club_name) & (df['referee'] == referee) & (df['date'] < date)].shape[0]

        matches_won_with_referee += df[(df['away_club_name'] == home_club_name) & (df['referee'] == referee) & (df['date'] < date)]['results'].value_counts().get(-1, 0)
        total_matches_with_referee += df[(df['away_club_name'] == home_club_name) & (df['referee'] == referee) & (df['date'] < date)].shape[0]

        ratio = matches_won_with_referee / total_matches_with_referee if total_matches_with_referee != 0 else np.nan
        row['home_club_win_percentage_with_referee_before'] = ratio

        # Away club
        matches_won_with_referee = df[(df['home_club_name'] == away_club_name) & (df['referee'] == referee) & (df['date'] < date)]['results'].value_counts().get(1, 0)
        total_matches_with_referee = df[(df['home_club_name'] == away_club_name) & (df['referee'] == referee) & (df['date'] < date)].shape[0]

        matches_won_with_referee += df[(df['away_club_name'] == away_club_name) & (df['referee'] == referee) & (df['date'] < date)]['results'].value_counts().get(-1, 0)
        total_matches_with_referee += df[(df['away_club_name'] == away_club_name) & (df['referee'] == referee) & (df['date'] < date)].shape[0]

        ratio = matches_won_with_referee / total_matches_with_referee if total_matches_with_referee != 0 else np.nan

        row['away_club_win_percentage_with_referee_before'] = ratio
        
        # Update the row# Update the row
        df.at[index, 'home_club_win_percentage_with_referee_before'] = row['home_club_win_percentage_with_referee_before']
        df.at[index, 'away_club_win_percentage_with_referee_before'] = row['away_club_win_percentage_with_referee_before']

playerValuation = None
playerAppearances = None
teamComps = None
dataTeams = None
playerAppearances = None
stats_players = None
teams_stats_updated = {}

def concatLists(liste):
    return list(set([item for sublist in liste for item in sublist]))
def get_score_player(id):
    global playerValuation
    playerValuation = pd.read_csv("data\player_valuation_before_season.csv", sep=",") if playerValuation is None else playerValuation
    player = playerValuation[playerValuation["player_id"] == id]["market_value_in_eur"].apply(lambda x: x/1e6)
    return player.mean() if len(player) > 0 else 0
def get_score_team(team):
    global teamComps
    if(teamComps is None):
        lineups = pd.read_csv("data\game_lineups.csv", sep=",")
        teamComps = lineups.groupby(['club_id'])['player_id'].apply(list).reset_index()
        # print(teamComps)
        teamComps = teamComps.groupby(['club_id'])['player_id'].apply(list).apply(concatLists).reset_index()
        teamComps["score"] = teamComps["player_id"].apply(lambda x: sum([get_score_player(i) for i in x]))
        
    return teamComps[teamComps["club_id"] == team]["score"].values[0]
def add_club_scores(X:pd.DataFrame):
    X["score_away_team"] = X["away_club_id"].apply(get_score_team)
    X["score_home_team"] = X["home_club_id"].apply(get_score_team)

def convert_string_to_thousands(s):
    # Remove the ? sign
    # get the sign of the value
    sign = 1 if s[0] != '-' else -1
    s = s.replace('-', '').replace('+', '')
    s = s[1:] if s[0] not in "1234567890" else s
    # print("process :", s)
    value = float(s[:-1] if len(s) > 1 else 0)
    multiplier = {'k': 1, 'm': 1000}.get(s[-1:], 1)
    # Return the numeric value multiplied by the multiplier
    return value * multiplier * sign
def add_price_players(data:pd.DataFrame):
    global dataTeams
    if(dataTeams is None):
        dataTeams = pd.read_csv("data\clubs_fr.csv", sep=",")
    dataTeams["transfer_num"] = dataTeams["net_transfer_record"].apply(convert_string_to_thousands)
    
    dataTemp = data.merge(dataTeams[['club_id', 'transfer_num']],
                                    left_on='home_club_id',
                                    right_on='club_id',
                                    how='left')#.rename(columns={'transfer_num': 'transfer_home_team'})
    data["transfer_home_team"] = dataTemp["transfer_num"]
    dataTemp = data.merge(dataTeams[['club_id', 'transfer_num']],
                                    left_on='away_club_id',
                                    right_on='club_id',
                                    how='left')#.rename(columns={'transfer_num': 'transfer_away_team'})
    data["transfer_away_team"] = dataTemp["transfer_num"]
def get_stats_all_players():
    global playerAppearances
    if(playerAppearances is None):
        playerAppearances = pd.read_csv("data\player_appearance.csv", sep=",")
    
    player_stats = playerAppearances.groupby('player_id').agg({
        'goals': ['mean', 'sum'],
        'assists': ['mean', 'sum'],
        'minutes_played': 'mean',
        'yellow_cards': ['mean', 'sum'],
        'red_cards': ['mean', 'sum'],
        'appearance_id': 'count'
    }).rename(columns={
        'goals': 'avg_goals_per_game total_goals',
        'assists': 'avg_assists_per_game total_assists',
        'minutes_played': 'avg_minutes_played_per_game',
        'yellow_cards': 'avg_yellow_cards_per_game total_yellow_cards',
        'red_cards': 'avg_red_cards_per_game total_red_cards',
        'appearance_id': 'total_games'
    })
    
    player_stats.columns = [col[0] for col in player_stats.columns.to_flat_index()]
    
    column_parts = [col.split() for col in player_stats.columns]
    new_column_names = []
    for i, parts in enumerate(column_parts):
        if (i > 0 and parts[0] == column_parts[i-1][0]):
            new_column_names.append(parts[1] if (len(parts) > 1) else parts[0])
        else:
            new_column_names.append(parts[0])
    player_stats.columns = new_column_names
    
    return player_stats
def get_list_players(team_id, date:datetime = None):
    global playerAppearances
    if(playerAppearances is None):
        playerAppearances = pd.read_csv("data\player_appearance.csv", sep=",")
    playerAppearances['date'] = pd.to_datetime(playerAppearances['date'])
    
    players_ids = playerAppearances[playerAppearances['date'] >= (date.strftime('%Y-%m-%d') if date != None else '2000-01-01')] [playerAppearances["player_current_club_id"] == team_id]["player_id"]
    players_ids = players_ids.drop_duplicates()
    return list(players_ids)

def get_players_team_stats(team_id):
    global stats_players
    if(stats_players is None):
        stats_players = get_stats_all_players()
    
    players_ids = get_list_players(team_id)
    # print(f'players_ids : {players_ids}')
    return stats_players[stats_players.index.isin(players_ids)]
def get_players_team_stats_updated(team_id):
    global teams_stats_updated
    if(team_id in teams_stats_updated):
        return teams_stats_updated[team_id]
    stats_players = get_players_team_stats(team_id)
    
    # The idea of this function is to give more distance between the values of the players stats and 
    def exp_log_fn(x, num_games):
        return np.exp(x * np.log(num_games)) * 100

    columns = ["avg_goals_per_game", "avg_assists_per_game","avg_yellow_cards_per_game","avg_red_cards_per_game"]
    stats_players[[f"{col}_updated" for col in columns]]  = stats_players[columns].apply(exp_log_fn, args = (stats_players["total_games"], ))
    stats_players.drop(columns=["avg_minutes_played_per_game"])
    teams_stats_updated[team_id] = stats_players
    return stats_players
def get_updated_stats_players_team_mean(team_id):
    stats_players = get_players_team_stats_updated(team_id)
    # print(stats_players)
    columns = ["avg_goals_per_game", "avg_assists_per_game","avg_yellow_cards_per_game","avg_red_cards_per_game"]
    # print(stats_players.columns)
    return stats_players[[f"{col}_updated" for col in columns]].mean() 
def add_updated_stats_players_team_mean(X:pd.DataFrame):
    columns = ["avg_goals_per_game", "avg_assists_per_game","avg_yellow_cards_per_game","avg_red_cards_per_game"]
    X[[f"home_club_{col}_updated" for col in columns]] = X["home_club_id"].apply(get_updated_stats_players_team_mean)
    X[[f"away_club_{col}_updated" for col in columns]] = X["away_club_id"].apply(get_updated_stats_players_team_mean)
    # X["away_team_players_stats_updated"] = X["away_club_id"].apply(get_updated_stats_players_team_mean)

def create_model(X_train, y_train):
    model = RandomForestClassifier()
    
    max_missing = 0.1 * len(X_train)
    missing_values = X_train.isna().sum()

    if all(missing_values < max_missing):
        imputer = SimpleImputer(strategy='mean')
    else:
        imputer = SimpleImputer(strategy='constant', fill_value=0)

    pipeline = Pipeline([
        ('imputer', imputer),
        ('model', model)
    ])
    
    pipeline.fit(X_train, y_train.values.ravel())
    return model    

def get_accuracy_with_model(model, X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model.fit(X_train, y_train.values.ravel())
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def test_data(X, y, model=None, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if model is None:
        model = create_model(X_train, y_train)
    return get_accuracy_with_model(model, X_train, X_test, y_train, y_test)

