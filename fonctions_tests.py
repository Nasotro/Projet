import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

def add_manager_win_percentage(df:pd.DataFrame):
    manager_home_win_percentage = {}

    for home_club_manager_name in df['home_club_manager_name'].unique():
        matches_won_at_home = df[df['home_club_manager_name'] == home_club_manager_name]['results'].value_counts().get(1, 0)
        total_matches_at_home = df[df['home_club_manager_name'] == home_club_manager_name].shape[0]

        ratio = matches_won_at_home / total_matches_at_home
        manager_home_win_percentage[home_club_manager_name] = ratio

    manager_away_win_percentage = {}

    for away_club_manager_name in df['away_club_manager_name'].unique():
        matches_won_away = df[df['away_club_manager_name'] == away_club_manager_name]['results'].value_counts().get(-1, 0)
        total_matches_away = df[df['away_club_manager_name'] == away_club_manager_name].shape[0]

        ratio = matches_won_away / total_matches_away
        manager_away_win_percentage[away_club_manager_name] = ratio
        
    df['home_club_manager_win_percentage'] = df['home_club_manager_name'].map(manager_home_win_percentage)
    df['away_club_manager_win_percentage'] = df['away_club_manager_name'].map(manager_away_win_percentage)

def add_club_win_percentage_with_referee(df:pd.DataFrame):
    win_percentage_with_referee = {}

    for referee in df['referee'].unique():
        for home_club_name in pd.concat([df['home_club_name'],(df['away_club_name'])]).unique():
            matches_won_with_referee = df[(df['home_club_name'] == home_club_name) & (df['referee'] == referee)]['results'].value_counts().get(1, 0)
            total_matches_with_referee = df[(df['home_club_name'] == home_club_name) & (df['referee'] == referee)].shape[0]

            matches_won_with_referee += df[(df['away_club_name'] == home_club_name) & (df['referee'] == referee)]['results'].value_counts().get(-1, 0)
            total_matches_with_referee += df[(df['away_club_name'] == home_club_name) & (df['referee'] == referee)].shape[0]

            ratio = matches_won_with_referee / total_matches_with_referee if total_matches_with_referee != 0 else np.nan
            win_percentage_with_referee[(home_club_name, referee)] = ratio
    
    df['home_club_win_percentage_with_referee'] = df.apply(lambda row: win_percentage_with_referee.get((row['home_club_name'], row['referee']), np.nan), axis=1)
    df['away_club_win_percentage_with_referee'] = df.apply(lambda row: win_percentage_with_referee.get((row['away_club_name'], row['referee']), np.nan), axis=1)

playerValuation = None
playerAppearances = None
def concatLits(liste):
    return list(set([item for sublist in liste for item in sublist]))
def get_score_player(id):
    global playerValuation
    playerValuation = pd.read_csv("data\player_valuation_before_season.csv", sep=",") if playerValuation is None else playerValuation
    player = playerValuation[playerValuation["player_id"] == id]["market_value_in_eur"].apply(lambda x: x/1e6)
    return player.mean() if len(player) > 0 else 0
teamComps = None
def get_score_team(team):
    global teamComps
    if(teamComps is None):
        lineups = pd.read_csv("data\game_lineups.csv", sep=",")
        teamComps = lineups.groupby(['club_id'])['player_id'].apply(list).reset_index()
        # print(teamComps)
        teamComps = teamComps.groupby(['club_id'])['player_id'].apply(list).apply(concatLits).reset_index()
        teamComps["score"] = teamComps["player_id"].apply(lambda x: sum([get_score_player(i) for i in x]))
        
    return teamComps[teamComps["club_id"] == team]["score"].values[0]
def add_club_scores(X:pd.DataFrame):
    X["score_away_team"] = X["away_club_id"].apply(get_score_team)
    X["score_home_team"] = X["home_club_id"].apply(get_score_team)

 
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

