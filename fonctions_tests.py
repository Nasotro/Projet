def add_manager_win_percentage(df):

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