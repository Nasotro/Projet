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

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

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
    
    pipeline.fit(X_train, y_train)
    return model    

def get_accuracy_with_model(model, X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    return accuracy_score(y_test, y_pred)

def test_data(X, y, model=None, test_size=0.2):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)
    if model is None:
        model = create_model(X_train, y_train)
    return get_accuracy_with_model(model, X_train, X_test, y_train, y_test)