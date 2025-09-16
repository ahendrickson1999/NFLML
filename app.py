import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor
from packaging import version
import sklearn

TOP_N_FEATURES = 40

@st.cache_data(show_spinner=False)
def fetch_nfl_data(seasons):
    df = nfl.import_schedules(seasons)
    df = df.dropna(subset=['home_score', 'away_score'])
    return df

@st.cache_data(show_spinner=False)
def feature_engineering(df):
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
    # Use correct OneHotEncoder argument for your scikit-learn version
    if version.parse(sklearn.__version__) >= version.parse("1.2"):
        encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    else:
        encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
    encoder.fit(all_teams.reshape(-1, 1))
    home_team_enc = encoder.transform(df['home_team'].values.reshape(-1, 1))
    away_team_enc = encoder.transform(df['away_team'].values.reshape(-1, 1))

    df['home_advantage'] = 1
    df['div_game'] = df['div_game'].astype(int)
    game_type_enc = pd.get_dummies(df['game_type'], prefix='type')
    roof_enc = pd.get_dummies(df['roof'], prefix='roof')
    surface_enc = pd.get_dummies(df['surface'], prefix='surface')
    home_coach_enc = pd.get_dummies(df['home_coach'], prefix='hcoach')
    away_coach_enc = pd.get_dummies(df['away_coach'], prefix='acoach')
    home_qb_enc = pd.get_dummies(df['home_qb_name'], prefix='hq')
    away_qb_enc = pd.get_dummies(df['away_qb_name'], prefix='aq')
    location_enc = pd.get_dummies(df['location'], prefix='loc')

    for col in ['temp', 'wind', 'spread_line', 'total_line', 'away_moneyline', 'home_moneyline']:
        df[col] = df[col].fillna(df[col].mean() if df[col].dtype != 'O' else 0)
    df['away_rest'] = df['away_rest'].fillna(7)
    df['home_rest'] = df['home_rest'].fillna(7)

    team_stats, team_def_stats, team_win_stats = {}, {}, {}
    for team in all_teams:
        team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('gameday')
        team_games['team_score'] = np.where(team_games['home_team'] == team, team_games['home_score'], team_games['away_score'])
        team_games['team_score_avg'] = team_games['team_score'].rolling(window=3, min_periods=1).mean()
        team_games['opp_score'] = np.where(team_games['home_team'] == team, team_games['away_score'], team_games['home_score'])
        team_games['opp_score_avg'] = team_games['opp_score'].rolling(window=3, min_periods=1).mean()
        team_games['win'] = np.where(
            ((team_games['home_team'] == team) & (team_games['home_score'] > team_games['away_score'])) | 
            ((team_games['away_team'] == team) & (team_games['away_score'] > team_games['home_score'])), 1, 0
        )
        team_games['win_rate'] = team_games['win'].rolling(window=3, min_periods=1).mean()
        team_stats[team] = team_games.set_index('game_id')['team_score_avg']
        team_def_stats[team] = team_games.set_index('game_id')['opp_score_avg']
        team_win_stats[team] = team_games.set_index('game_id')['win_rate']

    df['home_team_avg'] = df.apply(lambda row: team_stats[row['home_team']].get(row['game_id'], row['home_score']), axis=1)
    df['away_team_avg'] = df.apply(lambda row: team_stats[row['away_team']].get(row['game_id'], row['away_score']), axis=1)
    df['home_team_def_avg'] = df.apply(lambda row: team_def_stats[row['home_team']].get(row['game_id'], row['away_score']), axis=1)
    df['away_team_def_avg'] = df.apply(lambda row: team_def_stats[row['away_team']].get(row['game_id'], row['home_score']), axis=1)
    df['home_team_win_rate'] = df.apply(lambda row: team_win_stats[row['home_team']].get(row['game_id'], 0.5), axis=1)
    df['away_team_win_rate'] = df.apply(lambda row: team_win_stats[row['away_team']].get(row['game_id'], 0.5), axis=1)

    X = pd.concat([
        pd.DataFrame(home_team_enc, index=df.index, columns=[f'home_{t}' for t in encoder.categories_[0]]),
        pd.DataFrame(away_team_enc, index=df.index, columns=[f'away_{t}' for t in encoder.categories_[0]]),
        game_type_enc,
        roof_enc,
        surface_enc,
        home_coach_enc,
        away_coach_enc,
        home_qb_enc,
        away_qb_enc,
        location_enc,
        df[['home_advantage', 'div_game', 'temp', 'wind', 'spread_line', 'total_line',
            'away_moneyline', 'home_moneyline', 'away_rest', 'home_rest',
            'home_team_avg', 'away_team_avg', 'home_team_def_avg', 'away_team_def_avg',
            'home_team_win_rate', 'away_team_win_rate']]
    ], axis=1)
    y_home = df['home_score']
    y_away = df['away_score']
    return X, y_home, y_away, encoder, all_teams

def select_features(X, y, top_n):
    rfr = RandomForestRegressor(n_estimators=100, random_state=42)
    rfr.fit(X, y)
    importances = pd.Series(rfr.feature_importances_, index=X.columns)
    selected_features = importances.nlargest(top_n).index.tolist()
    return selected_features

def train_models(X, y):
    models = [
        RandomForestRegressor(n_estimators=100, random_state=42),
        XGBRegressor(n_estimators=100, random_state=42),
        LinearRegression(),
        GradientBoostingRegressor(n_estimators=100, random_state=42),
        HistGradientBoostingRegressor(random_state=42),
        AdaBoostRegressor(n_estimators=100, random_state=42),
        Ridge(),
        Lasso(),
        ElasticNet(),
        KNeighborsRegressor(),
        SVR()
    ]
    for model in models:
        model.fit(X, y)
    return models

def ensemble_predict(models, X):
    preds = np.column_stack([model.predict(X) for model in models])
    return preds.mean(axis=1)

def build_features_for_matchup(home_team, away_team, encoder, df, all_possible_columns, team_avgs, team_def_avgs, team_win_avgs):
    last_row = df.iloc[-1]
    input_dict = {}

    # Team encoding
    home_enc = encoder.transform([[home_team]])[0]
    away_enc = encoder.transform([[away_team]])[0]
    for i, col in enumerate([f'home_{t}' for t in encoder.categories_[0]]):
        input_dict[col] = home_enc[i]
    for i, col in enumerate([f'away_{t}' for t in encoder.categories_[0]]):
        input_dict[col] = away_enc[i]

    # Categorical dummies: use mode or default
    for col in all_possible_columns:
        if col.startswith('type_'):
            input_dict[col] = 0
            if f'type_{last_row.get("game_type", "REG")}' == col:
                input_dict[col] = 1
        elif col.startswith('roof_'):
            input_dict[col] = 0
            if f'roof_{last_row.get("roof", "outdoors")}' == col:
                input_dict[col] = 1
        elif col.startswith('surface_'):
            input_dict[col] = 0
            if f'surface_{last_row.get("surface", "grass")}' == col:
                input_dict[col] = 1
        elif col.startswith('hcoach_'):
            input_dict[col] = 0
            if f'hcoach_{last_row.get("home_coach", "")}' == col:
                input_dict[col] = 1
        elif col.startswith('acoach_'):
            input_dict[col] = 0
            if f'acoach_{last_row.get("away_coach", "")}' == col:
                input_dict[col] = 1
        elif col.startswith('hq_'):
            input_dict[col] = 0
            if f'hq_{last_row.get("home_qb_name", "")}' == col:
                input_dict[col] = 1
        elif col.startswith('aq_'):
            input_dict[col] = 0
            if f'aq_{last_row.get("away_qb_name", "")}' == col:
                input_dict[col] = 1
        elif col.startswith('loc_'):
            input_dict[col] = 0
            if f'loc_{last_row.get("location", "")}' == col:
                input_dict[col] = 1

    # Numeric and engineered features
    input_dict['home_advantage'] = 1
    input_dict['div_game'] = 0
    input_dict['temp'] = last_row.get('temp', 60)
    input_dict['wind'] = last_row.get('wind', 5)
    input_dict['spread_line'] = last_row.get('spread_line', 0)
    input_dict['total_line'] = last_row.get('total_line', 45)
    input_dict['away_moneyline'] = last_row.get('away_moneyline', 0)
    input_dict['home_moneyline'] = last_row.get('home_moneyline', 0)
    input_dict['away_rest'] = last_row.get('away_rest', 7)
    input_dict['home_rest'] = last_row.get('home_rest', 7)

    input_dict['home_team_avg'] = team_avgs.get(home_team, last_row.get('home_team_avg', 21))
    input_dict['away_team_avg'] = team_avgs.get(away_team, last_row.get('away_team_avg', 21))
    input_dict['home_team_def_avg'] = team_def_avgs.get(home_team, last_row.get('home_team_def_avg', 21))
    input_dict['away_team_def_avg'] = team_def_avgs.get(away_team, last_row.get('away_team_def_avg', 21))
    input_dict['home_team_win_rate'] = team_win_avgs.get(home_team, last_row.get('home_team_win_rate', 0.5))
    input_dict['away_team_win_rate'] = team_win_avgs.get(away_team, last_row.get('away_team_win_rate', 0.5))

    X_pred = pd.DataFrame([input_dict])
    for col in all_possible_columns:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[all_possible_columns]
    return X_pred

st.title("NFL Score Predictor (Ensemble+Feature Selection)")

seasons = [2021, 2022, 2023, 2024]

with st.spinner("Loading and training... (first run may take a minute)"):
    df = fetch_nfl_data(seasons)
    X, y_home, y_away, encoder, all_teams = feature_engineering(df)
    all_possible_columns = X.columns.tolist()
    team_avgs, team_def_avgs, team_win_avgs = {}, {}, {}
    for team in all_teams:
        scores = pd.concat([df[df['home_team']==team]['home_score'], df[df['away_team']==team]['away_score']])
        opp_scores = pd.concat([df[df['home_team']==team]['away_score'], df[df['away_team']==team]['home_score']])
        wins = pd.concat([df[(df['home_team']==team) & (df['home_score']>df['away_score'])]['game_id'],
                          df[(df['away_team']==team) & (df['away_score']>df['home_score'])]['game_id']])
        team_avgs[team] = scores.mean()
        team_def_avgs[team] = opp_scores.mean()
        team_win_avgs[team] = len(wins) / len(scores) if len(scores) > 0 else 0.5

    selected_features_home = select_features(X, y_home, TOP_N_FEATURES)
    selected_features_away = select_features(X, y_away, TOP_N_FEATURES)
    X_home_selected = X[selected_features_home]
    X_away_selected = X[selected_features_away]

    home_models = train_models(X_home_selected, y_home)
    away_models = train_models(X_away_selected, y_away)

today = datetime.today().date()

schedule_df = nfl.import_schedules([today.year])
schedule_df['gameday'] = pd.to_datetime(schedule_df['gameday']).dt.date
todays_games = schedule_df[schedule_df['gameday'] == today]

if len(todays_games) == 0:
    st.info("No NFL games today.")
else:
    st.subheader(f"Predictions for today's NFL games ({today}):")
    pred_rows = []
    for i, row in todays_games.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        X_pred_full = build_features_for_matchup(
            home_team, away_team, encoder, df,
            all_possible_columns, team_avgs, team_def_avgs, team_win_avgs
        )
        X_home_pred = X_pred_full[selected_features_home]
        X_away_pred = X_pred_full[selected_features_away]
        home_pred = ensemble_predict(home_models, X_home_pred)
        away_pred = ensemble_predict(away_models, X_away_pred)
        pred_rows.append({
            "Away Team": away_team,
            "Home Team": home_team,
            "Predicted Away Score": round(float(away_pred[0]), 1),
            "Predicted Home Score": round(float(home_pred[0]), 1)
        })
    st.dataframe(pd.DataFrame(pred_rows))
    st.caption("Scores are model predictions based on recent data.")

st.markdown("---")
st.subheader("Manual Matchup Prediction")
team_list = sorted(list(all_teams))
home_team = st.selectbox("Select Home Team", team_list)
away_team = st.selectbox("Select Away Team", team_list, index=1 if team_list[1] != home_team else 2)

if home_team and away_team and home_team != away_team:
    X_pred_full = build_features_for_matchup(
        home_team, away_team, encoder, df,
        all_possible_columns, team_avgs, team_def_avgs, team_win_avgs
    )
    X_home_pred = X_pred_full[selected_features_home]
    X_away_pred = X_pred_full[selected_features_away]
    home_pred = ensemble_predict(home_models, X_home_pred)
    away_pred = ensemble_predict(away_models, X_away_pred)
    st.success(f"Prediction: {away_team} @ {home_team}: {round(float(away_pred[0]),1)} - {round(float(home_pred[0]),1)}")
elif home_team == away_team:
    st.warning("Select different teams!")

st.caption("Powered by nfl_data_py and scikit-learn.")