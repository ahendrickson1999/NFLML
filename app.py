import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

st.set_page_config(page_title="NFL Game Predictor", layout="wide")

st.title("🏈 NFL Game Predictor: Winner & Total Points")
st.write("This app uses real NFL data and advanced machine learning (XGBoost/Random Forest) to predict the winner and total points of NFL games. Powered by [nfl_data_py](https://github.com/nflverse/nfl_data_py).")

@st.cache_data(show_spinner=True)
def load_data(years):
    games = nfl.import_schedules(years)
    pbp = nfl.import_pbp_data(years)
    teams = nfl.import_team_desc()
    return games, pbp, teams

# --- Feature Engineering ---
def build_features(games, pbp):
    # Calculate recent team stats (last 5 games rolling average)
    games_sorted = games.sort_values(['team', 'season', 'week'])
    for stat in ['points', 'total_yards', 'turnovers']:
        games_sorted[f'{stat}_rolling5'] = games_sorted.groupby('team')[stat].rolling(5, min_periods=1).mean().reset_index(0,drop=True)
    # Create matchup dataframe
    matchups = []
    for _, row in games.iterrows():
        # Home/Away team rolling stats
        home = games_sorted[(games_sorted['team']==row['home_team']) & (games_sorted['game_id']!=row['game_id']) & (games_sorted['game_date']<row['game_date'])]
        away = games_sorted[(games_sorted['team']==row['away_team']) & (games_sorted['game_id']!=row['game_id']) & (games_sorted['game_date']<row['game_date'])]
        home_last = home.iloc[-1] if not home.empty else {}
        away_last = away.iloc[-1] if not away.empty else {}
        matchups.append({
            'game_id': row['game_id'],
            'season': row['season'],
            'week': row['week'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_points_avg': home_last.get('points_rolling5', 0),
            'away_points_avg': away_last.get('points_rolling5', 0),
            'home_yards_avg': home_last.get('total_yards_rolling5', 0),
            'away_yards_avg': away_last.get('total_yards_rolling5', 0),
            'home_tov_avg': home_last.get('turnovers_rolling5', 0),
            'away_tov_avg': away_last.get('turnovers_rolling5', 0),
            'home_rest': (pd.to_datetime(row['game_date']) - pd.to_datetime(home_last.get('game_date','1970-01-01'))).days if home_last.get('game_date') else 7,
            'away_rest': (pd.to_datetime(row['game_date']) - pd.to_datetime(away_last.get('game_date','1970-01-01'))).days if away_last.get('game_date') else 7,
            'home_is_favorite': int(row.get('spread_line',0)<0 if 'spread_line' in row else 0),
            'spread': abs(row.get('spread_line',0)) if 'spread_line' in row else 0,
            'over_under': row.get('total_line', 0) if 'total_line' in row else 0,
            'home_score': row['home_score'],
            'away_score': row['away_score']
        })
    df = pd.DataFrame(matchups)
    df = df.dropna(subset=['home_score','away_score'])
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['total_points'] = df['home_score'] + df['away_score']
    # Drop games with missing stats
    return df

# --- Model Training ---
@st.cache_resource(show_spinner=True)
def train_models(df):
    feature_cols = [
        'home_points_avg','away_points_avg','home_yards_avg','away_yards_avg',
        'home_tov_avg','away_tov_avg','home_rest','away_rest','home_is_favorite',
        'spread','over_under'
    ]
    X = df[feature_cols]
    y_cls = df['home_win']
    y_reg = df['total_points']
    X_train, X_test, y_cls_train, y_cls_test, y_reg_train, y_reg_test = train_test_split(
        X, y_cls, y_reg, test_size=0.15, random_state=42
    )
    # Winner model
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    clf.fit(X_train, y_cls_train)
    # Points model
    reg = XGBRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
    reg.fit(X_train, y_reg_train)
    # Validation
    acc = accuracy_score(y_cls_test, clf.predict(X_test))
    mae = mean_absolute_error(y_reg_test, reg.predict(X_test))
    return clf, reg, feature_cols, acc, mae

# --- App Logic ---
years = st.sidebar.multiselect("Select NFL seasons to use for training:", list(range(2010, datetime.now().year+1)), default=[2023,2024])
games, pbp, teams = load_data(years)
st.info("Loaded {} games, {} play-by-play rows.".format(len(games), len(pbp)))
with st.spinner("Building features and training models..."):
    features = build_features(games, pbp)
    clf, reg, feature_cols, acc, mae = train_models(features)
st.success(f"Winner accuracy (holdout): {acc:.2%}. Total points MAE: {mae:.2f}")

# --- User Prediction Form ---
st.header("Predict a Matchup")
team_list = sorted(features['home_team'].unique())
col1, col2 = st.columns(2)
with col1:
    home_team = st.selectbox("Home Team", team_list, index=0)
with col2:
    away_team = st.selectbox("Away Team", [t for t in team_list if t != home_team], index=1)
spread = st.number_input("Vegas Spread (home - points):", value=0.0)
over_under = st.number_input("Vegas Over/Under:", value=44.0)

# Get latest rolling stats for selected teams
def get_last_stats(team, df):
    team_games = df[(df['home_team']==team) | (df['away_team']==team)]
    if team_games.empty:
        return 0,0,0
    last = team_games.iloc[-1]
    if last['home_team']==team:
        return last['home_points_avg'], last['home_yards_avg'], last['home_tov_avg'], last['home_rest']
    else:
        return last['away_points_avg'], last['away_yards_avg'], last['away_tov_avg'], last['away_rest']

if st.button("Predict!"):
    home_pts, home_yds, home_tov, home_rest = get_last_stats(home_team, features)
    away_pts, away_yds, away_tov, away_rest = get_last_stats(away_team, features)
    fav = int(spread < 0)
    input_df = pd.DataFrame([{
        'home_points_avg': home_pts, 'away_points_avg': away_pts,
        'home_yards_avg': home_yds, 'away_yards_avg': away_yds,
        'home_tov_avg': home_tov, 'away_tov_avg': away_tov,
        'home_rest': home_rest, 'away_rest': away_rest,
        'home_is_favorite': fav, 'spread': abs(spread), 'over_under': over_under
    }])
    win_prob = clf.predict_proba(input_df)[0,1]
    total_pts = reg.predict(input_df)[0]
    st.markdown(f"### 🏆 Winner Prediction: **{'Home' if win_prob>=0.5 else 'Away'} Team** ({home_team if win_prob>=0.5 else away_team})")
    st.markdown(f"- Probability Home Wins: **{win_prob:.1%}**")
    st.markdown(f"- Predicted Total Points: **{total_pts:.1f}**")
    st.markdown(f"- Vegas Spread: {spread}, Over/Under: {over_under}")
    st.caption("Features used: recent rolling averages, rest days, Vegas lines, home/away.")

st.markdown("---")
st.caption("Author: Your Name | Data from nfl_data_py | Models: XGBoost")
