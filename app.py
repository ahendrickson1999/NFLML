import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime, timedelta
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_absolute_error

st.set_page_config(page_title="NFL Game Predictor", layout="wide")

st.title("🏈 NFL Game Predictor: Winner & Total Points (Full Featured)")
st.write(
    "This app uses real NFL data and advanced machine learning (XGBoost) to predict the winner and total points of NFL games. Powered by [nfl_data_py](https://github.com/nflverse/nfl_data_py)."
)

@st.cache_data(show_spinner=True)
def load_data(years):
    # Load schedule (games), play-by-play, and team descriptions
    games = nfl.import_schedules(years)
    pbp = nfl.import_pbp_data(years)
    teams = nfl.import_team_desc()
    return games, pbp, teams

def add_vegas_lines(games):
    # Fill spread_line and total_line with zeros if missing for simplicity
    if 'spread_line' not in games.columns:
        games['spread_line'] = 0.0
    if 'total_line' not in games.columns:
        games['total_line'] = 44.0
    games['spread_line'] = games['spread_line'].fillna(0.0)
    games['total_line'] = games['total_line'].fillna(44.0)
    return games

def compute_team_game_stats(pbp):
    # Aggregate play-by-play data for core stats per team per game
    pbp = pbp[~pbp['season_type'].isin(['PRE'])]
    pbp = pbp[pbp['posteam'].notnull() & pbp['defteam'].notnull()]
    agg_stats = pbp.groupby(['game_id', 'posteam']).agg(
        points_scored=('touchdown', 'sum'),
        pass_yards=('passing_yards', 'sum'),
        rush_yards=('rushing_yards', 'sum'),
        turnovers=('interception', 'sum'),
        fumbles=('fumble_lost', 'sum')
    ).reset_index()
    agg_stats['total_yards'] = agg_stats['pass_yards'] + agg_stats['rush_yards']
    agg_stats['turnovers'] = agg_stats['turnovers'] + agg_stats['fumbles']
    return agg_stats

def get_rest_days(games):
    # Compute rest days since previous team game
    rest = []
    for team in set(games['home_team']).union(set(games['away_team'])):
        tgames = games[(games['home_team'] == team) | (games['away_team'] == team)].sort_values(['season', 'week'])
        last_date = None
        for idx, row in tgames.iterrows():
            gdate = pd.to_datetime(row['start_date'])
            if last_date is None:
                rest.append((row['game_id'], team, 7))
            else:
                diff = (gdate - last_date).days
                rest.append((row['game_id'], team, diff if diff > 0 else 7))
            last_date = gdate
    rest_df = pd.DataFrame(rest, columns=['game_id', 'team', 'rest_days'])
    return rest_df

def build_features(games, pbp):
    games = add_vegas_lines(games)
    # Compute per-team-per-game stats
    agg_stats = compute_team_game_stats(pbp)
    # Prepare long-form (per team per game) DataFrame
    home = games[['game_id','season','week','start_date','home_team','away_team','home_score','away_score','spread_line','total_line']].rename(
        columns={'home_team':'team','away_team':'opp','home_score':'points_scored','away_score':'opp_points'}
    )
    home['is_home'] = 1
    away = games[['game_id','season','week','start_date','away_team','home_team','away_score','home_score','spread_line','total_line']].rename(
        columns={'away_team':'team','home_team':'opp','away_score':'points_scored','home_score':'opp_points'}
    )
    away['is_home'] = 0
    long_games = pd.concat([home, away], ignore_index=True)
    # Merge in aggregated per-team stats
    long_games = long_games.merge(agg_stats, how='left', left_on=['game_id','team'], right_on=['game_id','posteam'])
    # Rolling stats
    long_games = long_games.sort_values(['team','season','week'])
    for stat in ['points_scored','total_yards','turnovers']:
        long_games[f'{stat}_rolling5'] = long_games.groupby('team')[stat].rolling(5, min_periods=1).mean().reset_index(0,drop=True)
    # Rest days
    rest_df = get_rest_days(games)
    long_games = long_games.merge(rest_df, how='left', on=['game_id','team'])
    # Prepare matchup features per game (merge home/away features)
    features = []
    for _, row in games.iterrows():
        # Get last rolling stats for home & away teams before this game
        def get_last(team, week, season, is_home):
            prev = long_games[
                (long_games['team'] == team) &
                ((long_games['season'] < season) | ((long_games['season'] == season) & (long_games['week'] < week)))
            ]
            prev = prev[prev['is_home'] == is_home] if not prev.empty else prev
            return prev.iloc[-1] if not prev.empty else {}
        home_last = get_last(row['home_team'], row['week'], row['season'], 1)
        away_last = get_last(row['away_team'], row['week'], row['season'], 0)
        features.append({
            'game_id': row['game_id'],
            'season': row['season'],
            'week': row['week'],
            'home_team': row['home_team'],
            'away_team': row['away_team'],
            'home_points_avg': home_last.get('points_scored_rolling5', 0),
            'away_points_avg': away_last.get('points_scored_rolling5', 0),
            'home_yards_avg': home_last.get('total_yards_rolling5', 0),
            'away_yards_avg': away_last.get('total_yards_rolling5', 0),
            'home_tov_avg': home_last.get('turnovers_rolling5', 0),
            'away_tov_avg': away_last.get('turnovers_rolling5', 0),
            'home_rest': home_last.get('rest_days', 7),
            'away_rest': away_last.get('rest_days', 7),
            'home_is_favorite': int(row['spread_line'] < 0),
            'spread': abs(row['spread_line']),
            'over_under': row['total_line'],
            'home_score': row['home_score'],
            'away_score': row['away_score']
        })
    df = pd.DataFrame(features)
    df = df.dropna(subset=['home_score','away_score'])
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['total_points'] = df['home_score'] + df['away_score']
    return df

@st.cache_resource(show_spinner=True)
def train_models(df):
    feature_cols = [
        'home_points_avg','away_points_avg','home_yards_avg','away_yards_avg',
        'home_tov_avg','away_tov_avg','home_rest','away_rest','home_is_favorite',
        'spread','over_under'
    ]
    X = df[feature_cols].fillna(0)
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
years = st.sidebar.multiselect(
    "Select NFL seasons to use for training:",
    list(range(2010, datetime.now().year+1)),
    default=[2021, 2022, 2023]
)
games, pbp, teams = load_data(years)
st.info(f"Loaded {len(games)} games, {len(pbp)} play-by-play rows.")

with st.spinner("Building features and training models (this can take a few minutes on first run)..."):
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

def get_last_stats(team, features, is_home):
    # Get the latest rolling stats for the team
    games = features[(features['home_team'] == team) if is_home else (features['away_team'] == team)]
    if games.empty:
        return [0]*4
    last = games.iloc[-1]
    if is_home:
        return [
            last['home_points_avg'],
            last['home_yards_avg'],
            last['home_tov_avg'],
            last['home_rest']
        ]
    else:
        return [
            last['away_points_avg'],
            last['away_yards_avg'],
            last['away_tov_avg'],
            last['away_rest']
        ]

if st.button("Predict!"):
    home_pts, home_yds, home_tov, home_rest = get_last_stats(home_team, features, True)
    away_pts, away_yds, away_tov, away_rest = get_last_stats(away_team, features, False)
    fav = int(spread < 0)
    input_df = pd.DataFrame([{
        'home_points_avg': home_pts,
        'away_points_avg': away_pts,
        'home_yards_avg': home_yds,
        'away_yards_avg': away_yds,
        'home_tov_avg': home_tov,
        'away_tov_avg': away_tov,
        'home_rest': home_rest,
        'away_rest': away_rest,
        'home_is_favorite': fav,
        'spread': abs(spread),
        'over_under': over_under
    }])
    win_prob = clf.predict_proba(input_df)[0,1]
    total_pts = reg.predict(input_df)[0]
    st.markdown(
        f"### 🏆 Winner Prediction: **{'Home' if win_prob>=0.5 else 'Away'} Team** "
        f"({home_team if win_prob>=0.5 else away_team})"
    )
    st.markdown(f"- Probability Home Wins: **{win_prob:.1%}**")
    st.markdown(f"- Predicted Total Points: **{total_pts:.1f}**")
    st.markdown(f"- Vegas Spread: {spread}, Over/Under: {over_under}")
    st.caption("Features used: rolling averages (points, yards, turnovers), rest days, Vegas lines, home/away.")

st.markdown("---")
st.caption("Author: Your Name | Data from nfl_data_py | Models: XGBoost")
