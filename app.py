import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold
from packaging import version
import sklearn

TOP_N_FEATURES = 40
STACK_FOLDS = 3  # Reduced for speed

@st.cache_data(show_spinner=False)
def fetch_nfl_data(seasons):
    df = nfl.import_schedules(seasons)
    df = df.dropna(subset=['home_score', 'away_score'])
    return df

@st.cache_data(show_spinner=False)
def feature_engineering(df):
    all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
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

    all_teams = list(all_teams)
    # --- Advanced Feature Engineering ---

    # 1. Win/Loss streaks, rolling averages, home/away splits
    streaks = {}
    rolling_3 = {}
    rolling_5 = {}
    home_rolling_3 = {}
    away_rolling_3 = {}
    for team in all_teams:
        # Get games for this team
        team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('gameday')
        # Win/Loss
        team_games['team_score'] = np.where(team_games['home_team'] == team, team_games['home_score'], team_games['away_score'])
        team_games['opp_score'] = np.where(team_games['home_team'] == team, team_games['away_score'], team_games['home_score'])
        team_games['win'] = (team_games['team_score'] > team_games['opp_score']).astype(int)
        # Streak
        streak = []
        cur = 0
        for result in team_games['win']:
            if result:
                cur = cur + 1 if cur >= 0 else 1
            else:
                cur = cur - 1 if cur <= 0 else -1
            streak.append(cur)
        streaks[team] = dict(zip(team_games['game_id'], streak))
        # Rolling avg scored/allowed last 3 and 5
        team_games['score_rolling_3'] = team_games['team_score'].rolling(3, min_periods=1).mean()
        team_games['allow_rolling_3'] = team_games['opp_score'].rolling(3, min_periods=1).mean()
        team_games['score_rolling_5'] = team_games['team_score'].rolling(5, min_periods=1).mean()
        team_games['allow_rolling_5'] = team_games['opp_score'].rolling(5, min_periods=1).mean()
        rolling_3[team] = dict(zip(team_games['game_id'], zip(team_games['score_rolling_3'], team_games['allow_rolling_3'])))
        rolling_5[team] = dict(zip(team_games['game_id'], zip(team_games['score_rolling_5'], team_games['allow_rolling_5'])))
        # Home/away splits: rolling average in last 3 home/away games
        home_games = team_games[team_games['home_team'] == team]
        away_games = team_games[team_games['away_team'] == team]
        h_rolling = home_games['home_score'].rolling(3, min_periods=1).mean()
        a_rolling = away_games['away_score'].rolling(3, min_periods=1).mean()
        home_rolling_3[team] = dict(zip(home_games['game_id'], h_rolling))
        away_rolling_3[team] = dict(zip(away_games['game_id'], a_rolling))

    df['home_team_streak'] = df.apply(lambda row: streaks[row['home_team']].get(row['game_id'], 0), axis=1)
    df['away_team_streak'] = df.apply(lambda row: streaks[row['away_team']].get(row['game_id'], 0), axis=1)
    df['home_team_scored_last3'] = df.apply(lambda row: rolling_3[row['home_team']].get(row['game_id'], (row['home_score'], row['away_score']))[0], axis=1)
    df['home_team_allowed_last3'] = df.apply(lambda row: rolling_3[row['home_team']].get(row['game_id'], (row['home_score'], row['away_score']))[1], axis=1)
    df['away_team_scored_last3'] = df.apply(lambda row: rolling_3[row['away_team']].get(row['game_id'], (row['away_score'], row['home_score']))[0], axis=1)
    df['away_team_allowed_last3'] = df.apply(lambda row: rolling_3[row['away_team']].get(row['game_id'], (row['away_score'], row['home_score']))[1], axis=1)
    df['home_team_scored_last5'] = df.apply(lambda row: rolling_5[row['home_team']].get(row['game_id'], (row['home_score'], row['away_score']))[0], axis=1)
    df['home_team_allowed_last5'] = df.apply(lambda row: rolling_5[row['home_team']].get(row['game_id'], (row['home_score'], row['away_score']))[1], axis=1)
    df['away_team_scored_last5'] = df.apply(lambda row: rolling_5[row['away_team']].get(row['game_id'], (row['away_score'], row['home_score']))[0], axis=1)
    df['away_team_allowed_last5'] = df.apply(lambda row: rolling_5[row['away_team']].get(row['game_id'], (row['away_score'], row['home_score']))[1], axis=1)
    df['home_team_home_scored_last3'] = df.apply(lambda row: home_rolling_3[row['home_team']].get(row['game_id'], row['home_score']), axis=1)
    df['away_team_away_scored_last3'] = df.apply(lambda row: away_rolling_3[row['away_team']].get(row['game_id'], row['away_score']), axis=1)

    # 2. Head-to-head history: last 3 meetings
    def last_n_meetings(row, n=3, col='winner'):
        games = df[
            ((df['home_team'] == row['home_team']) & (df['away_team'] == row['away_team'])) |
            ((df['home_team'] == row['away_team']) & (df['away_team'] == row['home_team']))
        ]
        games = games[games['gameday'] < row['gameday']].sort_values('gameday', ascending=False).head(n)
        if col == "winner":
            # 1=home won, 0=away won, flip if swapped
            results = []
            for _, game in games.iterrows():
                if game['home_team'] == row['home_team']:
                    results.append(1 if game['home_score'] > game['away_score'] else 0)
                else:
                    results.append(1 if game['away_score'] > game['home_score'] else 0)
            return np.mean(results) if results else 0.5
        elif col == "margin":
            margins = []
            for _, game in games.iterrows():
                if game['home_team'] == row['home_team']:
                    margins.append(game['home_score'] - game['away_score'])
                else:
                    margins.append(game['away_score'] - game['home_score'])
            return np.mean(margins) if margins else 0
        return 0

    df['h2h_last3_winrate'] = df.apply(lambda row: last_n_meetings(row, 3, col='winner'), axis=1)
    df['h2h_last3_margin'] = df.apply(lambda row: last_n_meetings(row, 3, col='margin'), axis=1)

    # 3. Rest differential
    df['rest_diff'] = df['home_rest'] - df['away_rest']

    # 4. Weather buckets
    df['cold_game'] = (df['temp'] < 40).astype(int)
    df['hot_game'] = (df['temp'] > 80).astype(int)
    df['windy_game'] = (df['wind'] > 15).astype(int)

    # 5. Game importance: week number, playoff flag, late season
    df['week'] = pd.to_datetime(df['gameday'], errors='coerce').dt.isocalendar().week.fillna(0).astype(int)
    df['late_season'] = (df['week'] > 14).astype(int)
    df['is_playoff'] = (df['game_type'] == 'POST').astype(int)

    # 6. Turnover differential (if present)
    if 'home_turnovers' in df.columns and 'away_turnovers' in df.columns:
        df['home_turnover_diff'] = df['away_turnovers'] - df['home_turnovers']
        df['away_turnover_diff'] = df['home_turnovers'] - df['away_turnovers']
    else:
        df['home_turnover_diff'] = 0
        df['away_turnover_diff'] = 0

    # --- Existing rolling team stats ---
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

    # --- Combine all features ---
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
        df[[
            'home_advantage', 'div_game', 'temp', 'wind', 'spread_line', 'total_line',
            'away_moneyline', 'home_moneyline', 'away_rest', 'home_rest',
            'home_team_avg', 'away_team_avg', 'home_team_def_avg', 'away_team_def_avg',
            'home_team_win_rate', 'away_team_win_rate',
            # Advanced features
            'home_team_streak', 'away_team_streak',
            'home_team_scored_last3', 'away_team_scored_last3',
            'home_team_allowed_last3', 'away_team_allowed_last3',
            'home_team_scored_last5', 'away_team_scored_last5',
            'home_team_allowed_last5', 'away_team_allowed_last5',
            'home_team_home_scored_last3', 'away_team_away_scored_last3',
            'h2h_last3_winrate', 'h2h_last3_margin',
            'rest_diff', 'cold_game', 'hot_game', 'windy_game',
            'week', 'late_season', 'is_playoff',
            'home_turnover_diff', 'away_turnover_diff'
        ]]
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

    # Use team averages from training data for new matchups
    input_dict['home_team_avg'] = team_avgs.get(home_team, last_row.get('home_team_avg', 21))
    input_dict['away_team_avg'] = team_avgs.get(away_team, last_row.get('away_team_avg', 21))
    input_dict['home_team_def_avg'] = team_def_avgs.get(home_team, last_row.get('home_team_def_avg', 21))
    input_dict['away_team_def_avg'] = team_def_avgs.get(away_team, last_row.get('away_team_def_avg', 21))
    input_dict['home_team_win_rate'] = team_win_avgs.get(home_team, last_row.get('home_team_win_rate', 0.5))
    input_dict['away_team_win_rate'] = team_win_avgs.get(away_team, last_row.get('away_team_win_rate', 0.5))

    # New engineered features - fill with safe defaults or averages
    for col in [
        'home_team_streak', 'away_team_streak',
        'home_team_scored_last3', 'away_team_scored_last3',
        'home_team_allowed_last3', 'away_team_allowed_last3',
        'home_team_scored_last5', 'away_team_scored_last5',
        'home_team_allowed_last5', 'away_team_allowed_last5',
        'home_team_home_scored_last3', 'away_team_away_scored_last3',
        'h2h_last3_winrate', 'h2h_last3_margin',
        'rest_diff', 'cold_game', 'hot_game', 'windy_game',
        'week', 'late_season', 'is_playoff',
        'home_turnover_diff', 'away_turnover_diff'
    ]:
        input_dict[col] = last_row.get(col, 0)

    X_pred = pd.DataFrame([input_dict])
    for col in all_possible_columns:
        if col not in X_pred.columns:
            X_pred[col] = 0
    X_pred = X_pred[all_possible_columns]
    X_pred = X_pred.fillna(0)
    X_pred = X_pred.replace([np.inf, -np.inf], 0)
    return X_pred

st.title("NFL Game Winner & Score Predictor (Stacked Ensemble + More Seasons + Advanced Features)")

seasons = list(range(2018, datetime.today().year + 1))

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

    y_winner = (df['home_score'] > df['away_score']).astype(int)
    y_margin = df['home_score'] - df['away_score']
    y_total = df['home_score'] + df['away_score']

    selected_features_margin = select_features(X, y_margin, TOP_N_FEATURES)
    selected_features_total = select_features(X, y_total, TOP_N_FEATURES)
    selected_features_cls = select_features(X, y_winner, TOP_N_FEATURES)

    X_margin_selected = X[selected_features_margin]
    X_total_selected = X[selected_features_total]
    X_cls_selected = X[selected_features_cls]

    base_clf_models = [
        RandomForestClassifier(n_estimators=60, random_state=42),
        XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
        LogisticRegression(max_iter=5000, random_state=42)
    ]
    base_margin_models = [
        RandomForestRegressor(n_estimators=60, random_state=42),
        XGBRegressor(n_estimators=60, random_state=42)
    ]
    base_total_models = [
        RandomForestRegressor(n_estimators=60, random_state=42),
        XGBRegressor(n_estimators=60
                     
                     , random_state=42)
    ]

    def get_stacking_preds(X, y, model_list, problem_type="reg"):
        n_folds = STACK_FOLDS
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        oof_preds = np.zeros((len(X), len(model_list)))
        for i, model in enumerate(model_list):
            for train_idx, valid_idx in kf.split(X):
                X_train, X_valid = X.iloc[train_idx], X.iloc[valid_idx]
                y_train = y.iloc[train_idx]
                model.fit(X_train, y_train)
                if problem_type == "cls":
                    if hasattr(model, 'predict_proba'):
                        oof_preds[valid_idx, i] = model.predict_proba(X_valid)[:, 1]
                    else:
                        oof_preds[valid_idx, i] = model.predict(X_valid)
                else:
                    oof_preds[valid_idx, i] = model.predict(X_valid)
        return oof_preds

    def fit_base_models(X, y, model_list, problem_type="reg"):
        for model in model_list:
            model.fit(X, y)
        return model_list

    X_cls_stack = get_stacking_preds(X_cls_selected, y_winner, base_clf_models, problem_type="cls")
    meta_clf = LogisticRegression(max_iter=5000, random_state=42)
    meta_clf.fit(X_cls_stack, y_winner)
    fit_base_models(X_cls_selected, y_winner, base_clf_models, "cls")

    X_margin_stack = get_stacking_preds(X_margin_selected, y_margin, base_margin_models, problem_type="reg")
    meta_margin = Ridge()
    meta_margin.fit(X_margin_stack, y_margin)
    fit_base_models(X_margin_selected, y_margin, base_margin_models, "reg")

    X_total_stack = get_stacking_preds(X_total_selected, y_total, base_total_models, problem_type="reg")
    meta_total = Ridge()
    meta_total.fit(X_total_stack, y_total)
    fit_base_models(X_total_selected, y_total, base_total_models, "reg")

    def get_residual_std_stack(meta, X_stack, y):
        pred = meta.predict(X_stack)
        return (y - pred).std()
    resid_std_margin = get_residual_std_stack(meta_margin, X_margin_stack, y_margin)
    resid_std_total = get_residual_std_stack(meta_total, X_total_stack, y_total)

def stacking_predict(models, meta, X, problem_type="reg", noise_std=0):
    stack_feats = []
    for model in models:
        if problem_type == "cls":
            if hasattr(model, "predict_proba"):
                stack_feats.append(model.predict_proba(X)[:, 1])
            else:
                stack_feats.append(model.predict(X))
        else:
            stack_feats.append(model.predict(X))
    stack_feats = np.column_stack(stack_feats)
    pred = meta.predict(stack_feats)
    if noise_std > 0:
        pred += np.random.normal(0, noise_std, size=pred.shape)
    return pred

def predict_game(X_pred_full):
    # Winner classification
    X_cls_pred = X_pred_full[selected_features_cls]
    # Get stacking features for meta-classifier
    stack_feats = []
    for model in base_clf_models:
        if hasattr(model, "predict_proba"):
            stack_feats.append(model.predict_proba(X_cls_pred)[:, 1])
        else:
            stack_feats.append(model.predict(X_cls_pred))
    stack_feats = np.column_stack(stack_feats)
    win_proba = meta_clf.predict_proba(stack_feats)[0, 1]
    pred_winner = int(win_proba > 0.5)

    # Margin regression
    X_margin_pred = X_pred_full[selected_features_margin]
    pred_margin = stacking_predict(
        base_margin_models, meta_margin, X_margin_pred, problem_type="reg", noise_std=resid_std_margin
    )[0]

    # Total regression
    X_total_pred = X_pred_full[selected_features_total]
    pred_total = stacking_predict(
        base_total_models, meta_total, X_total_pred, problem_type="reg", noise_std=resid_std_total
    )[0]

    # If classifier says home wins, margin is positive; else, flip sign
    if pred_winner == 0:
        pred_margin = -abs(pred_margin)
    else:
        pred_margin = abs(pred_margin)

    # Calculate scores
    home_score = (pred_total + pred_margin) / 2
    away_score = (pred_total - pred_margin) / 2

    # Clamp negatives, round using standard rounding
    home_score = max(0, round(home_score))
    away_score = max(0, round(away_score))

    return {
        "winner": "Home" if pred_winner == 1 else "Away",
        "predicted_home_score": home_score,
        "predicted_away_score": away_score,
        "home_win_proba": win_proba,
        "away_win_proba": 1 - win_proba
    }

# ------------ UPCOMING GAMES (NEXT 6 DAYS) -------------
today = datetime.today().date()
six_days_later = today + timedelta(days=6)

schedule_df = nfl.import_schedules([today.year])
schedule_df['gameday'] = pd.to_datetime(schedule_df['gameday']).dt.date

upcoming_games = schedule_df[
    (schedule_df['gameday'] >= today) &
    (schedule_df['gameday'] <= six_days_later) &
    (schedule_df['home_score'].isnull()) &
    (schedule_df['away_score'].isnull())
]

if len(upcoming_games) == 0:
    st.info("No NFL games in the next 6 days.")
else:
    st.subheader(f"Predictions for NFL games in the next 6 days ({today} to {six_days_later}):")
    pred_rows = []
    for i, row in upcoming_games.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        X_pred_full = build_features_for_matchup(
            home_team, away_team, encoder, df,
            all_possible_columns, team_avgs, team_def_avgs, team_win_avgs
        )
        result = predict_game(X_pred_full)
        pred_rows.append({
            "Date": row['gameday'],
            "Away Team": away_team,
            "Home Team": home_team,
            "Predicted Away Score": result["predicted_away_score"],
            "Predicted Home Score": result["predicted_home_score"],
            "Predicted Winner": result["winner"],
            "Home Win Prob": f"{100*result['home_win_proba']:.1f}%",
            "Away Win Prob": f"{100*result['away_win_proba']:.1f}%"
        })
    pred_df = pd.DataFrame(pred_rows)
    pred_df = pred_df.sort_values("Date")
    st.dataframe(pred_df)
    st.caption("Predictions for upcoming NFL games in the next 6 days.")

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
    result = predict_game(X_pred_full)
    st.success(
        f"Prediction: {away_team} @ {home_team}: {result['predicted_away_score']} - {result['predicted_home_score']}  \n"
        f"Winner: {result['winner']} team  \n"
        f"Home Win Probability: {100*result['home_win_proba']:.1f}%  \n"
        f"Away Win Probability: {100*result['away_win_proba']:.1f}%"
    )
elif home_team == away_team:
    st.warning("Select different teams!")

st.caption("Powered by nfl_data_py and scikit-learn. Stacked ensemble (meta-model) with historical data since 2010 and advanced feature engineering.")