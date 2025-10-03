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

    df['home_advantage'] = 3
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
    return X, y_home, y_away, encoder, all_teams, df

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
    input_dict['home_advantage'] = 3
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
    X_pred = X_pred.fillna(0)
    X_pred = X_pred.replace([np.inf, -np.inf], 0)
    return X_pred

st.title("NFL Game Winner & Score Predictor (Stacked Ensemble + More Seasons)")

seasons = list(range(2020, datetime.today().year + 1))

with st.spinner("Loading and training... (first run may take a minute)"):
    df = fetch_nfl_data(seasons)
    X, y_home, y_away, encoder, all_teams, df = feature_engineering(df)
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
        RandomForestClassifier(n_estimators=120, random_state=42),
        XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss"),
        LogisticRegression(max_iter=4500, random_state=42)
    ]
    base_margin_models = [
        RandomForestRegressor(n_estimators=120, random_state=42),
        XGBRegressor(n_estimators=120, random_state=42)
    ]
    base_total_models = [
        RandomForestRegressor(n_estimators=120, random_state=42),
        XGBRegressor(n_estimators=120, random_state=42)
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
    meta_clf = LogisticRegression(max_iter=4500, random_state=42)
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

    if pred_winner == 0:
        pred_margin = -abs(pred_margin)
    else:
        pred_margin = abs(pred_margin)

    home_score = (pred_total + pred_margin) / 2
    away_score = (pred_total - pred_margin) / 2

    home_score = max(0, round(home_score))
    away_score = max(0, round(away_score))

    return {
        "winner": "Home" if pred_winner == 1 else "Away",
        "predicted_home_score": home_score,
        "predicted_away_score": away_score,
        "home_win_proba": win_proba,
        "away_win_proba": 1 - win_proba
    }

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




########## Second attempt ###################
import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---- CONFIG ----
DEFAULT_SEASONS = [2024,2025]
ROLLING_WINDOW = 3

# ---- LOAD DATA ----
@st.cache_data(show_spinner=False)
def load_data(seasons):
    sched = nfl.import_schedules(seasons)
    pbp = nfl.import_pbp_data(seasons)
    sched = sched.dropna(subset=['home_score', 'away_score'], how='all')  # allow future games with both null
    pbp = pbp[pbp['season'].isin(seasons)]
    return sched, pbp

# ---- PRE-AGGREGATE PER-TEAM, PER-GAME STATS ----
def get_first_or_nan(series):
    try:
        return series.iloc[0]
    except Exception:
        return np.nan

def preaggregate_team_game_stats(sched, pbp):
    features = []
    for _, game in sched.iterrows():
        for team, is_home in [(game['home_team'], True), (game['away_team'], False)]:
            game_id = game['game_id']
            season = game['season']
            week = game['week']
            off = pbp[(pbp['game_id'] == game_id) & (pbp['posteam'] == team)]
            deff = pbp[(pbp['game_id'] == game_id) & (pbp['defteam'] == team)]

            # Turnover Differential
            takeaways = ((deff['interception'] == 1) | (deff['fumble_lost'] == 1)).sum()
            giveaways = ((off['interception'] == 1) | (off['fumble_lost'] == 1)).sum()
            turnover_diff = takeaways - giveaways

            # QB Efficiency (EPA/play)
            qb_eff = off['epa'].mean()

            # Yards per Play Differential
            off_ypp = off['yards_gained'].mean()
            def_ypp = deff['yards_gained'].mean()
            ypp_diff = (off_ypp - def_ypp) if (off_ypp is not None and def_ypp is not None) else 0

            # Success Rate Differential (EPA > 0)
            off_succ = (off['epa'] > 0).mean()
            def_succ = (deff['epa'] > 0).mean()
            success_rate_diff = off_succ - def_succ

            # Explosive Plays Differential (20+ yards)
            off_exp = (off['yards_gained'] >= 20).sum()
            def_exp = (deff['yards_gained'] >= 20).sum()
            explosive_diff = off_exp - def_exp

            # Pressure/Sack Rate Differential
            off_dropbacks = off['pass_attempt'].sum()
            def_dropbacks = deff['pass_attempt'].sum()
            off_sacks = off['sack'].sum()
            def_sacks = deff['sack'].sum()
            off_sack_rate = (off_sacks / off_dropbacks) if off_dropbacks else 0
            def_sack_rate = (def_sacks / def_dropbacks) if def_dropbacks else 0
            sack_rate_diff = def_sack_rate - off_sack_rate

            # Red Zone Efficiency Differential
            off_rz_trips = off[(off['yardline_100'] <= 20) & (off['down'] == 1)]
            off_rz_tds = off_rz_trips['touchdown'].sum() if not off_rz_trips.empty else 0
            off_rz_pct = (off_rz_tds / len(off_rz_trips)) if len(off_rz_trips) > 0 else 0
            def_rz_trips = deff[(deff['yardline_100'] <= 20) & (deff['down'] == 1)]
            def_rz_tds = def_rz_trips['touchdown'].sum() if not def_rz_trips.empty else 0
            def_rz_pct = (def_rz_tds / len(def_rz_trips)) if len(def_rz_trips) > 0 else 0
            rz_diff = off_rz_pct - def_rz_pct

            # Third Down Conversion % Differential
            off_3rd = off[off['down'] == 3]
            off_3rd_conv = (off_3rd['first_down'] == 1).mean() if len(off_3rd) > 0 else 0
            def_3rd = deff[deff['down'] == 3]
            def_3rd_conv = (def_3rd['first_down'] == 1).mean() if len(def_3rd) > 0 else 0
            third_down_diff = off_3rd_conv - def_3rd_conv

            # Starting Field Position Differential (FIXED)
            try:
                off_fp_series = 100 - off.groupby('game_id').first()['yardline_100']
                off_fp = get_first_or_nan(off_fp_series)
            except Exception:
                off_fp = np.nan
            try:
                def_fp_series = 100 - deff.groupby('game_id').first()['yardline_100']
                def_fp = get_first_or_nan(def_fp_series)
            except Exception:
                def_fp = np.nan
            field_pos_diff = (off_fp - def_fp) if pd.notnull(off_fp) and pd.notnull(def_fp) else 0

            # Special Teams Efficiency (FG% + Net Punt)
            off_fg = off[off['play_type'] == 'field_goal']
            fg_made = (off_fg['field_goal_result'] == 'made').sum()
            fg_att = len(off_fg)
            fg_pct = (fg_made / fg_att) if fg_att else 0
            punts = off[off['play_type'] == 'punt']
            net_punt = punts['punt_net_yards'].mean() if len(punts) > 0 and 'punt_net_yards' in punts.columns else 0
            special_teams = fg_pct + (net_punt / 100 if net_punt else 0)

            # Rushing Efficiency Differential
            off_rush = off[off['play_type'] == 'run']
            def_rush = deff[deff['play_type'] == 'run']
            off_rush_ypp = off_rush['yards_gained'].mean() if len(off_rush) > 0 else 0
            def_rush_ypp = def_rush['yards_gained'].mean() if len(def_rush) > 0 else 0
            rush_yards_diff = off_rush_ypp - def_rush_ypp

            # Time of Possession Differential (minutes)
            if 'drive_time_sec' in off.columns:
                off_top = off.groupby('game_id')['drive_time_sec'].sum().mean() / 60
            else:
                off_top = np.nan
            if 'drive_time_sec' in deff.columns:
                def_top = deff.groupby('game_id')['drive_time_sec'].sum().mean() / 60
            else:
                def_top = np.nan
            top_diff = (off_top - def_top) if pd.notnull(off_top) and pd.notnull(def_top) else 0

            # Penalty Yards Differential
            off_pen = off[off['penalty'] == 1]['penalty_yards'].sum()
            def_pen = deff[deff['penalty'] == 1]['penalty_yards'].sum()
            penalty_diff = off_pen - def_pen

            features.append({
                'team': team, 'game_id': game_id, 'season': season, 'week': week,
                'turnover_diff': turnover_diff,
                'qb_efficiency': qb_eff,
                'ypp_diff': ypp_diff,
                'success_rate_diff': success_rate_diff,
                'explosive_diff': explosive_diff,
                'sack_rate_diff': sack_rate_diff,
                'rz_diff': rz_diff,
                'third_down_diff': third_down_diff,
                'field_pos_diff': field_pos_diff,
                'special_teams': special_teams,
                'rush_yards_diff': rush_yards_diff,
                'top_diff': top_diff,
                'penalty_diff': penalty_diff
            })
    return pd.DataFrame(features)

# ---- COMPUTE 3-GAME ROLLING AVERAGES ----
def compute_rolling_features(team_game_stats, feature_names, rolling_window=3):
    team_game_stats = team_game_stats.sort_values(['team', 'season', 'week'])
    for col in feature_names:
        team_game_stats[f'{col}_roll'] = (
            team_game_stats.groupby('team')[col]
            .transform(lambda x: x.shift(1).rolling(rolling_window, min_periods=1).mean())
        )
    return team_game_stats

# ---- BUILD FINAL FEATURE DF FOR MODELING ----
def build_feature_df(sched, team_game_stats, feature_names):
    merged = sched.copy()
    for col in feature_names:
        merged = merged.merge(
            team_game_stats[['game_id', 'team', f'{col}_roll']].rename(
                columns={f'{col}_roll': f'home_{col}_roll'}),
            left_on=['game_id', 'home_team'],
            right_on=['game_id', 'team'],
            how='left'
        )
        merged = merged.merge(
            team_game_stats[['game_id', 'team', f'{col}_roll']].rename(
                columns={f'{col}_roll': f'away_{col}_roll'}),
            left_on=['game_id', 'away_team'],
            right_on=['game_id', 'team'],
            how='left'
        )
    # Compute differentials (home - away)
    feature_cols = []
    for col in feature_names:
        diff_col = f'{col}_diff'
        merged[diff_col] = merged[f'home_{col}_roll'] - merged[f'away_{col}_roll']
        feature_cols.append(diff_col)
    X = merged[feature_cols].fillna(0)
    y_home = merged['home_score']
    y_away = merged['away_score']
    return X, y_home, y_away, merged, feature_cols

# ---- STREAMLIT APP ----
st.title("NFL Game Predictor (Optimized, 3-Game Rolling Stats)")

# Allow selecting future season for predictions (e.g., 2025)
future_year = datetime.today().year + 1
season_options = sorted(list(set(list(range(2010, datetime.today().year + 1)) + [future_year])))
seasons = st.multiselect("Seasons to load", options=season_options[:-1], default=DEFAULT_SEASONS)
selected_predict_season = st.selectbox("Prediction Season", season_options, index=len(season_options)-1)

with st.spinner("Loading and training... (first run may take a minute)"):
    sched, pbp = load_data(seasons)
    feature_names = [
        'turnover_diff', 'qb_efficiency', 'ypp_diff', 'success_rate_diff', 'explosive_diff', 'sack_rate_diff',
        'rz_diff', 'third_down_diff', 'field_pos_diff', 'special_teams', 'rush_yards_diff', 'top_diff', 'penalty_diff'
    ]
    team_game_stats = preaggregate_team_game_stats(sched, pbp)
    team_game_stats = compute_rolling_features(team_game_stats, feature_names, rolling_window=ROLLING_WINDOW)
    X, y_home, y_away, merged, feature_cols = build_feature_df(sched, team_game_stats, feature_names)

    y_winner = (y_home > y_away).astype(int)
    y_margin = y_home - y_away
    y_total = y_home + y_away

    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X, y_winner)
    margin_reg = RandomForestRegressor(n_estimators=120, random_state=42)
    margin_reg.fit(X, y_margin)
    total_reg = RandomForestRegressor(n_estimators=120, random_state=42)
    total_reg.fit(X, y_total)

    def get_recent_rolling_stats(team, season, week):
        # Use all games up to the most recent available (if future)
        team_games = team_game_stats[
            (team_game_stats['team'] == team) &
            ((team_game_stats['season'] < season) |
             ((team_game_stats['season'] == season) & (team_game_stats['week'] < week)))
        ]
        if team_games.empty:
            # fallback: use latest available
            team_games = team_game_stats[team_game_stats['team'] == team]
        if not team_games.empty:
            row = team_games.sort_values(['season', 'week']).iloc[-1]
            return {f'{col}_roll': row.get(f'{col}_roll', 0) for col in feature_names}
        else:
            return {f'{col}_roll': 0 for col in feature_names}

    def build_features_for_matchup(home_team, away_team, season, week):
        home_stats = get_recent_rolling_stats(home_team, season, week)
        away_stats = get_recent_rolling_stats(away_team, season, week)
        feats = {}
        for col in feature_names:
            feats[f'{col}_diff'] = home_stats.get(f'{col}_roll', 0) - away_stats.get(f'{col}_roll', 0)
        return pd.DataFrame([feats])

    def predict_game(X_pred):
        win_proba = clf.predict_proba(X_pred)[0, 1]
        pred_winner = int(win_proba > 0.5)
        pred_margin = margin_reg.predict(X_pred)[0]
        pred_total = total_reg.predict(X_pred)[0]
        if pred_winner == 0:
            pred_margin = -abs(pred_margin)
        else:
            pred_margin = abs(pred_margin)
        home_score = (pred_total + pred_margin) / 2
        away_score = (pred_total - pred_margin) / 2
        home_score = max(0, round(home_score))
        away_score = max(0, round(away_score))
        return {
            "winner": "Home" if pred_winner == 1 else "Away",
            "predicted_home_score": home_score,
            "predicted_away_score": away_score,
            "home_win_proba": win_proba,
            "away_win_proba": 1 - win_proba
        }

# ---- UPCOMING GAMES ----

# Always use the real date for live filtering
today = datetime.today().date()
six_days_later = today + timedelta(days=6)
schedule_df = sched.copy()
schedule_df['gameday'] = pd.to_datetime(schedule_df['gameday']).dt.date

upcoming_games = schedule_df[
    (schedule_df['gameday'] >= today) &
    (schedule_df['gameday'] <= six_days_later) &
    (schedule_df['home_score'].isnull()) &
    (schedule_df['away_score'].isnull())
]

if len(upcoming_games) == 0:
    if selected_predict_season not in sched['season'].unique():
        st.warning(f"No schedule data found for {selected_predict_season}. If you're trying to predict future games, use the manual matchup prediction below!")
    else:
        st.info("No NFL games in the next 6 days.")
else:
    st.subheader(f"Predictions for NFL games in the next 6 days ({today} to {six_days_later}):")
    pred_rows = []
    for i, row in upcoming_games.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        season = row['season']
        week = row['week']
        X_pred = build_features_for_matchup(home_team, away_team, season, week)
        result = predict_game(X_pred)
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
all_teams = sorted(list(set(list(sched['home_team'].unique()) + list(sched['away_team'].unique()))))
home_team = st.selectbox("Select Home Team", all_teams)
away_team = st.selectbox("Select Away Team", all_teams, index=1 if all_teams[1] != home_team else 2)
predict_season = selected_predict_season
predict_week = st.number_input("Week (for rolling context)", min_value=1, max_value=22, value=1)

if home_team and away_team and home_team != away_team:
    X_pred = build_features_for_matchup(home_team, away_team, predict_season, predict_week)
    result = predict_game(X_pred)
    st.success(
        f"Prediction: {away_team} @ {home_team}: {result['predicted_away_score']} - {result['predicted_home_score']}  \n"
        f"Winner: {result['winner']} team  \n"
        f"Home Win Probability: {100*result['home_win_proba']:.1f}%  \n"
        f"Away Win Probability: {100*result['away_win_proba']:.1f}%"
    )
elif home_team == away_team:
    st.warning("Select different teams!")

st.caption("Powered by nfl_data_py and scikit-learn. All features are 3-game rolling averages, optimized to avoid memory issues.")


st.caption("Powered by nfl_data_py and scikit-learn. Stacked ensemble (meta-model) with historical data since 2010.")

