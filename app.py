import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

# ---- CONFIG ----
DEFAULT_SEASONS = [2022,2023,2024]  # Start with one season. Add more as RAM allows.
ROLLING_WINDOW = 3

# ---- LOAD DATA ----
@st.cache_data(show_spinner=False)
def load_data(seasons):
    sched = nfl.import_schedules(seasons)
    pbp = nfl.import_pbp_data(seasons)
    sched = sched.dropna(subset=['home_score', 'away_score'])
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

seasons = st.multiselect("Seasons to load", options=list(range(2010, datetime.today().year + 1)), default=DEFAULT_SEASONS)
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
        team_games = team_game_stats[
            (team_game_stats['team'] == team) &
            ((team_game_stats['season'] < season) | ((team_game_stats['season'] == season) & (team_game_stats['week'] < week)))
        ].sort_values(['season', 'week']).tail(1)
        if not team_games.empty:
            row = team_games.iloc[-1]
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
season = st.selectbox("Season", seasons, index=len(seasons)-1)
week = st.number_input("Week (for rolling context)", min_value=1, max_value=22, value=18)

if home_team and away_team and home_team != away_team:
    X_pred = build_features_for_matchup(home_team, away_team, season, week)
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
