import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold
import sklearn
from packaging import version

# ---- ADVANCED FEATURE ENGINEERING ----
@st.cache_data(show_spinner=False)
def fetch_nfl_data(seasons):
    # Load schedules and play-by-play
    sched = nfl.import_schedules(seasons)
    pbp = nfl.import_pbp_data(seasons)
    sched = sched.dropna(subset=['home_score', 'away_score'])
    # Filter pbp for relevant games only
    pbp = pbp[pbp['season'].isin(seasons)]
    return sched, pbp

def summarize_team_season(pbp, season):
    # Aggregate per-team, per-season stats
    teams = pd.concat([pbp['posteam'], pbp['defteam']]).dropna().unique()
    summary = []
    for team in teams:
        team_pbp_off = pbp[pbp['posteam'] == team]
        team_pbp_def = pbp[pbp['defteam'] == team]

        # Turnover Differential (per game)
        takeaways = ((team_pbp_def['interception'] == 1) | (team_pbp_def['fumble_lost'] == 1)).sum()
        giveaways = ((team_pbp_off['interception'] == 1) | (team_pbp_off['fumble_lost'] == 1)).sum()
        games_played = len(team_pbp_off['game_id'].unique())
        turnover_diff = (takeaways - giveaways) / games_played if games_played else 0

        # QB Efficiency (EPA/play, Passer Rating Differential not included)
        qb_eff = team_pbp_off['epa'].mean()

        # Yards per Play Differential
        off_ypp = team_pbp_off['yards_gained'].mean()
        def_ypp = team_pbp_def['yards_gained'].mean()
        ypp_diff = (off_ypp - def_ypp) if off_ypp and def_ypp else 0

        # Success Rate Differential (EPA > 0)
        off_succ = (team_pbp_off['epa'] > 0).mean()
        def_succ = (team_pbp_def['epa'] > 0).mean()
        success_rate_diff = off_succ - def_succ

        # Explosive Plays Differential (20+ yard gains)
        off_exp = (team_pbp_off['yards_gained'] >= 20).sum() / games_played if games_played else 0
        def_exp = (team_pbp_def['yards_gained'] >= 20).sum() / games_played if games_played else 0
        explosive_diff = off_exp - def_exp

        # Pressure/Sack Rate Differential
        off_dropbacks = team_pbp_off['pass_attempt'].sum()
        def_dropbacks = team_pbp_def['pass_attempt'].sum()
        off_sacks = team_pbp_off['sack'].sum()
        def_sacks = team_pbp_def['sack'].sum()
        off_sack_rate = (off_sacks / off_dropbacks) if off_dropbacks else 0
        def_sack_rate = (def_sacks / def_dropbacks) if def_dropbacks else 0
        sack_rate_diff = def_sack_rate - off_sack_rate

        # Red Zone Efficiency Differential (TDs / trips inside 20)
        off_rz_trips = team_pbp_off[(team_pbp_off['yardline_100'] <= 20) & (team_pbp_off['down'] == 1)]
        off_rz_tds = off_rz_trips['touchdown'].sum()
        off_rz_pct = (off_rz_tds / len(off_rz_trips)) if len(off_rz_trips) > 0 else 0
        def_rz_trips = team_pbp_def[(team_pbp_def['yardline_100'] <= 20) & (team_pbp_def['down'] == 1)]
        def_rz_tds = def_rz_trips['touchdown'].sum()
        def_rz_pct = (def_rz_tds / len(def_rz_trips)) if len(def_rz_trips) > 0 else 0
        rz_diff = off_rz_pct - def_rz_pct

        # Third Down Conversion % Differential
        off_3rd = team_pbp_off[team_pbp_off['down'] == 3]
        off_3rd_conv = (off_3rd['first_down'] == 1).mean() if len(off_3rd) > 0 else 0
        def_3rd = team_pbp_def[team_pbp_def['down'] == 3]
        def_3rd_conv = (def_3rd['first_down'] == 1).mean() if len(def_3rd) > 0 else 0
        third_down_diff = off_3rd_conv - def_3rd_conv

        # Starting Field Position Differential
        if 'yardline_100' in team_pbp_off.columns:
            off_fp = 100 - team_pbp_off.groupby('game_id').first()['yardline_100'].mean()
        else:
            off_fp = np.nan
        if 'yardline_100' in team_pbp_def.columns:
            def_fp = 100 - team_pbp_def.groupby('game_id').first()['yardline_100'].mean()
        else:
            def_fp = np.nan
        field_pos_diff = (off_fp - def_fp) if pd.notnull(off_fp) and pd.notnull(def_fp) else 0

        # Special Teams Efficiency (FG% + Net Punt, rough proxy)
        off_fg = team_pbp_off[(team_pbp_off['play_type'] == 'field_goal')]
        fg_made = (off_fg['field_goal_result'] == 'made').sum()
        fg_att = len(off_fg)
        fg_pct = (fg_made / fg_att) if fg_att else 0
        punts = team_pbp_off[team_pbp_off['play_type'] == 'punt']
        net_punt = (punts['punt_net_yards']).mean() if len(punts) > 0 and 'punt_net_yards' in punts.columns else 0
        special_teams = fg_pct + (net_punt / 100 if net_punt else 0)

        # Rushing Efficiency Differential
        off_rush = team_pbp_off[team_pbp_off['play_type'] == 'run']
        def_rush = team_pbp_def[team_pbp_def['play_type'] == 'run']
        off_rush_ypp = off_rush['yards_gained'].mean() if len(off_rush) > 0 else 0
        def_rush_ypp = def_rush['yards_gained'].mean() if len(def_rush) > 0 else 0
        rush_yards_diff = off_rush_ypp - def_rush_ypp

        # Time of Possession Differential (average per game, minutes)
        if 'game_seconds_remaining' in team_pbp_off.columns:
            off_top = team_pbp_off.groupby('game_id')['drive_time_sec'].sum().mean() / 60
        else:
            off_top = np.nan
        if 'game_seconds_remaining' in team_pbp_def.columns:
            def_top = team_pbp_def.groupby('game_id')['drive_time_sec'].sum().mean() / 60
        else:
            def_top = np.nan
        top_diff = (off_top - def_top) if pd.notnull(off_top) and pd.notnull(def_top) else 0

        # Penalty Yards Differential (per game)
        off_pen = team_pbp_off[team_pbp_off['penalty'] == 1]['penalty_yards'].sum() / games_played if games_played else 0
        def_pen = team_pbp_def[team_pbp_def['penalty'] == 1]['penalty_yards'].sum() / games_played if games_played else 0
        penalty_diff = off_pen - def_pen

        summary.append({
            'season': season,
            'team': team,
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
    return pd.DataFrame(summary)

@st.cache_data(show_spinner=False)
def build_feature_df(sched, pbp, seasons):
    # For each season, get advanced features for all teams
    team_stats = []
    for season in seasons:
        team_stats.append(summarize_team_season(pbp[pbp['season'] == season], season))
    team_stats_df = pd.concat(team_stats)
    # Map advanced features to each game in sched
    merged = sched.copy()
    for col in [
        'turnover_diff', 'qb_efficiency', 'ypp_diff', 'success_rate_diff', 'explosive_diff', 'sack_rate_diff',
        'rz_diff', 'third_down_diff', 'field_pos_diff', 'special_teams', 'rush_yards_diff', 'top_diff', 'penalty_diff'
    ]:
        merged = merged.merge(
            team_stats_df[['season', 'team', col]].rename(columns={col: 'home_' + col}),
            left_on=['season', 'home_team'],
            right_on=['season', 'team'],
            how='left'
        )
        merged = merged.merge(
            team_stats_df[['season', 'team', col]].rename(columns={col: 'away_' + col}),
            left_on=['season', 'away_team'],
            right_on=['season', 'team'],
            how='left'
        )
    # Compute differentials (home - away) for each feature
    feature_cols = []
    for col in [
        'turnover_diff', 'qb_efficiency', 'ypp_diff', 'success_rate_diff', 'explosive_diff', 'sack_rate_diff',
        'rz_diff', 'third_down_diff', 'field_pos_diff', 'special_teams', 'rush_yards_diff', 'top_diff', 'penalty_diff'
    ]:
        merged[f'{col}_diff'] = merged[f'home_{col}'] - merged[f'away_{col}']
        feature_cols.append(f'{col}_diff')
    X = merged[feature_cols].fillna(0)
    y_home = merged['home_score']
    y_away = merged['away_score']
    return X, y_home, y_away, merged, feature_cols

# ---- STREAMLIT APP ----

st.title("NFL Game Winner & Score Predictor (Advanced Features)")

seasons = list(range(2010, datetime.today().year + 1))

with st.spinner("Loading and training... (first run may take a minute)"):
    sched, pbp = fetch_nfl_data(seasons)
    X, y_home, y_away, merged, feature_cols = build_feature_df(sched, pbp, seasons)

    # Model setup: winner classification, margin, total
    y_winner = (y_home > y_away).astype(int)
    y_margin = y_home - y_away
    y_total = y_home + y_away

    # Fit models (simple random forest/ensemble)
    clf = RandomForestClassifier(n_estimators=120, random_state=42)
    clf.fit(X, y_winner)
    margin_reg = RandomForestRegressor(n_estimators=120, random_state=42)
    margin_reg.fit(X, y_margin)
    total_reg = RandomForestRegressor(n_estimators=120, random_state=42)
    total_reg.fit(X, y_total)

    # For prediction: compute features for a given matchup
    def get_team_season_stats(team, season):
        row = merged[(merged['season'] == season) & (merged['home_team'] == team)]
        if len(row) == 0:
            row = merged[(merged['season'] == season) & (merged['away_team'] == team)]
        if len(row) == 0:
            # Fallback to latest available
            row = merged[(merged['home_team'] == team) | (merged['away_team'] == team)].iloc[-1:]
        return row.iloc[0] if len(row) > 0 else None

    def build_features_for_matchup(home_team, away_team, season):
        home_stats = get_team_season_stats(home_team, season)
        away_stats = get_team_season_stats(away_team, season)
        feats = {}
        for col in [
            'turnover_diff', 'qb_efficiency', 'ypp_diff', 'success_rate_diff', 'explosive_diff', 'sack_rate_diff',
            'rz_diff', 'third_down_diff', 'field_pos_diff', 'special_teams', 'rush_yards_diff', 'top_diff', 'penalty_diff'
        ]:
            h_val = home_stats[f'home_{col}'] if home_stats is not None and f'home_{col}' in home_stats else 0
            a_val = away_stats[f'home_{col}'] if away_stats is not None and f'home_{col}' in away_stats else 0
            feats[f'{col}_diff'] = h_val - a_val
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
        X_pred = build_features_for_matchup(home_team, away_team, season)
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

if home_team and away_team and home_team != away_team:
    X_pred = build_features_for_matchup(home_team, away_team, season)
    result = predict_game(X_pred)
    st.success(
        f"Prediction: {away_team} @ {home_team}: {result['predicted_away_score']} - {result['predicted_home_score']}  \n"
        f"Winner: {result['winner']} team  \n"
        f"Home Win Probability: {100*result['home_win_proba']:.1f}%  \n"
        f"Away Win Probability: {100*result['away_win_proba']:.1f}%"
    )
elif home_team == away_team:
    st.warning("Select different teams!")

st.caption("Powered by nfl_data_py and scikit-learn. Features: turnover, efficiency, explosive, 3rd down, sack, RZ, field pos, special teams, rushing, time, penalty differentials.")
