# import streamlit as st
# import pandas as pd
# import numpy as np
# import nfl_data_py as nfl
# from datetime import datetime
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
# from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
# from sklearn.neighbors import KNeighborsRegressor
# from sklearn.svm import SVR
# from sklearn.preprocessing import OneHotEncoder
# from xgboost import XGBRegressor, XGBClassifier
# from packaging import version
# import sklearn

# TOP_N_FEATURES = 40

# @st.cache_data(show_spinner=False)
# def fetch_nfl_data(seasons):
#     df = nfl.import_schedules(seasons)
#     df = df.dropna(subset=['home_score', 'away_score'])
#     return df

# @st.cache_data(show_spinner=False)
# def feature_engineering(df):
#     all_teams = pd.concat([df['home_team'], df['away_team']]).unique()
#     # Use correct OneHotEncoder argument for your scikit-learn version
#     if version.parse(sklearn.__version__) >= version.parse("1.2"):
#         encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
#     else:
#         encoder = OneHotEncoder(sparse=False, handle_unknown='ignore')
#     encoder.fit(all_teams.reshape(-1, 1))
#     home_team_enc = encoder.transform(df['home_team'].values.reshape(-1, 1))
#     away_team_enc = encoder.transform(df['away_team'].values.reshape(-1, 1))

#     df['home_advantage'] = 1
#     df['div_game'] = df['div_game'].astype(int)
#     game_type_enc = pd.get_dummies(df['game_type'], prefix='type')
#     roof_enc = pd.get_dummies(df['roof'], prefix='roof')
#     surface_enc = pd.get_dummies(df['surface'], prefix='surface')
#     home_coach_enc = pd.get_dummies(df['home_coach'], prefix='hcoach')
#     away_coach_enc = pd.get_dummies(df['away_coach'], prefix='acoach')
#     home_qb_enc = pd.get_dummies(df['home_qb_name'], prefix='hq')
#     away_qb_enc = pd.get_dummies(df['away_qb_name'], prefix='aq')
#     location_enc = pd.get_dummies(df['location'], prefix='loc')

#     for col in ['temp', 'wind', 'spread_line', 'total_line', 'away_moneyline', 'home_moneyline']:
#         df[col] = df[col].fillna(df[col].mean() if df[col].dtype != 'O' else 0)
#     df['away_rest'] = df['away_rest'].fillna(7)
#     df['home_rest'] = df['home_rest'].fillna(7)

#     team_stats, team_def_stats, team_win_stats = {}, {}, {}
#     for team in all_teams:
#         team_games = df[(df['home_team'] == team) | (df['away_team'] == team)].sort_values('gameday')
#         team_games['team_score'] = np.where(team_games['home_team'] == team, team_games['home_score'], team_games['away_score'])
#         team_games['team_score_avg'] = team_games['team_score'].rolling(window=3, min_periods=1).mean()
#         team_games['opp_score'] = np.where(team_games['home_team'] == team, team_games['away_score'], team_games['home_score'])
#         team_games['opp_score_avg'] = team_games['opp_score'].rolling(window=3, min_periods=1).mean()
#         team_games['win'] = np.where(
#             ((team_games['home_team'] == team) & (team_games['home_score'] > team_games['away_score'])) | 
#             ((team_games['away_team'] == team) & (team_games['away_score'] > team_games['home_score'])), 1, 0
#         )
#         team_games['win_rate'] = team_games['win'].rolling(window=3, min_periods=1).mean()
#         team_stats[team] = team_games.set_index('game_id')['team_score_avg']
#         team_def_stats[team] = team_games.set_index('game_id')['opp_score_avg']
#         team_win_stats[team] = team_games.set_index('game_id')['win_rate']

#     df['home_team_avg'] = df.apply(lambda row: team_stats[row['home_team']].get(row['game_id'], row['home_score']), axis=1)
#     df['away_team_avg'] = df.apply(lambda row: team_stats[row['away_team']].get(row['game_id'], row['away_score']), axis=1)
#     df['home_team_def_avg'] = df.apply(lambda row: team_def_stats[row['home_team']].get(row['game_id'], row['away_score']), axis=1)
#     df['away_team_def_avg'] = df.apply(lambda row: team_def_stats[row['away_team']].get(row['game_id'], row['home_score']), axis=1)
#     df['home_team_win_rate'] = df.apply(lambda row: team_win_stats[row['home_team']].get(row['game_id'], 0.5), axis=1)
#     df['away_team_win_rate'] = df.apply(lambda row: team_win_stats[row['away_team']].get(row['game_id'], 0.5), axis=1)

#     X = pd.concat([
#         pd.DataFrame(home_team_enc, index=df.index, columns=[f'home_{t}' for t in encoder.categories_[0]]),
#         pd.DataFrame(away_team_enc, index=df.index, columns=[f'away_{t}' for t in encoder.categories_[0]]),
#         game_type_enc,
#         roof_enc,
#         surface_enc,
#         home_coach_enc,
#         away_coach_enc,
#         home_qb_enc,
#         away_qb_enc,
#         location_enc,
#         df[['home_advantage', 'div_game', 'temp', 'wind', 'spread_line', 'total_line',
#             'away_moneyline', 'home_moneyline', 'away_rest', 'home_rest',
#             'home_team_avg', 'away_team_avg', 'home_team_def_avg', 'away_team_def_avg',
#             'home_team_win_rate', 'away_team_win_rate']]
#     ], axis=1)
#     y_home = df['home_score']
#     y_away = df['away_score']
#     return X, y_home, y_away, encoder, all_teams

# def select_features(X, y, top_n):
#     rfr = RandomForestRegressor(n_estimators=100, random_state=42)
#     rfr.fit(X, y)
#     importances = pd.Series(rfr.feature_importances_, index=X.columns)
#     selected_features = importances.nlargest(top_n).index.tolist()
#     return selected_features

# def build_features_for_matchup(home_team, away_team, encoder, df, all_possible_columns, team_avgs, team_def_avgs, team_win_avgs):
#     last_row = df.iloc[-1]
#     input_dict = {}

#     # Team encoding
#     home_enc = encoder.transform([[home_team]])[0]
#     away_enc = encoder.transform([[away_team]])[0]
#     for i, col in enumerate([f'home_{t}' for t in encoder.categories_[0]]):
#         input_dict[col] = home_enc[i]
#     for i, col in enumerate([f'away_{t}' for t in encoder.categories_[0]]):
#         input_dict[col] = away_enc[i]

#     # Categorical dummies: use mode or default
#     for col in all_possible_columns:
#         if col.startswith('type_'):
#             input_dict[col] = 0
#             if f'type_{last_row.get("game_type", "REG")}' == col:
#                 input_dict[col] = 1
#         elif col.startswith('roof_'):
#             input_dict[col] = 0
#             if f'roof_{last_row.get("roof", "outdoors")}' == col:
#                 input_dict[col] = 1
#         elif col.startswith('surface_'):
#             input_dict[col] = 0
#             if f'surface_{last_row.get("surface", "grass")}' == col:
#                 input_dict[col] = 1
#         elif col.startswith('hcoach_'):
#             input_dict[col] = 0
#             if f'hcoach_{last_row.get("home_coach", "")}' == col:
#                 input_dict[col] = 1
#         elif col.startswith('acoach_'):
#             input_dict[col] = 0
#             if f'acoach_{last_row.get("away_coach", "")}' == col:
#                 input_dict[col] = 1
#         elif col.startswith('hq_'):
#             input_dict[col] = 0
#             if f'hq_{last_row.get("home_qb_name", "")}' == col:
#                 input_dict[col] = 1
#         elif col.startswith('aq_'):
#             input_dict[col] = 0
#             if f'aq_{last_row.get("away_qb_name", "")}' == col:
#                 input_dict[col] = 1
#         elif col.startswith('loc_'):
#             input_dict[col] = 0
#             if f'loc_{last_row.get("location", "")}' == col:
#                 input_dict[col] = 1

#     # Numeric and engineered features
#     input_dict['home_advantage'] = 1
#     input_dict['div_game'] = 0
#     input_dict['temp'] = last_row.get('temp', 60)
#     input_dict['wind'] = last_row.get('wind', 5)
#     input_dict['spread_line'] = last_row.get('spread_line', 0)
#     input_dict['total_line'] = last_row.get('total_line', 45)
#     input_dict['away_moneyline'] = last_row.get('away_moneyline', 0)
#     input_dict['home_moneyline'] = last_row.get('home_moneyline', 0)
#     input_dict['away_rest'] = last_row.get('away_rest', 7)
#     input_dict['home_rest'] = last_row.get('home_rest', 7)

#     input_dict['home_team_avg'] = team_avgs.get(home_team, last_row.get('home_team_avg', 21))
#     input_dict['away_team_avg'] = team_avgs.get(away_team, last_row.get('away_team_avg', 21))
#     input_dict['home_team_def_avg'] = team_def_avgs.get(home_team, last_row.get('home_team_def_avg', 21))
#     input_dict['away_team_def_avg'] = team_def_avgs.get(away_team, last_row.get('away_team_def_avg', 21))
#     input_dict['home_team_win_rate'] = team_win_avgs.get(home_team, last_row.get('home_team_win_rate', 0.5))
#     input_dict['away_team_win_rate'] = team_win_avgs.get(away_team, last_row.get('away_team_win_rate', 0.5))

#     X_pred = pd.DataFrame([input_dict])
#     for col in all_possible_columns:
#         if col not in X_pred.columns:
#             X_pred[col] = 0
#     X_pred = X_pred[all_possible_columns]
#     X_pred = X_pred.fillna(0)
#     X_pred = X_pred.replace([np.inf, -np.inf], 0)
#     return X_pred

# st.title("NFL Game Winner & Score Predictor (Classification + Regression)")

# seasons = [2021, 2022, 2023, 2024]

# with st.spinner("Loading and training... (first run may take a minute)"):
#     df = fetch_nfl_data(seasons)
#     X, y_home, y_away, encoder, all_teams = feature_engineering(df)
#     all_possible_columns = X.columns.tolist()
#     team_avgs, team_def_avgs, team_win_avgs = {}, {}, {}
#     for team in all_teams:
#         scores = pd.concat([df[df['home_team']==team]['home_score'], df[df['away_team']==team]['away_score']])
#         opp_scores = pd.concat([df[df['home_team']==team]['away_score'], df[df['away_team']==team]['home_score']])
#         wins = pd.concat([df[(df['home_team']==team) & (df['home_score']>df['away_score'])]['game_id'],
#                           df[(df['away_team']==team) & (df['away_score']>df['home_score'])]['game_id']])
#         team_avgs[team] = scores.mean()
#         team_def_avgs[team] = opp_scores.mean()
#         team_win_avgs[team] = len(wins) / len(scores) if len(scores) > 0 else 0.5

#     # Two-stage targets
#     y_winner = (df['home_score'] > df['away_score']).astype(int)
#     y_margin = df['home_score'] - df['away_score']
#     y_total = df['home_score'] + df['away_score']

#     # Feature selection for regression
#     selected_features_margin = select_features(X, y_margin, TOP_N_FEATURES)
#     selected_features_total = select_features(X, y_total, TOP_N_FEATURES)

#     X_margin_selected = X[selected_features_margin]
#     X_total_selected = X[selected_features_total]

#     # Winner classifier (using all features for best separation)
#     clf = RandomForestClassifier(n_estimators=100, random_state=42)
#     clf.fit(X, y_winner)

#     # Margin regressor
#     margin_models = [
#         RandomForestRegressor(n_estimators=100, random_state=42),
#         XGBRegressor(n_estimators=100, random_state=42),
#         GradientBoostingRegressor(n_estimators=100, random_state=42),
#         HistGradientBoostingRegressor(random_state=42),
#         AdaBoostRegressor(n_estimators=100, random_state=42),
#         LinearRegression(),
#         Ridge(),
#         Lasso(),
#         ElasticNet(),
#         KNeighborsRegressor(),
#         SVR()
#     ]
#     for model in margin_models:
#         model.fit(X_margin_selected, y_margin)

#     # Total regressor
#     total_models = [
#         RandomForestRegressor(n_estimators=100, random_state=42),
#         XGBRegressor(n_estimators=100, random_state=42),
#         GradientBoostingRegressor(n_estimators=100, random_state=42),
#         HistGradientBoostingRegressor(random_state=42),
#         AdaBoostRegressor(n_estimators=100, random_state=42),
#         LinearRegression(),
#         Ridge(),
#         Lasso(),
#         ElasticNet(),
#         KNeighborsRegressor(),
#         SVR()
#     ]
#     for model in total_models:
#         model.fit(X_total_selected, y_total)

#     # Estimate residual std for margin and total for noise
#     def get_residual_std(models, X, y):
#         preds = np.column_stack([model.predict(X) for model in models])
#         ensemble_pred = preds.mean(axis=1)
#         residuals = y - ensemble_pred
#         return residuals.std()
#     resid_std_margin = get_residual_std(margin_models, X_margin_selected, y_margin)
#     resid_std_total = get_residual_std(total_models, X_total_selected, y_total)

# def ensemble_predict(models, X, noise_std=0):
#     preds = np.column_stack([model.predict(X) for model in models])
#     mean_pred = preds.mean(axis=1)
#     if noise_std > 0:
#         noise = np.random.normal(0, noise_std, size=mean_pred.shape)
#         mean_pred += noise
#     return mean_pred

# def predict_game(X_pred_full):
#     # Winner classification
#     proba = clf.predict_proba(X_pred_full)[0]
#     pred_winner = clf.predict(X_pred_full)[0]  # 1 = home, 0 = away

#     # Margin regression
#     X_margin_pred = X_pred_full[selected_features_margin]
#     pred_margin = ensemble_predict(margin_models, X_margin_pred, noise_std=resid_std_margin)[0]

#     # Total regression
#     X_total_pred = X_pred_full[selected_features_total]
#     pred_total = ensemble_predict(total_models, X_total_pred, noise_std=resid_std_total)[0]

#     # If classifier says home wins, margin is positive; else, flip sign
#     if pred_winner == 0:
#         pred_margin = -abs(pred_margin)
#     else:
#         pred_margin = abs(pred_margin)

#     # Calculate scores
#     home_score = (pred_total + pred_margin) / 2
#     away_score = (pred_total - pred_margin) / 2

#     # Clamp negatives, round to 1 decimal
#     home_score = max(0, round(home_score, 1))
#     away_score = max(0, round(away_score, 1))

#     return {
#         "winner": "Home" if pred_winner == 1 else "Away",
#         "predicted_home_score": home_score,
#         "predicted_away_score": away_score,
#         "home_win_proba": proba[1],
#         "away_win_proba": proba[0]
#     }

# today = datetime.today().date()

# schedule_df = nfl.import_schedules([today.year])
# schedule_df['gameday'] = pd.to_datetime(schedule_df['gameday']).dt.date
# todays_games = schedule_df[schedule_df['gameday'] == today]

# if len(todays_games) == 0:
#     st.info("No NFL games today.")
# else:
#     st.subheader(f"Predictions for today's NFL games ({today}):")
#     pred_rows = []
#     for i, row in todays_games.iterrows():
#         home_team = row['home_team']
#         away_team = row['away_team']
#         X_pred_full = build_features_for_matchup(
#             home_team, away_team, encoder, df,
#             all_possible_columns, team_avgs, team_def_avgs, team_win_avgs
#         )
#         result = predict_game(X_pred_full)
#         pred_rows.append({
#             "Away Team": away_team,
#             "Home Team": home_team,
#             "Predicted Away Score": result["predicted_away_score"],
#             "Predicted Home Score": result["predicted_home_score"],
#             "Predicted Winner": result["winner"],
#             "Home Win Prob": f"{100*result['home_win_proba']:.1f}%",
#             "Away Win Prob": f"{100*result['away_win_proba']:.1f}%"
#         })
#     st.dataframe(pd.DataFrame(pred_rows))
#     st.caption("Scores are model predictions based on classification (winner) + regression (margin/total) with ensemble and noise.")

# st.markdown("---")
# st.subheader("Manual Matchup Prediction")
# team_list = sorted(list(all_teams))
# home_team = st.selectbox("Select Home Team", team_list)
# away_team = st.selectbox("Select Away Team", team_list, index=1 if team_list[1] != home_team else 2)

# if home_team and away_team and home_team != away_team:
#     X_pred_full = build_features_for_matchup(
#         home_team, away_team, encoder, df,
#         all_possible_columns, team_avgs, team_def_avgs, team_win_avgs
#     )
#     result = predict_game(X_pred_full)
#     st.success(
#         f"Prediction: {away_team} @ {home_team}: {result['predicted_away_score']} - {result['predicted_home_score']}  \n"
#         f"Winner: {result['winner']} team  \n"
#         f"Home Win Probability: {100*result['home_win_proba']:.1f}%  \n"
#         f"Away Win Probability: {100*result['away_win_proba']:.1f}%"
#     )
# elif home_team == away_team:
#     st.warning("Select different teams!")

# st.caption("Powered by nfl_data_py and scikit-learn. Two-stage prediction: winner (classification) + margin/total (regression).")

import streamlit as st
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, HistGradientBoostingRegressor, AdaBoostRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import KFold
from packaging import version
import sklearn

TOP_N_FEATURES = 40
STACK_FOLDS = 5  # For stacking

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
    X_pred = X_pred.fillna(0)
    X_pred = X_pred.replace([np.inf, -np.inf], 0)
    return X_pred

st.title("NFL Game Winner & Score Predictor (Stacked Ensemble + More Seasons)")

# --- More Seasons! ---
seasons = list(range(2010, datetime.today().year + 1))

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

    # Targets
    y_winner = (df['home_score'] > df['away_score']).astype(int)
    y_margin = df['home_score'] - df['away_score']
    y_total = df['home_score'] + df['away_score']

    # Feature selection
    selected_features_margin = select_features(X, y_margin, TOP_N_FEATURES)
    selected_features_total = select_features(X, y_total, TOP_N_FEATURES)
    selected_features_cls = select_features(X, y_winner, TOP_N_FEATURES)

    X_margin_selected = X[selected_features_margin]
    X_total_selected = X[selected_features_total]
    X_cls_selected = X[selected_features_cls]

    # --- STACKING ENSEMBLE CONSTRUCTION --- #
    # Base models
    base_clf_models = [
        RandomForestClassifier(n_estimators=120, random_state=42),
        XGBClassifier(n_estimators=100, random_state=42, use_label_encoder=False, eval_metric="logloss"),
        LogisticRegression(max_iter=10000, random_state=42)
    ]
    base_margin_models = [
        RandomForestRegressor(n_estimators=120, random_state=42),
        XGBRegressor(n_estimators=120, random_state=42),
        GradientBoostingRegressor(n_estimators=120, random_state=42),
        Ridge(),
        ElasticNet()
    ]
    base_total_models = [
        RandomForestRegressor(n_estimators=120, random_state=42),
        XGBRegressor(n_estimators=120, random_state=42),
        GradientBoostingRegressor(n_estimators=120, random_state=42),
        Ridge(),
        ElasticNet()
    ]

    # Out-of-fold stacking
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

    # For final prediction, fit on all data:
    def fit_base_models(X, y, model_list, problem_type="reg"):
        for model in model_list:
            model.fit(X, y)
        return model_list

    # Classification stacking
    X_cls_stack = get_stacking_preds(X_cls_selected, y_winner, base_clf_models, problem_type="cls")
    meta_clf = LogisticRegression(max_iter=10000, random_state=42)
    meta_clf.fit(X_cls_stack, y_winner)
    fit_base_models(X_cls_selected, y_winner, base_clf_models, "cls")

    # Margin stacking
    X_margin_stack = get_stacking_preds(X_margin_selected, y_margin, base_margin_models, problem_type="reg")
    meta_margin = Ridge()
    meta_margin.fit(X_margin_stack, y_margin)
    fit_base_models(X_margin_selected, y_margin, base_margin_models, "reg")

    # Total stacking
    X_total_stack = get_stacking_preds(X_total_selected, y_total, base_total_models, problem_type="reg")
    meta_total = Ridge()
    meta_total.fit(X_total_stack, y_total)
    fit_base_models(X_total_selected, y_total, base_total_models, "reg")

    # Estimate residual std for margin and total for noise
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
    win_proba = stacking_predict(base_clf_models, meta_clf, X_cls_pred, problem_type="cls")[0]
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

    # Clamp negatives, round to 1 decimal
    home_score = max(0, round(home_score, 1))
    away_score = max(0, round(away_score, 1))

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

# Only keep games not yet played
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

st.caption("Powered by nfl_data_py and scikit-learn. Stacked ensemble (meta-model) with historical data since 2010.")