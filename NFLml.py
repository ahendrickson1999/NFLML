# import os
# import pandas as pd
# import numpy as np
# import nfl_data_py as nfl
# from datetime import datetime
# from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.metrics import accuracy_score, mean_absolute_error

# # Ensemble & tuning imports
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, StackingClassifier, StackingRegressor
# from sklearn.linear_model import LogisticRegression, Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline
# import joblib
# import scipy.stats as stats

# def add_vegas_lines(games):
#     if 'spread_line' not in games.columns:
#         games['spread_line'] = 0.0
#     if 'total_line' not in games.columns:
#         games['total_line'] = 44.0
#     games['spread_line'] = games['spread_line'].fillna(0.0)
#     games['total_line'] = games['total_line'].fillna(44.0)
#     return games

# def get_date_col(games):
#     for candidate in ['gametime', 'start_time', 'datetime', 'game_date', 'start_date', 'date']:
#         if candidate in games.columns:
#             return candidate
#     raise Exception(f"No valid date column found in games columns: {games.columns.tolist()}")

# def compute_team_game_stats(pbp):
#     pbp = pbp[~pbp['season_type'].isin(['PRE'])]
#     pbp = pbp[pbp['posteam'].notnull() & pbp['defteam'].notnull()]
#     pbp['explosive_play'] = ((pbp['passing_yards'] >= 20) | (pbp['rushing_yards'] >= 20)).astype(int)
#     pbp['is_red_zone'] = pbp['yardline_100'] <= 20
#     pbp['is_third_down'] = pbp['down'] == 3

#     # Third down conversions: made if first down gained or TD
#     pbp['third_down_converted'] = (
#         (pbp['is_third_down']) &
#         ((pbp.get('first_down', 0) == 1) | (pbp.get('touchdown', 0) == 1))
#     ).astype(int)

#     agg_stats = pbp.groupby(['game_id', 'posteam']).agg(
#         points_scored=('touchdown', 'sum'),
#         pass_yards=('passing_yards', 'sum'),
#         rush_yards=('rushing_yards', 'sum'),
#         turnovers=('interception', 'sum'),
#         fumbles=('fumble_lost', 'sum'),
#         explosive_plays=('explosive_play', 'sum'),
#         total_plays=('play_id', 'count'),
#         red_zone_plays=('is_red_zone', 'sum'),
#         red_zone_tds=('touchdown', lambda x: x[pbp.loc[x.index, 'yardline_100'] <= 20].sum() if len(x) > 0 else 0),
#         third_down_plays=('is_third_down', 'sum'),
#         third_down_conversions=('third_down_converted', 'sum'),
#         sacks=('sack', 'sum')
#     ).reset_index()
#     agg_stats['total_yards'] = agg_stats['pass_yards'] + agg_stats['rush_yards']
#     agg_stats['turnovers'] = agg_stats['turnovers'] + agg_stats['fumbles']
#     # Rates
#     agg_stats['explosive_play_rate'] = agg_stats['explosive_plays'] / agg_stats['total_plays'].replace(0, np.nan)
#     agg_stats['red_zone_eff'] = agg_stats['red_zone_tds'] / agg_stats['red_zone_plays'].replace(0, np.nan)
#     agg_stats['third_down_conv_rate'] = agg_stats['third_down_conversions'] / agg_stats['third_down_plays'].replace(0, np.nan)
#     # Defensive stats for margin
#     defense = agg_stats.rename(columns={
#         "posteam": "defteam",
#         "turnovers": "def_turnovers",
#         "sacks": "def_sacks"
#     })[['game_id','defteam','def_turnovers','def_sacks']]
#     return agg_stats, defense

# def get_rest_days(games, date_col):
#     rest = []
#     for team in set(games['home_team']).union(set(games['away_team'])):
#         tgames = games[(games['home_team'] == team) | (games['away_team'] == team)].sort_values(['season', 'week'])
#         last_date = None
#         for idx, row in tgames.iterrows():
#             gdate = pd.to_datetime(row[date_col])
#             if last_date is None:
#                 rest.append((row['game_id'], team, 7))
#             else:
#                 diff = (gdate - last_date).days
#                 rest.append((row['game_id'], team, diff if diff > 0 else 7))
#             last_date = gdate
#     rest_df = pd.DataFrame(rest, columns=['game_id', 'team', 'rest_days'])
#     return rest_df

# def build_features(games, pbp):
#     games = add_vegas_lines(games)
#     date_col = get_date_col(games)
#     agg_stats, defense = compute_team_game_stats(pbp)

#     # Prepare long-form DataFrame
#     home = games[['game_id','season','week',date_col,'home_team','away_team','home_score','away_score','spread_line','total_line']].rename(
#         columns={date_col:'date','home_team':'team','away_team':'opp','home_score':'points_scored','away_score':'opp_points'}
#     )
#     home['is_home'] = 1
#     away = games[['game_id','season','week',date_col,'away_team','home_team','away_score','home_score','spread_line','total_line']].rename(
#         columns={date_col:'date','away_team':'team','home_team':'opp','away_score':'points_scored','home_score':'opp_points'}
#     )
#     away['is_home'] = 0
#     long_games = pd.concat([home, away], ignore_index=True)

#     # Merge in aggregated per-team stats
#     long_games['team'] = long_games['team'].astype(str)
#     agg_stats['posteam'] = agg_stats['posteam'].astype(str)
#     long_games['game_id'] = long_games['game_id'].astype(str)
#     agg_stats['game_id'] = agg_stats['game_id'].astype(str)
#     long_games = long_games.merge(agg_stats, how='left', left_on=['game_id','team'], right_on=['game_id','posteam'])

#     # Defensive stats for margin (opponent)
#     defense['defteam'] = defense['defteam'].astype(str)
#     long_games = long_games.merge(defense, how='left', left_on=['game_id','opp'], right_on=['game_id','defteam'], suffixes=('', '_opp'))

#     # Defensive fillna for all engineered columns
#     for col in ['points_scored', 'total_yards', 'turnovers', 'explosive_play_rate', 'red_zone_eff', 'third_down_conv_rate', 'sacks', 'def_sacks', 'def_turnovers']:
#         if col not in long_games.columns:
#             long_games[col] = 0
#         long_games[col] = long_games[col].fillna(0)

#     long_games = long_games.sort_values(['team','season','week'])
#     # Rolling stats (advanced) - using window=3 like original code (name kept _rolling5 for backwards compatibility)
#     for stat in ['points_scored','total_yards','turnovers','explosive_play_rate','red_zone_eff','third_down_conv_rate','sacks','def_sacks','def_turnovers']:
#         long_games[f'{stat}_rolling5'] = long_games.groupby('team')[stat].rolling(3, min_periods=1).mean().reset_index(0,drop=True)

#     rest_df = get_rest_days(games, date_col)
#     long_games = long_games.merge(rest_df, how='left', on=['game_id','team'])

#     features = []
#     for _, row in games.iterrows():
#         def get_last(team, week, season, is_home):
#             prev = long_games[
#                 (long_games['team'] == team) &
#                 ((long_games['season'] < season) | ((long_games['season'] == season) & (long_games['week'] < week)))
#             ]
#             prev = prev[prev['is_home'] == is_home] if not prev.empty else prev
#             return prev.iloc[-1] if not prev.empty else {}
#         home_last = get_last(row['home_team'], row['week'], row['season'], 1)
#         away_last = get_last(row['away_team'], row['week'], row['season'], 0)
#         features.append({
#             'game_id': row['game_id'],
#             'season': row['season'],
#             'week': row['week'],
#             'home_team': row['home_team'],
#             'away_team': row['away_team'],
#             'home_points_avg': home_last.get('points_scored_rolling5', 0),
#             'away_points_avg': away_last.get('points_scored_rolling5', 0),
#             'home_yards_avg': home_last.get('total_yards_rolling5', 0),
#             'away_yards_avg': away_last.get('total_yards_rolling5', 0),
#             'home_tov_avg': home_last.get('turnovers_rolling5', 0),
#             'away_tov_avg': away_last.get('turnovers_rolling5', 0),
#             'home_rest': home_last.get('rest_days', 7),
#             'away_rest': away_last.get('rest_days', 7),
#             'home_is_favorite': int(row['spread_line'] < 0),
#             'spread': abs(row['spread_line']),
#             'over_under': row['total_line'],
#             'home_explosive_rate': home_last.get('explosive_play_rate_rolling5', 0),
#             'away_explosive_rate': away_last.get('explosive_play_rate_rolling5', 0),
#             'home_rz_eff': home_last.get('red_zone_eff_rolling5', 0),
#             'away_rz_eff': away_last.get('red_zone_eff_rolling5', 0),
#             'home_3rd_down': home_last.get('third_down_conv_rate_rolling5', 0),
#             'away_3rd_down': away_last.get('third_down_conv_rate_rolling5', 0),
#             'home_sacks': home_last.get('sacks_rolling5', 0),
#             'away_sacks': away_last.get('sacks_rolling5', 0),
#             'home_def_turnover_margin': home_last.get('def_turnovers_rolling5', 0) - home_last.get('turnovers_rolling5', 0),
#             'away_def_turnover_margin': away_last.get('def_turnovers_rolling5', 0) - away_last.get('turnovers_rolling5', 0),
#             'home_score': row.get('home_score', np.nan),
#             'away_score': row.get('away_score', np.nan)
#         })
#     df = pd.DataFrame(features)
#     df = df.dropna(subset=['home_score','away_score'])
#     df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
#     df['total_points'] = df['home_score'] + df['away_score']
#     df['spread_actual'] = df['home_score'] - df['away_score']
#     return df

# def _time_train_test_split(df, X, y_cols, test_size=0.15):
#     """
#     Time-based train/test split: sorts by season/week and splits last fraction as test.
#     y_cols: list of y series corresponding to X columns order (list of pandas Series)
#     Returns: X_train, X_test, list(y_train...), list(y_test...)
#     """
#     df_sorted = df.sort_values(['season', 'week']).reset_index(drop=True)
#     n = len(df_sorted)
#     split_at = int(n * (1 - test_size))
#     train_idx = df_sorted.index[:split_at]
#     test_idx = df_sorted.index[split_at:]
#     X_train = X.loc[train_idx].reset_index(drop=True)
#     X_test = X.loc[test_idx].reset_index(drop=True)
#     y_trains = []
#     y_tests = []
#     for y in y_cols:
#         y_trains.append(y.loc[train_idx].reset_index(drop=True))
#         y_tests.append(y.loc[test_idx].reset_index(drop=True))
#     return X_train, X_test, y_trains, y_tests

# def tune_classifiers(X, y, n_iter=20, cv_splits=5, random_state=42):
#     """Tune XGBClassifier and RandomForestClassifier using RandomizedSearchCV (time-series CV)."""
#     tscv = TimeSeriesSplit(n_splits=cv_splits)
#     results = {}

#     xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, verbosity=0)
#     xgb_param_dist = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [3, 6, 8, 10],
#         'learning_rate': [0.01, 0.05, 0.08, 0.1],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0]
#     }
#     rs_xgb = RandomizedSearchCV(xgb, xgb_param_dist, n_iter=n_iter, cv=tscv, scoring='accuracy', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_xgb.fit(X, y)
#     results['xgb'] = rs_xgb.best_estimator_

#     rf = RandomForestClassifier(random_state=random_state)
#     rf_param_dist = {
#         'n_estimators': [100, 200, 400],
#         'max_depth': [None, 6, 10, 15],
#         'max_features': ['auto', 'sqrt', 'log2']
#     }
#     rs_rf = RandomizedSearchCV(rf, rf_param_dist, n_iter=n_iter, cv=tscv, scoring='accuracy', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_rf.fit(X, y)
#     results['rf'] = rs_rf.best_estimator_

#     return results

# def tune_regressors(X, y, n_iter=20, cv_splits=5, random_state=42):
#     """Tune XGBRegressor, RandomForestRegressor, GradientBoostingRegressor using RandomizedSearchCV (time-series CV)."""
#     tscv = TimeSeriesSplit(n_splits=cv_splits)
#     results = {}

#     xgb = XGBRegressor(random_state=random_state, verbosity=0)
#     xgb_param_dist = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [3, 6, 8, 10],
#         'learning_rate': [0.01, 0.05, 0.08, 0.1],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0]
#     }
#     rs_xgb = RandomizedSearchCV(xgb, xgb_param_dist, n_iter=n_iter, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_xgb.fit(X, y)
#     results['xgb'] = rs_xgb.best_estimator_

#     rf = RandomForestRegressor(random_state=random_state)
#     rf_param_dist = {
#         'n_estimators': [100, 200, 400],
#         'max_depth': [None, 6, 10, 15],
#         'max_features': ['auto', 'sqrt', 'log2']
#     }
#     rs_rf = RandomizedSearchCV(rf, rf_param_dist, n_iter=n_iter, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_rf.fit(X, y)
#     results['rf'] = rs_rf.best_estimator_

#     gbr = GradientBoostingRegressor(random_state=random_state)
#     gbr_param_dist = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [3, 5, 6, 8],
#         'learning_rate': [0.01, 0.05, 0.08, 0.1],
#         'subsample': [0.6, 0.8, 1.0]
#     }
#     rs_gbr = RandomizedSearchCV(gbr, gbr_param_dist, n_iter=n_iter, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_gbr.fit(X, y)
#     results['gbr'] = rs_gbr.best_estimator_

#     return results

# def train_models(df, tune=True, n_iter=20, cv_splits=5, save_models=False, model_dir='models', random_state=42, test_size=0.25):
#     """
#     Train ensembles with optional hyperparameter tuning.
#     - tune: run randomized tuning for base estimators and use their best versions in stacking.
#     - n_iter: number of parameter settings sampled in RandomizedSearchCV.
#     - cv_splits: number of TimeSeriesSplit folds.
#     - save_models: if True, saves the resulting pipelines to model_dir.
#     """
#     feature_cols = [
#         'home_points_avg','away_points_avg','home_yards_avg','away_yards_avg',
#         'home_tov_avg','away_tov_avg','home_rest','away_rest','home_is_favorite',
#         'spread','over_under',
#         # Advanced
#         'home_explosive_rate', 'away_explosive_rate',
#         'home_rz_eff', 'away_rz_eff',
#         'home_3rd_down', 'away_3rd_down',
#         'home_sacks', 'away_sacks',
#         'home_def_turnover_margin', 'away_def_turnover_margin'
#     ]
#     X = df[feature_cols].fillna(0).reset_index(drop=True)
#     y_cls = df['home_win'].reset_index(drop=True)
#     y_reg = df['total_points'].reset_index(drop=True)
#     y_spread = df['spread_actual'].reset_index(drop=True)

#     # Time-based train/test split
#     X_train, X_test, [y_cls_train, y_reg_train, y_spread_train], [y_cls_test, y_reg_test, y_spread_test] = _time_train_test_split(
#         df.reset_index(drop=True),
#         X,
#         [y_cls, y_reg, y_spread],
#         test_size=test_size
#     )

#     # Optionally tune base estimators
#     if tune:
#         print("Tuning classifier base estimators (this may take a while)...")
#         cls_best = tune_classifiers(X_train, y_cls_train, n_iter=n_iter, cv_splits=cv_splits, random_state=random_state)
#         print("Tuning regressors for total points...")
#         reg_best = tune_regressors(X_train, y_reg_train, n_iter=n_iter, cv_splits=cv_splits, random_state=random_state)
#         print("Tuning regressors for spread...")
#         spread_best = tune_regressors(X_train, y_spread_train, n_iter=max(8, int(n_iter/2)), cv_splits=cv_splits, random_state=random_state)
#     else:
#         # default untuned estimators
#         cls_best = {
#             'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=8, learning_rate=0.08, random_state=random_state),
#             'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state)
#         }
#         reg_best = {
#             'xgb': XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.1, random_state=random_state),
#             'rf': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=random_state),
#             'gbr': GradientBoostingRegressor(n_estimators=200, max_depth=6, learning_rate=0.1, random_state=random_state)
#         }
#         spread_best = {
#             'xgb': XGBRegressor(n_estimators=300, max_depth=10, learning_rate=0.1, random_state=random_state+1),
#             'rf': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=random_state+1)
#         }

#     # Build stacking classifier pipeline
#     estimators_cls = [
#         ('xgb', cls_best['xgb']),
#         ('rf', cls_best['rf'])
#     ]
#     final_estimator_cls = LogisticRegression(max_iter=200, random_state=random_state)
#     stacking_clf = StackingClassifier(estimators=estimators_cls, final_estimator=final_estimator_cls, passthrough=False, n_jobs=-1)
#     clf_pipeline = make_pipeline(StandardScaler(), stacking_clf)
#     clf_pipeline.fit(X_train, y_cls_train)

#     # Build stacking regressor for total points
#     estimators_reg = [
#         ('xgb', reg_best['xgb']),
#         ('rf', reg_best['rf']),
#         ('gbr', reg_best['gbr'])
#     ]
#     final_estimator_reg = Ridge(alpha=1.0, random_state=random_state) if hasattr(Ridge, 'random_state') else Ridge(alpha=1.0)
#     stacking_reg = StackingRegressor(estimators=estimators_reg, final_estimator=final_estimator_reg, passthrough=False, n_jobs=-1)
#     reg_pipeline = make_pipeline(StandardScaler(), stacking_reg)
#     reg_pipeline.fit(X_train, y_reg_train)

#     # Build stacking regressor for spread
#     estimators_spread = [
#         ('xgb', spread_best['xgb']),
#         ('rf', spread_best['rf'])
#     ]
#     final_estimator_spread = Ridge(alpha=1.0)
#     stacking_spread = StackingRegressor(estimators=estimators_spread, final_estimator=final_estimator_spread, passthrough=False, n_jobs=-1)
#     reg_spread_pipeline = make_pipeline(StandardScaler(), stacking_spread)
#     reg_spread_pipeline.fit(X_train, y_spread_train)

#     # Evaluate on holdout
#     cls_preds = clf_pipeline.predict(X_test)
#     acc = accuracy_score(y_cls_test, cls_preds)
#     mae = mean_absolute_error(y_reg_test, reg_pipeline.predict(X_test))
#     spread_mae = mean_absolute_error(y_spread_test, reg_spread_pipeline.predict(X_test))
#     print(f"Ensemble Winner accuracy (holdout): {acc:.2%}. Total points MAE: {mae:.2f}. Spread MAE: {spread_mae:.2f}")

#     # Optionally save models
#     if save_models:
#         os.makedirs(model_dir, exist_ok=True)
#         joblib.dump(clf_pipeline, os.path.join(model_dir, 'clf_pipeline.joblib'))
#         joblib.dump(reg_pipeline, os.path.join(model_dir, 'reg_pipeline.joblib'))
#         joblib.dump(reg_spread_pipeline, os.path.join(model_dir, 'reg_spread_pipeline.joblib'))
#         print(f"Saved models to {model_dir}")

#     return clf_pipeline, reg_pipeline, reg_spread_pipeline, feature_cols

# def get_last_stats(team, features, is_home):
#     games = features[(features['home_team'] == team) if is_home else (features['away_team'] == team)]
#     if games.empty:
#         return [0]*9
#     last = games.iloc[-1]
#     if is_home:
#         return [
#             last['home_points_avg'],
#             last['home_yards_avg'],
#             last['home_tov_avg'],
#             last['home_rest'],
#             last['home_explosive_rate'],
#             last['home_rz_eff'],
#             last['home_3rd_down'],
#             last['home_sacks'],
#             last['home_def_turnover_margin']
#         ]
#     else:
#         return [
#             last['away_points_avg'],
#             last['away_yards_avg'],
#             last['away_tov_avg'],
#             last['away_rest'],
#             last['away_explosive_rate'],
#             last['away_rz_eff'],
#             last['away_3rd_down'],
#             last['away_sacks'],
#             last['away_def_turnover_margin']
#         ]

# def predict_upcoming_games(games, features, clf, reg, reg_spread, feature_cols):
#     future_games = games[games['home_score'].isna() | games['away_score'].isna()]
#     if future_games.empty:
#         print("No unplayed games found in the schedule.")
#         return

#     # Find the next week with scheduled (unplayed) games
#     if "week" in future_games.columns:
#         next_week = future_games["week"].min()
#         filtered_games = future_games[future_games["week"] == next_week].copy()
#         print(f"\n--- Predictions for Upcoming Games: Next Scheduled Week (week {next_week}) ---")
#     else:
#         filtered_games = future_games.copy()
#         print("\n--- Predictions for All Unplayed Games (no week info found) ---")

#     for _, row in filtered_games.iterrows():
#         home_team = row['home_team']
#         away_team = row['away_team']
#         spread = row.get('spread_line', 0.0)
#         over_under = row.get('total_line', 44.0)
#         home_stats = get_last_stats(home_team, features, True)
#         away_stats = get_last_stats(away_team, features, False)
#         (home_pts, home_yds, home_tov, home_rest, home_explosive, home_rz, home_3rd, home_sacks, home_def_to) = home_stats
#         (away_pts, away_yds, away_tov, away_rest, away_explosive, away_rz, away_3rd, away_sacks, away_def_to) = away_stats
#         fav = int(spread < 0)
#         input_df = pd.DataFrame([{
#             'home_points_avg': home_pts,
#             'away_points_avg': away_pts,
#             'home_yards_avg': home_yds,
#             'away_yards_avg': away_yds,
#             'home_tov_avg': home_tov,
#             'away_tov_avg': away_tov,
#             'home_rest': home_rest,
#             'away_rest': away_rest,
#             'home_is_favorite': fav,
#             'spread': abs(spread),
#             'over_under': over_under,
#             'home_explosive_rate': home_explosive,
#             'away_explosive_rate': away_explosive,
#             'home_rz_eff': home_rz,
#             'away_rz_eff': away_rz,
#             'home_3rd_down': home_3rd,
#             'away_3rd_down': away_3rd,
#             'home_sacks': home_sacks,
#             'away_sacks': away_sacks,
#             'home_def_turnover_margin': home_def_to,
#             'away_def_turnover_margin': away_def_to
#         }])
#         win_prob = clf.predict_proba(input_df)[0,1]
#         total_pts = reg.predict(input_df)[0]
#         spread_pred = reg_spread.predict(input_df)[0]
#         if spread_pred > 0:
#             spread_str = f"Home team ({home_team}) -{abs(spread_pred):.1f}"
#         else:
#             spread_str = f"Away team ({away_team}) -{abs(spread_pred):.1f}"
#         week_str = f"Week {row['week']}" if "week" in row else ""
#         print(f"{week_str} | {home_team} vs {away_team} | Prob {home_team} Wins: {win_prob:.1%} | Pred Total: {total_pts:.1f} | Pred Spread: {spread_str}")
#         print("-----------------------------------------------")

# def main():
#     years = [2023,2024,2025]  # Change as needed
#     print("Loading NFL data for years:", years)
#     games = nfl.import_schedules(years)
#     pbp = nfl.import_pbp_data(years)
#     print(f"Loaded {len(games)} games, {len(pbp)} play-by-play rows.")
#     features = build_features(games, pbp)

#     # Train with tuning enabled (set tune=False to skip tuning)
#     clf, reg, reg_spread, feature_cols = train_models(features, tune=True, n_iter=24, cv_splits=5, save_models=False, model_dir='models')

#     # Predict upcoming games
#     predict_upcoming_games(games, features, clf, reg, reg_spread, feature_cols)

# if __name__ == "__main__":
#     main()

# import os
# import pandas as pd
# import numpy as np
# import nfl_data_py as nfl
# from datetime import datetime
# from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
# from xgboost import XGBClassifier, XGBRegressor
# from sklearn.metrics import accuracy_score, mean_absolute_error

# # Ensemble & tuning imports
# from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingRegressor, StackingClassifier, StackingRegressor, HistGradientBoostingRegressor
# from sklearn.linear_model import LogisticRegression, Ridge
# from sklearn.preprocessing import StandardScaler
# from sklearn.pipeline import make_pipeline, Pipeline
# import joblib

# # Explainability imports
# import matplotlib.pyplot as plt
# from sklearn.inspection import permutation_importance, PartialDependenceDisplay
# from sklearn.utils import check_random_state

# # Optional shap import (graceful fallback)
# try:
#     import shap
#     _HAS_SHAP = True
# except Exception:
#     _HAS_SHAP = False

# import scipy.stats as stats

# # ------------------------
# # Data / feature engineering
# # ------------------------

# def add_vegas_lines(games):
#     if 'spread_line' not in games.columns:
#         games['spread_line'] = 0.0
#     if 'total_line' not in games.columns:
#         games['total_line'] = 44.0
#     games['spread_line'] = games['spread_line'].fillna(0.0)
#     games['total_line'] = games['total_line'].fillna(44.0)
#     return games

# def get_date_col(games):
#     for candidate in ['gametime', 'start_time', 'datetime', 'game_date', 'start_date', 'date']:
#         if candidate in games.columns:
#             return candidate
#     raise Exception(f"No valid date column found in games columns: {games.columns.tolist()}")

# def compute_team_game_stats(pbp):
#     pbp = pbp[~pbp['season_type'].isin(['PRE'])]
#     pbp = pbp[pbp['posteam'].notnull() & pbp['defteam'].notnull()]
#     pbp['explosive_play'] = ((pbp.get('passing_yards', 0) >= 20) | (pbp.get('rushing_yards', 0) >= 20)).astype(int)
#     pbp['is_red_zone'] = pbp.get('yardline_100', 999) <= 20
#     pbp['is_third_down'] = pbp.get('down', 0) == 3

#     # Third down conversions: made if first down gained or TD
#     pbp['third_down_converted'] = (
#         (pbp['is_third_down']) &
#         ((pbp.get('first_down', 0) == 1) | (pbp.get('touchdown', 0) == 1))
#     ).astype(int)

#     agg_stats = pbp.groupby(['game_id', 'posteam']).agg(
#         points_scored=('touchdown', 'sum'),
#         pass_yards=('passing_yards', 'sum'),
#         rush_yards=('rushing_yards', 'sum'),
#         turnovers=('interception', 'sum'),
#         fumbles=('fumble_lost', 'sum'),
#         explosive_plays=('explosive_play', 'sum'),
#         total_plays=('play_id', 'count'),
#         red_zone_plays=('is_red_zone', 'sum'),
#         red_zone_tds=('touchdown', lambda x: x[pbp.loc[x.index, 'yardline_100'] <= 20].sum() if len(x) > 0 else 0),
#         third_down_plays=('is_third_down', 'sum'),
#         third_down_conversions=('third_down_converted', 'sum'),
#         sacks=('sack', 'sum')
#     ).reset_index()
#     agg_stats['total_yards'] = agg_stats['pass_yards'] + agg_stats['rush_yards']
#     agg_stats['turnovers'] = agg_stats['turnovers'] + agg_stats['fumbles']
#     # Rates
#     agg_stats['explosive_play_rate'] = agg_stats['explosive_plays'] / agg_stats['total_plays'].replace(0, np.nan)
#     agg_stats['red_zone_eff'] = agg_stats['red_zone_tds'] / agg_stats['red_zone_plays'].replace(0, np.nan)
#     agg_stats['third_down_conv_rate'] = agg_stats['third_down_conversions'] / agg_stats['third_down_plays'].replace(0, np.nan)
#     # Defensive stats for margin
#     defense = agg_stats.rename(columns={
#         "posteam": "defteam",
#         "turnovers": "def_turnovers",
#         "sacks": "def_sacks"
#     })[['game_id','defteam','def_turnovers','def_sacks']]
#     return agg_stats, defense

# def get_rest_days(games, date_col):
#     rest = []
#     for team in set(games['home_team']).union(set(games['away_team'])):
#         tgames = games[(games['home_team'] == team) | (games['away_team'] == team)].sort_values(['season', 'week'])
#         last_date = None
#         for idx, row in tgames.iterrows():
#             gdate = pd.to_datetime(row[date_col])
#             if last_date is None:
#                 rest.append((row['game_id'], team, 7))
#             else:
#                 diff = (gdate - last_date).days
#                 rest.append((row['game_id'], team, diff if diff > 0 else 7))
#             last_date = gdate
#     rest_df = pd.DataFrame(rest, columns=['game_id', 'team', 'rest_days'])
#     return rest_df

# def build_features(games, pbp):
#     games = add_vegas_lines(games)
#     date_col = get_date_col(games)
#     agg_stats, defense = compute_team_game_stats(pbp)

#     # Prepare long-form DataFrame
#     home = games[['game_id','season','week',date_col,'home_team','away_team','home_score','away_score','spread_line','total_line']].rename(
#         columns={date_col:'date','home_team':'team','away_team':'opp','home_score':'points_scored','away_score':'opp_points'}
#     )
#     home['is_home'] = 1
#     away = games[['game_id','season','week',date_col,'away_team','home_team','away_score','home_score','spread_line','total_line']].rename(
#         columns={date_col:'date','away_team':'team','home_team':'opp','away_score':'points_scored','home_score':'opp_points'}
#     )
#     away['is_home'] = 0
#     long_games = pd.concat([home, away], ignore_index=True)

#     # Merge in aggregated per-team stats
#     long_games['team'] = long_games['team'].astype(str)
#     agg_stats['posteam'] = agg_stats['posteam'].astype(str)
#     long_games['game_id'] = long_games['game_id'].astype(str)
#     agg_stats['game_id'] = agg_stats['game_id'].astype(str)
#     long_games = long_games.merge(agg_stats, how='left', left_on=['game_id','team'], right_on=['game_id','posteam'])

#     # Defensive stats for margin (opponent)
#     defense['defteam'] = defense['defteam'].astype(str)
#     long_games = long_games.merge(defense, how='left', left_on=['game_id','opp'], right_on=['game_id','defteam'], suffixes=('', '_opp'))

#     # Defensive fillna for all engineered columns
#     for col in ['points_scored', 'total_yards', 'turnovers', 'explosive_play_rate', 'red_zone_eff', 'third_down_conv_rate', 'sacks', 'def_sacks', 'def_turnovers']:
#         if col not in long_games.columns:
#             long_games[col] = 0
#         long_games[col] = long_games[col].fillna(0)

#     long_games = long_games.sort_values(['team','season','week'])
#     # Rolling stats (advanced) - using window=3 like original code (name kept _rolling5 for backwards compatibility)
#     for stat in ['points_scored','total_yards','turnovers','explosive_play_rate','red_zone_eff','third_down_conv_rate','sacks','def_sacks','def_turnovers']:
#         long_games[f'{stat}_rolling5'] = long_games.groupby('team')[stat].rolling(3, min_periods=1).mean().reset_index(0,drop=True)

#     rest_df = get_rest_days(games, date_col)
#     long_games = long_games.merge(rest_df, how='left', on=['game_id','team'])

#     features = []
#     for _, row in games.iterrows():
#         def get_last(team, week, season, is_home):
#             prev = long_games[
#                 (long_games['team'] == team) &
#                 ((long_games['season'] < season) | ((long_games['season'] == season) & (long_games['week'] < week)))
#             ]
#             prev = prev[prev['is_home'] == is_home] if not prev.empty else prev
#             return prev.iloc[-1] if not prev.empty else {}
#         home_last = get_last(row['home_team'], row['week'], row['season'], 1)
#         away_last = get_last(row['away_team'], row['week'], row['season'], 0)
#         features.append({
#             'game_id': row['game_id'],
#             'season': row['season'],
#             'week': row['week'],
#             'home_team': row['home_team'],
#             'away_team': row['away_team'],
#             'home_points_avg': home_last.get('points_scored_rolling5', 0),
#             'away_points_avg': away_last.get('points_scored_rolling5', 0),
#             'home_yards_avg': home_last.get('total_yards_rolling5', 0),
#             'away_yards_avg': away_last.get('total_yards_rolling5', 0),
#             'home_tov_avg': home_last.get('turnovers_rolling5', 0),
#             'away_tov_avg': away_last.get('turnovers_rolling5', 0),
#             'home_rest': home_last.get('rest_days', 7),
#             'away_rest': away_last.get('rest_days', 7),
#             'home_is_favorite': int(row['spread_line'] < 0),
#             'spread': abs(row['spread_line']),
#             'over_under': row['total_line'],
#             'home_explosive_rate': home_last.get('explosive_play_rate_rolling5', 0),
#             'away_explosive_rate': away_last.get('explosive_play_rate_rolling5', 0),
#             'home_rz_eff': home_last.get('red_zone_eff_rolling5', 0),
#             'away_rz_eff': away_last.get('red_zone_eff_rolling5', 0),
#             'home_3rd_down': home_last.get('third_down_conv_rate_rolling5', 0),
#             'away_3rd_down': away_last.get('third_down_conv_rate_rolling5', 0),
#             'home_sacks': home_last.get('sacks_rolling5', 0),
#             'away_sacks': away_last.get('sacks_rolling5', 0),
#             'home_def_turnover_margin': home_last.get('def_turnovers_rolling5', 0) - home_last.get('turnovers_rolling5', 0),
#             'away_def_turnover_margin': away_last.get('def_turnovers_rolling5', 0) - away_last.get('turnovers_rolling5', 0),
#             'home_score': row.get('home_score', np.nan),
#             'away_score': row.get('away_score', np.nan)
#         })
#     df = pd.DataFrame(features)
#     df = df.dropna(subset=['home_score','away_score'])
#     df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
#     df['total_points'] = df['home_score'] + df['away_score']
#     df['spread_actual'] = df['home_score'] - df['away_score']
#     return df

# # ------------------------
# # Train / tuning / stacking
# # ------------------------

# def _time_train_test_split(df, X, y_cols, test_size=0.15):
#     """
#     Time-based train/test split: sorts by season/week and splits last fraction as test.
#     y_cols: list of y series corresponding to X columns order (list of pandas Series)
#     Returns: X_train, X_test, [y_train...], [y_test...]
#     """
#     df_sorted = df.sort_values(['season', 'week']).reset_index(drop=True)
#     n = len(df_sorted)
#     split_at = int(n * (1 - test_size))
#     train_idx = df_sorted.index[:split_at]
#     test_idx = df_sorted.index[split_at:]
#     X_train = X.loc[train_idx].reset_index(drop=True)
#     X_test = X.loc[test_idx].reset_index(drop=True)
#     y_trains = []
#     y_tests = []
#     for y in y_cols:
#         y_trains.append(y.loc[train_idx].reset_index(drop=True))
#         y_tests.append(y.loc[test_idx].reset_index(drop=True))
#     return X_train, X_test, y_trains, y_tests

# def tune_classifiers(X, y, n_iter=20, cv_splits=5, random_state=42):
#     """Tune XGBClassifier and RandomForestClassifier using RandomizedSearchCV (time-series CV)."""
#     tscv = TimeSeriesSplit(n_splits=cv_splits)
#     results = {}

#     xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=random_state, verbosity=0)
#     xgb_param_dist = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [3, 6, 8, 10],
#         'learning_rate': [0.01, 0.05, 0.08, 0.1],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0]
#     }
#     rs_xgb = RandomizedSearchCV(xgb, xgb_param_dist, n_iter=n_iter, cv=tscv, scoring='accuracy', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_xgb.fit(X, y)
#     results['xgb'] = rs_xgb.best_estimator_

#     rf = RandomForestClassifier(random_state=random_state)
#     rf_param_dist = {
#         'n_estimators': [100, 200, 400],
#         'max_depth': [None, 6, 10, 15],
#         'max_features': ['auto', 'sqrt', 'log2']
#     }
#     rs_rf = RandomizedSearchCV(rf, rf_param_dist, n_iter=n_iter, cv=tscv, scoring='accuracy', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_rf.fit(X, y)
#     results['rf'] = rs_rf.best_estimator_

#     return results

# def tune_regressors(X, y, n_iter=20, cv_splits=5, random_state=42):
#     """Tune XGBRegressor, RandomForestRegressor, GradientBoostingRegressor using RandomizedSearchCV (time-series CV)."""
#     tscv = TimeSeriesSplit(n_splits=cv_splits)
#     results = {}

#     xgb = XGBRegressor(random_state=random_state, verbosity=0)
#     xgb_param_dist = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [3, 6, 8, 10],
#         'learning_rate': [0.01, 0.05, 0.08, 0.1],
#         'subsample': [0.6, 0.8, 1.0],
#         'colsample_bytree': [0.6, 0.8, 1.0]
#     }
#     rs_xgb = RandomizedSearchCV(xgb, xgb_param_dist, n_iter=n_iter, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_xgb.fit(X, y)
#     results['xgb'] = rs_xgb.best_estimator_

#     rf = RandomForestRegressor(random_state=random_state)
#     rf_param_dist = {
#         'n_estimators': [100, 200, 400],
#         'max_depth': [None, 6, 10, 15],
#         'max_features': ['auto', 'sqrt', 'log2']
#     }
#     rs_rf = RandomizedSearchCV(rf, rf_param_dist, n_iter=n_iter, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_rf.fit(X, y)
#     results['rf'] = rs_rf.best_estimator_

#     gbr = GradientBoostingRegressor(random_state=random_state)
#     gbr_param_dist = {
#         'n_estimators': [100, 200, 300],
#         'max_depth': [3, 5, 6, 8],
#         'learning_rate': [0.01, 0.05, 0.08, 0.1],
#         'subsample': [0.6, 0.8, 1.0]
#     }
#     rs_gbr = RandomizedSearchCV(gbr, gbr_param_dist, n_iter=n_iter, cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=random_state, verbose=0)
#     rs_gbr.fit(X, y)
#     results['gbr'] = rs_gbr.best_estimator_

#     return results

# def train_models(df, tune=True, n_iter=20, cv_splits=5, save_models=False, model_dir='models', random_state=42, test_size=0.15, feature_cols=None):
#     """
#     Train ensembles with optional hyperparameter tuning.
#     - feature_cols: optional list of columns to use (if None use default full list)
#     Returns:
#       clf_pipeline, home_reg_pipeline, away_reg_pipeline, used_feature_cols, holdout_tuple
#     """
#     default_feature_cols = [
#         'home_points_avg','away_points_avg','home_yards_avg','away_yards_avg',
#         'home_tov_avg','away_tov_avg','home_rest','away_rest','home_is_favorite',
#         'spread','over_under',
#         # Advanced
#         'home_explosive_rate', 'away_explosive_rate',
#         'home_rz_eff', 'away_rz_eff',
#         'home_3rd_down', 'away_3rd_down',
#         'home_sacks', 'away_sacks',
#         'home_def_turnover_margin', 'away_def_turnover_margin'
#     ]

#     if feature_cols is None:
#         feature_cols = default_feature_cols
#     else:
#         missing = [f for f in feature_cols if f not in df.columns]
#         if missing:
#             raise ValueError(f"Requested feature_cols contain names not in DataFrame: {missing}")

#     X = df[feature_cols].fillna(0).reset_index(drop=True)
#     y_cls = df['home_win'].reset_index(drop=True)
#     y_home = df['home_score'].reset_index(drop=True)
#     y_away = df['away_score'].reset_index(drop=True)

#     # Time-based train/test split
#     X_train, X_test, [y_cls_train, y_home_train, y_away_train], [y_cls_test, y_home_test, y_away_test] = _time_train_test_split(
#         df.reset_index(drop=True),
#         X,
#         [y_cls, y_home, y_away],
#         test_size=test_size
#     )

#     # Optionally tune base estimators for classifier only (regressors use MAE-optimized HGB below by default)
#     if tune:
#         print("Tuning classifier base estimators (this may take a while)...")
#         cls_best = tune_classifiers(X_train, y_cls_train, n_iter=n_iter, cv_splits=cv_splits, random_state=random_state)
#     else:
#         cls_best = {
#             'xgb': XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=200, max_depth=8, learning_rate=0.08, random_state=random_state),
#             'rf': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=random_state)
#         }

#     # Build stacking classifier pipeline
#     estimators_cls = [
#         ('xgb', cls_best['xgb']),
#         ('rf', cls_best['rf'])
#     ]
#     final_estimator_cls = LogisticRegression(max_iter=200, random_state=random_state)
#     stacking_clf = StackingClassifier(estimators=estimators_cls, final_estimator=final_estimator_cls, passthrough=False, n_jobs=-1)
#     clf_pipeline = make_pipeline(StandardScaler(), stacking_clf)
#     clf_pipeline.fit(X_train, y_cls_train)

#     # --- NEW: Train two regressors (home_score, away_score) optimized for MAE ---
#     home_reg = HistGradientBoostingRegressor(loss='absolute_error', max_iter=1000, learning_rate=0.05, max_depth=8, random_state=random_state)
#     away_reg = HistGradientBoostingRegressor(loss='absolute_error', max_iter=1000, learning_rate=0.05, max_depth=8, random_state=random_state+1)

#     home_reg_pipeline = make_pipeline(StandardScaler(), home_reg)
#     away_reg_pipeline = make_pipeline(StandardScaler(), away_reg)

#     print("Training home score regressor (MAE objective)...")
#     home_reg_pipeline.fit(X_train, y_home_train)
#     print("Training away score regressor (MAE objective)...")
#     away_reg_pipeline.fit(X_train, y_away_train)

#     # Evaluate on holdout: derive spread and total from home/away preds
#     home_pred_test = home_reg_pipeline.predict(X_test)
#     away_pred_test = away_reg_pipeline.predict(X_test)
#     spread_pred = home_pred_test - away_pred_test
#     total_pred = home_pred_test + away_pred_test

#     spread_actual = y_home_test - y_away_test
#     total_actual = y_home_test + y_away_test

#     spread_mae = mean_absolute_error(spread_actual, spread_pred)
#     total_mae = mean_absolute_error(total_actual, total_pred)
#     cls_preds = clf_pipeline.predict(X_test)
#     cls_acc = accuracy_score(y_cls_test, cls_preds)

#     print(f"[TRAIN] Using {len(feature_cols)} features. Classifier acc (holdout): {cls_acc:.3f}. Spread MAE: {spread_mae:.3f}. Total MAE: {total_mae:.3f}")

#     # Optionally save models
#     if save_models:
#         os.makedirs(model_dir, exist_ok=True)
#         joblib.dump(clf_pipeline, os.path.join(model_dir, 'clf_pipeline.joblib'))
#         joblib.dump(home_reg_pipeline, os.path.join(model_dir, 'home_reg_pipeline.joblib'))
#         joblib.dump(away_reg_pipeline, os.path.join(model_dir, 'away_reg_pipeline.joblib'))
#         print(f"Saved models to {model_dir}")

#     holdout = (X_train, X_test, y_cls_train, y_cls_test, y_home_train, y_home_test, y_away_train, y_away_test)
#     return clf_pipeline, home_reg_pipeline, away_reg_pipeline, feature_cols, holdout

# # ------------------------
# # Explainability / feature selection
# # ------------------------

# def _find_stacking_in_pipeline(pipeline):
#     """Return the first StackingClassifier/Regressor found in a sklearn Pipeline (or None)."""
#     if isinstance(pipeline, Pipeline):
#         for name, obj in pipeline.named_steps.items():
#             if isinstance(obj, (StackingClassifier, StackingRegressor)):
#                 return name, obj
#         return None, None
#     else:
#         if isinstance(pipeline, (StackingClassifier, StackingRegressor)):
#             return None, pipeline
#         return None, None

# def _get_preprocessor_before_stacking(pipeline):
#     """
#     Return a sklearn Pipeline consisting of steps before the stacking estimator.
#     If no stacking in pipeline, return pipeline itself if it's a Pipeline else None.
#     """
#     if not isinstance(pipeline, Pipeline):
#         return None
#     steps = pipeline.steps
#     for i, (name, obj) in enumerate(steps):
#         if isinstance(obj, (StackingClassifier, StackingRegressor)):
#             if i == 0:
#                 return None
#             return Pipeline(steps[:i])
#     return Pipeline(steps[:-1]) if len(steps) > 1 else None

# def plot_tree_importances(pipeline_or_model, feature_names, top_n=20, savepath=None, title=None):
#     found_name, stacking = _find_stacking_in_pipeline(pipeline_or_model)
#     outputs = []
#     if stacking is None:
#         model = pipeline_or_model
#         if hasattr(model, 'feature_importances_'):
#             imp = model.feature_importances_
#             df = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False).head(top_n)
#             fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
#             ax.barh(df['feature'][::-1], df['importance'][::-1])
#             ax.set_title(title or "Feature importances")
#             plt.tight_layout()
#             if savepath:
#                 fig.savefig(savepath)
#             plt.close(fig)
#             return [df]
#         else:
#             print("No stacking object found and model has no feature_importances_.")
#             return []
#     else:
#         fitted_estimators = getattr(stacking, 'estimators_', None)
#         if fitted_estimators is None:
#             print("Stacking estimator found but it does not have fitted estimators_. Ensure pipeline was fit.")
#             return []
#         for i, est in enumerate(fitted_estimators):
#             if est is None:
#                 continue
#             name = getattr(est, '__class__', type(est)).__name__
#             if hasattr(est, 'feature_importances_'):
#                 imp = est.feature_importances_
#                 df = pd.DataFrame({'feature': feature_names, 'importance': imp}).sort_values('importance', ascending=False).head(top_n)
#                 fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
#                 ax.barh(df['feature'][::-1], df['importance'][::-1])
#                 ax.set_title(title or f"Base estimator {i} ({name}) feature importances")
#                 plt.tight_layout()
#                 if savepath:
#                     p = savepath.replace('.png', f'_{i}_{name}.png')
#                     fig.savefig(p)
#                 plt.close(fig)
#                 outputs.append((name, df))
#         return outputs

# def permutation_importance_report(pipeline, X, y, scoring=None, n_repeats=30, random_state=42, n_jobs=-1, savepath=None):
#     rng = check_random_state(random_state)
#     r = permutation_importance(pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=rng, n_jobs=n_jobs)
#     imp_means = r.importances_mean
#     imp_std = r.importances_std
#     df = pd.DataFrame({
#         'feature': X.columns if hasattr(X, 'columns') else [f'f{i}' for i in range(X.shape[1])],
#         'importance_mean': imp_means,
#         'importance_std': imp_std
#     }).sort_values('importance_mean', ascending=False).reset_index(drop=True)

#     # Plot bar chart
#     fig, ax = plt.subplots(figsize=(8, max(4, 0.35 * len(df))))
#     ax.barh(df['feature'][::-1], df['importance_mean'][::-1], xerr=df['importance_std'][::-1])
#     ax.set_title(f'Permutation importance (scoring={scoring})')
#     plt.tight_layout()
#     if savepath:
#         fig.savefig(savepath)
#     plt.close(fig)
#     return df

# def shap_explain_pipeline(pipeline, X, feature_names=None, outdir='explain_outputs', sample_size=200, base_estimator_only=True):
#     os.makedirs(outdir, exist_ok=True)
#     results = {}

#     if not _HAS_SHAP:
#         results['error'] = "shap not installed. Install with `pip install shap` to enable SHAP explanations."
#         print(results['error'])
#         return results

#     stacking_name, stacking = _find_stacking_in_pipeline(pipeline)
#     preprocessor = _get_preprocessor_before_stacking(pipeline)

#     X_df = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X)
#     sample = X_df.sample(min(sample_size, len(X_df)), random_state=1)

#     try:
#         if stacking is not None:
#             fitted = getattr(stacking, 'estimators_', None)
#             if fitted:
#                 explained = 0
#                 for i, base in enumerate(fitted):
#                     if base is None:
#                         continue
#                     model_name = getattr(base, '__class__', type(base)).__name__
#                     is_tree_like = hasattr(base, 'feature_importances_') or 'XGB' in model_name or 'LGBM' in model_name
#                     if not base_estimator_only or is_tree_like:
#                         if preprocessor is not None:
#                             X_trans = preprocessor.transform(sample)
#                             background = preprocessor.transform(X_df.sample(min(100, len(X_df)), random_state=0))
#                         else:
#                             X_trans = sample.values
#                             background = X_df.sample(min(100, len(X_df)), random_state=0).values
#                         try:
#                             explainer = shap.Explainer(base, background)
#                             shap_vals = explainer(X_trans)
#                             plt.figure(figsize=(8,6))
#                             shap.summary_plot(shap_vals, features=sample if preprocessor is None else pd.DataFrame(X_trans, columns=feature_names), feature_names=feature_names, show=False)
#                             fname = os.path.join(outdir, f"shap_summary_base_{i}_{model_name}.png")
#                             plt.savefig(fname, bbox_inches='tight')
#                             plt.close()
#                             results[f'shap_base_{i}'] = {'model': model_name, 'file': fname}
#                             explained += 1
#                         except Exception as e:
#                             results[f'shap_base_{i}_error'] = str(e)
#                     if base_estimator_only and explained >= 1:
#                         break
#                 if explained == 0:
#                     results['shap_note'] = "No suitable tree-like base estimators were explained."
#             else:
#                 results['shap_note'] = "Stacking found but no fitted estimators_."
#         if ('shap_base_0' not in results) and ('shap_base_0_error' not in results):
#             final_est = None
#             if isinstance(pipeline, Pipeline):
#                 final_est = list(pipeline.named_steps.values())[-1]
#             else:
#                 final_est = pipeline
#             final_name = getattr(final_est, '__class__', type(final_est)).__name__
#             preproc = _get_preprocessor_before_stacking(pipeline) if stacking is None else _get_preprocessor_before_stacking(pipeline)
#             if preproc is not None:
#                 X_trans = preproc.transform(sample)
#                 background = preproc.transform(X_df.sample(min(100, len(X_df)), random_state=0))
#             else:
#                 X_trans = sample.values
#                 background = X_df.sample(min(100, len(X_df)), random_state=0).values
#             try:
#                 explainer = shap.Explainer(final_est, background)
#                 shap_vals = explainer(X_trans)
#                 plt.figure(figsize=(8,6))
#                 shap.summary_plot(shap_vals, features=sample if preproc is None else pd.DataFrame(X_trans, columns=feature_names), feature_names=feature_names, show=False)
#                 fname = os.path.join(outdir, f"shap_summary_final_{final_name}.png")
#                 plt.savefig(fname, bbox_inches='tight')
#                 plt.close()
#                 results['shap_final'] = {'model': final_name, 'file': fname}
#             except Exception as e:
#                 results['shap_final_error'] = str(e)
#     except Exception as e:
#         results['shap_error'] = str(e)

#     return results

# def partial_dependence_plots(pipeline, X, features, outdir=None, grid_resolution=50):
#     os.makedirs(outdir, exist_ok=True) if outdir else None
#     fig, ax = plt.subplots(figsize=(8, 4 * len(features)))
#     display = PartialDependenceDisplay.from_estimator(pipeline, X, features=features, grid_resolution=grid_resolution, ax=ax)
#     plt.tight_layout()
#     if outdir:
#         plt.savefig(os.path.join(outdir, "partial_dependence.png"), bbox_inches='tight')
#     plt.close(fig)
#     return display

# def explain_model_workflow(pipeline, X_train, X_test, y_test, feature_names, outdir='explain_outputs', do_shap=True, do_permutation=True, do_tree=True, top_k=10):
#     os.makedirs(outdir, exist_ok=True)
#     results = {}

#     if do_tree:
#         try:
#             tree_res = plot_tree_importances(pipeline, feature_names, top_n=20, savepath=os.path.join(outdir, 'tree_importances.png'))
#             results['tree'] = tree_res
#         except Exception as e:
#             results['tree_error'] = str(e)

#     if do_permutation:
#         try:
#             scorer = 'accuracy' if len(np.unique(y_test)) == 2 else 'neg_mean_absolute_error'
#             perm_df = permutation_importance_report(pipeline, X_test, y_test, scoring=scorer, n_repeats=30, savepath=os.path.join(outdir, 'permutation_importance.png'))
#             results['permutation'] = perm_df
#         except Exception as e:
#             results['permutation_error'] = str(e)

#     if do_shap:
#         try:
#             shap_res = shap_explain_pipeline(pipeline, pd.concat([X_train, X_test]).reset_index(drop=True), feature_names=feature_names, outdir=outdir, sample_size=200)
#             results['shap'] = shap_res
#         except Exception as e:
#             results['shap_error'] = str(e)

#     # PDP for top permutation features if available
#     try:
#         if 'permutation' in results and isinstance(results['permutation'], pd.DataFrame):
#             top_features = results['permutation']['feature'].tolist()[:top_k]
#             partial_dependence_plots(pipeline, pd.concat([X_train, X_test]).reset_index(drop=True), top_features, outdir=outdir)
#             results['pdp_features'] = top_features
#     except Exception as e:
#         results['pdp_error'] = str(e)

#     return results

# def select_top_k_by_permutation(pipeline, X, y, k=10, n_repeats=30, scoring=None, random_state=42):
#     """
#     Compute permutation importance on (pipeline, X, y) and return a list of top-k features
#     and the full importance DataFrame (sorted). Use on holdout to avoid leakage.
#     """
#     r = permutation_importance(pipeline, X, y, scoring=scoring, n_repeats=n_repeats, random_state=random_state, n_jobs=-1)
#     df = pd.DataFrame({
#         'feature': X.columns,
#         'importance_mean': r.importances_mean,
#         'importance_std': r.importances_std
#     }).sort_values('importance_mean', ascending=False).reset_index(drop=True)
#     top_k = df['feature'].tolist()[:k]
#     return top_k, df

# def select_top_k_by_shap(pipeline, X, k=10, sample_size=200):
#     """
#     Use SHAP to compute mean(|SHAP|) per feature and return top-k features.
#     - Requires shap to be installed.
#     - We sample up to sample_size rows to keep it reasonably fast.
#     """
#     if not _HAS_SHAP:
#         raise ImportError("shap not installed. Install with `pip install shap` to use select_top_k_by_shap.")

#     X_df = X.copy() if hasattr(X, 'copy') else pd.DataFrame(X)
#     sample = X_df.sample(min(sample_size, len(X_df)), random_state=1)

#     stacking_name, stacking = _find_stacking_in_pipeline(pipeline)
#     preproc = _get_preprocessor_before_stacking(pipeline)

#     if preproc is not None:
#         X_trans = preproc.transform(sample)
#         background = preproc.transform(X_df.sample(min(100, len(X_df)), random_state=0))
#         feature_names = X_df.columns.tolist()
#         background_df = pd.DataFrame(background, columns=feature_names)
#         X_trans_df = pd.DataFrame(X_trans, columns=feature_names)
#     else:
#         X_trans_df = sample
#         background_df = X_df.sample(min(100, len(X_df)), random_state=0)

#     # Try tree-like base estimators first
#     if stacking is not None:
#         fitted = getattr(stacking, 'estimators_', None)
#         if fitted:
#             for base in fitted:
#                 if base is None:
#                     continue
#                 name = getattr(base, '__class__', type(base)).__name__
#                 is_tree_like = hasattr(base, 'feature_importances_') or 'XGB' in name or 'LGBM' in name
#                 if is_tree_like:
#                     explainer = shap.Explainer(base, background_df)
#                     shap_vals = explainer(X_trans_df)
#                     mean_abs = np.mean(np.abs(shap_vals.values), axis=0)
#                     df = pd.DataFrame({'feature': X_df.columns, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
#                     top_k = df['feature'].tolist()[:k]
#                     return top_k, df
#     # fallback: final estimator
#     final_est = pipeline
#     if isinstance(pipeline, Pipeline):
#         final_est = list(pipeline.named_steps.values())[-1]
#     explainer = shap.Explainer(final_est, background_df)
#     shap_vals = explainer(X_trans_df)
#     mean_abs = np.mean(np.abs(shap_vals.values), axis=0)
#     df = pd.DataFrame({'feature': X_df.columns, 'mean_abs_shap': mean_abs}).sort_values('mean_abs_shap', ascending=False).reset_index(drop=True)
#     top_k = df['feature'].tolist()[:k]
#     return top_k, df

# # ------------------------
# # Prediction utilities & CLI
# # ------------------------

# def get_last_stats(team, features, is_home):
#     games = features[(features['home_team'] == team) if is_home else (features['away_team'] == team)]
#     if games.empty:
#         return [0]*9
#     last = games.iloc[-1]
#     if is_home:
#         return [
#             last['home_points_avg'],
#             last['home_yards_avg'],
#             last['home_tov_avg'],
#             last['home_rest'],
#             last['home_explosive_rate'],
#             last['home_rz_eff'],
#             last['home_3rd_down'],
#             last['home_sacks'],
#             last['home_def_turnover_margin']
#         ]
#     else:
#         return [
#             last['away_points_avg'],
#             last['away_yards_avg'],
#             last['away_tov_avg'],
#             last['away_rest'],
#             last['away_explosive_rate'],
#             last['away_rz_eff'],
#             last['away_3rd_down'],
#             last['away_sacks'],
#             last['away_def_turnover_margin']
#         ]

# def predict_upcoming_games(games, features, clf, home_reg, away_reg, feature_cols):
#     future_games = games[games['home_score'].isna() | games['away_score'].isna()]
#     if future_games.empty:
#         print("No unplayed games found in the schedule.")
#         return

#     # Find the next week with scheduled (unplayed) games
#     if "week" in future_games.columns:
#         next_week = future_games["week"].min()
#         filtered_games = future_games[future_games["week"] == next_week].copy()
#         print(f"\n--- Predictions for Upcoming Games: Next Scheduled Week (week {next_week}) ---")
#     else:
#         filtered_games = future_games.copy()
#         print("\n--- Predictions for All Unplayed Games (no week info found) ---")

#     for _, row in filtered_games.iterrows():
#         home_team = row['home_team']
#         away_team = row['away_team']
#         spread = row.get('spread_line', 0.0)
#         over_under = row.get('total_line', 44.0)
#         home_stats = get_last_stats(home_team, features, True)
#         away_stats = get_last_stats(away_team, features, False)
#         (home_pts, home_yds, home_tov, home_rest, home_explosive, home_rz, home_3rd, home_sacks, home_def_to) = home_stats
#         (away_pts, away_yds, away_tov, away_rest, away_explosive, away_rz, away_3rd, away_sacks, away_def_to) = away_stats
#         fav = int(spread < 0)
#         input_df = pd.DataFrame([{
#             'home_points_avg': home_pts,
#             'away_points_avg': away_pts,
#             'home_yards_avg': home_yds,
#             'away_yards_avg': away_yds,
#             'home_tov_avg': home_tov,
#             'away_tov_avg': away_tov,
#             'home_rest': home_rest,
#             'away_rest': away_rest,
#             'home_is_favorite': fav,
#             'spread': abs(spread),
#             'over_under': over_under,
#             'home_explosive_rate': home_explosive,
#             'away_explosive_rate': away_explosive,
#             'home_rz_eff': home_rz,
#             'away_rz_eff': away_rz,
#             'home_3rd_down': home_3rd,
#             'away_3rd_down': away_3rd,
#             'home_sacks': home_sacks,
#             'away_sacks': away_sacks,
#             'home_def_turnover_margin': home_def_to,
#             'away_def_turnover_margin': away_def_to
#         }])
#         # Ensure all expected feature cols exist and are ordered
#         for c in feature_cols:
#             if c not in input_df.columns:
#                 input_df[c] = 0
#         input_df = input_df[feature_cols]

#         win_prob = clf.predict_proba(input_df)[0,1]
#         home_score_pred = home_reg.predict(input_df)[0]
#         away_score_pred = away_reg.predict(input_df)[0]
#         spread_pred = home_score_pred - away_score_pred
#         total_pred = home_score_pred + away_score_pred

#         if spread_pred > 0:
#             spread_str = f"Home team ({home_team}) -{abs(spread_pred):.1f}"
#         else:
#             spread_str = f"Away team ({away_team}) -{abs(spread_pred):.1f}"
#         week_str = f"Week {row['week']}" if "week" in row else ""
#         print(f"{week_str} | {home_team} vs {away_team} | Prob {home_team} Wins: {win_prob:.1%} | Pred Home: {home_score_pred:.1f} | Pred Away: {away_score_pred:.1f} | Pred Total: {total_pred:.1f} | Pred Spread: {spread_str}")
#         print("-----------------------------------------------")

# # ------------------------
# # Main
# # ------------------------

# def main():
#     years = [2023,2024,2025]  # Change as needed
#     print("Loading NFL data for years:", years)
#     games = nfl.import_schedules(years)
#     pbp = nfl.import_pbp_data(years)
#     print(f"Loaded {len(games)} games, {len(pbp)} play-by-play rows.")
#     features = build_features(games, pbp)

#     # Train models: returns home/away regressors now
#     clf, home_reg, away_reg, feature_cols, holdout = train_models(features, tune=True, n_iter=24, cv_splits=5, save_models=False, model_dir='models')

#     # Optionally run explainability for classifier (example)
#     X_train, X_test, y_cls_train, y_cls_test, y_home_train, y_home_test, y_away_train, y_away_test = holdout
#     explain_outdir = 'explain_outputs_clf'
#     print(f"Generating explainability outputs to {explain_outdir} ...")
#     try:
#         explain_results = explain_model_workflow(clf, X_train, X_test, y_cls_test, feature_cols, outdir=explain_outdir, do_shap=False, do_permutation=True, do_tree=True, top_k=10)
#         print("Explain results keys:", list(explain_results.keys()))
#     except Exception as e:
#         print("Explain workflow failed:", e)

#     # Predict upcoming games using home/away regressors
#     predict_upcoming_games(games, features, clf, home_reg, away_reg, feature_cols)

# if __name__ == "__main__":
#     main()

# """
# nfl_game_prediction.py

# Full updated script with:
#  - time-aware train/test splitting (temporal holdout)
#  - hyperparameter tuning for regressors (HGB, XGB, RF) with RandomizedSearchCV and TimeSeriesSplit,
#    optimized using neg_mean_absolute_error
#  - time-series-aware stacking (manual OOF stacking that works with TimeSeriesSplit)
#  - residual-correction stage trained on OOF ensemble predictions (no leakage)
#  - classifier for winner prediction (XGBoost) trained on same time-aware split
#  - prediction function that uses stacking + residual correction to produce home/away score, spread, total
#  - helpers for explainability & saving models are left minimal to keep focus on modeling pipeline
# """

import os
import pandas as pd
import numpy as np
import nfl_data_py as nfl
from datetime import datetime
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV
from sklearn.metrics import accuracy_score, mean_absolute_error
from xgboost import XGBClassifier, XGBRegressor

from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
import joblib
import scipy.stats as stats

# ------------------------
# Data / feature engineering (kept similar to your previous versions)
# ------------------------

def add_vegas_lines(games):
    if 'spread_line' not in games.columns:
        games['spread_line'] = 0.0
    if 'total_line' not in games.columns:
        games['total_line'] = 44.0
    games['spread_line'] = games['spread_line'].fillna(0.0)
    games['total_line'] = games['total_line'].fillna(44.0)
    return games

def get_date_col(games):
    for candidate in ['gametime', 'start_time', 'datetime', 'game_date', 'start_date', 'date']:
        if candidate in games.columns:
            return candidate
    raise Exception(f"No valid date column found in games columns: {games.columns.tolist()}")

def compute_team_game_stats(pbp):
    pbp = pbp[~pbp.get('season_type', pd.Series()).isin(['PRE'])]
    pbp = pbp[pbp.get('posteam', pd.Series()).notnull() & pbp.get('defteam', pd.Series()).notnull()]
    pbp['explosive_play'] = ((pbp.get('passing_yards', 0) >= 20) | (pbp.get('rushing_yards', 0) >= 20)).astype(int)
    pbp['is_red_zone'] = pbp.get('yardline_100', 999) <= 20
    pbp['is_third_down'] = pbp.get('down', 0) == 3

    pbp['third_down_converted'] = (
        (pbp['is_third_down']) &
        ((pbp.get('first_down', 0) == 1) | (pbp.get('touchdown', 0) == 1))
    ).astype(int)

    agg_stats = pbp.groupby(['game_id', 'posteam']).agg(
        points_scored=('touchdown', 'sum'),
        pass_yards=('passing_yards', 'sum'),
        rush_yards=('rushing_yards', 'sum'),
        turnovers=('interception', 'sum'),
        fumbles=('fumble_lost', 'sum'),
        explosive_plays=('explosive_play', 'sum'),
        total_plays=('play_id', 'count'),
        red_zone_plays=('is_red_zone', 'sum'),
        red_zone_tds=('touchdown', lambda x: x[pbp.loc[x.index, 'yardline_100'] <= 20].sum() if len(x) > 0 else 0),
        third_down_plays=('is_third_down', 'sum'),
        third_down_conversions=('third_down_converted', 'sum'),
        sacks=('sack', 'sum')
    ).reset_index()
    agg_stats['total_yards'] = agg_stats['pass_yards'] + agg_stats['rush_yards']
    agg_stats['turnovers'] = agg_stats['turnovers'] + agg_stats['fumbles']
    agg_stats['explosive_play_rate'] = agg_stats['explosive_plays'] / agg_stats['total_plays'].replace(0, np.nan)
    agg_stats['red_zone_eff'] = agg_stats['red_zone_tds'] / agg_stats['red_zone_plays'].replace(0, np.nan)
    agg_stats['third_down_conv_rate'] = agg_stats['third_down_conversions'] / agg_stats['third_down_plays'].replace(0, np.nan)

    defense = agg_stats.rename(columns={
        "posteam": "defteam",
        "turnovers": "def_turnovers",
        "sacks": "def_sacks"
    })[['game_id','defteam','def_turnovers','def_sacks']]
    return agg_stats, defense

def get_rest_days(games, date_col):
    rest = []
    for team in set(games['home_team']).union(set(games['away_team'])):
        tgames = games[(games['home_team'] == team) | (games['away_team'] == team)].sort_values(['season', 'week'])
        last_date = None
        for idx, row in tgames.iterrows():
            gdate = pd.to_datetime(row[date_col])
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
    date_col = get_date_col(games)
    agg_stats, defense = compute_team_game_stats(pbp)

    home = games[['game_id','season','week',date_col,'home_team','away_team','home_score','away_score','spread_line','total_line']].rename(
        columns={date_col:'date','home_team':'team','away_team':'opp','home_score':'points_scored','away_score':'opp_points'}
    )
    home['is_home'] = 1
    away = games[['game_id','season','week',date_col,'away_team','home_team','away_score','home_score','spread_line','total_line']].rename(
        columns={date_col:'date','away_team':'team','home_team':'opp','away_score':'points_scored','home_score':'opp_points'}
    )
    away['is_home'] = 0
    long_games = pd.concat([home, away], ignore_index=True)

    long_games['team'] = long_games['team'].astype(str)
    agg_stats['posteam'] = agg_stats['posteam'].astype(str)
    long_games['game_id'] = long_games['game_id'].astype(str)
    agg_stats['game_id'] = agg_stats['game_id'].astype(str)
    long_games = long_games.merge(agg_stats, how='left', left_on=['game_id','team'], right_on=['game_id','posteam'])

    defense['defteam'] = defense['defteam'].astype(str)
    long_games = long_games.merge(defense, how='left', left_on=['game_id','opp'], right_on=['game_id','defteam'], suffixes=('', '_opp'))

    for col in ['points_scored', 'total_yards', 'turnovers', 'explosive_play_rate', 'red_zone_eff', 'third_down_conv_rate', 'sacks', 'def_sacks', 'def_turnovers']:
        if col not in long_games.columns:
            long_games[col] = 0
        long_games[col] = long_games[col].fillna(0)

    long_games = long_games.sort_values(['team','season','week'])
    for stat in ['points_scored','total_yards','turnovers','explosive_play_rate','red_zone_eff','third_down_conv_rate','sacks','def_sacks','def_turnovers']:
        long_games[f'{stat}_rolling5'] = long_games.groupby('team')[stat].rolling(3, min_periods=1).mean().reset_index(0,drop=True)

    rest_df = get_rest_days(games, date_col)
    long_games = long_games.merge(rest_df, how='left', on=['game_id','team'])

    features = []
    for _, row in games.iterrows():
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
            'home_explosive_rate': home_last.get('explosive_play_rate_rolling5', 0),
            'away_explosive_rate': away_last.get('explosive_play_rate_rolling5', 0),
            'home_rz_eff': home_last.get('red_zone_eff_rolling5', 0),
            'away_rz_eff': away_last.get('red_zone_eff_rolling5', 0),
            'home_3rd_down': home_last.get('third_down_conv_rate_rolling5', 0),
            'away_3rd_down': away_last.get('third_down_conv_rate_rolling5', 0),
            'home_sacks': home_last.get('sacks_rolling5', 0),
            'away_sacks': away_last.get('sacks_rolling5', 0),
            'home_def_turnover_margin': home_last.get('def_turnovers_rolling5', 0) - home_last.get('turnovers_rolling5', 0),
            'away_def_turnover_margin': away_last.get('def_turnovers_rolling5', 0) - away_last.get('turnovers_rolling5', 0),
            'home_score': row.get('home_score', np.nan),
            'away_score': row.get('away_score', np.nan)
        })
    df = pd.DataFrame(features)
    df = df.dropna(subset=['home_score','away_score'])
    df['home_win'] = (df['home_score'] > df['away_score']).astype(int)
    df['total_points'] = df['home_score'] + df['away_score']
    df['spread_actual'] = df['home_score'] - df['away_score']
    return df

# ------------------------
# Time-aware split & tuning
# ------------------------

def time_train_test_split(df, X, y_cols, test_size=0.15):
    df_sorted = df.sort_values(['season', 'week']).reset_index(drop=True)
    n = len(df_sorted)
    split_at = int(n * (1 - test_size))
    train_idx = df_sorted.index[:split_at]
    test_idx = df_sorted.index[split_at:]
    X_train = X.iloc[train_idx].reset_index(drop=True)
    X_test = X.iloc[test_idx].reset_index(drop=True)
    ys_train = []
    ys_test = []
    for y in y_cols:
        ys_train.append(y.iloc[train_idx].reset_index(drop=True))
        ys_test.append(y.iloc[test_idx].reset_index(drop=True))
    return X_train, X_test, ys_train, ys_test

def tune_regressors_mae(X, y, n_iter=20, cv_splits=5, random_state=42):
    tscv = TimeSeriesSplit(n_splits=cv_splits)
    results = {}

    # HGB tuning
    hgb = HistGradientBoostingRegressor(loss='absolute_error', random_state=random_state)
    hgb_param_dist = {
        'max_iter': [200, 400, 800],
        'learning_rate': [0.01, 0.03, 0.05],
        'max_depth': [3, 5, 8],
        'min_samples_leaf': [20, 50, 100]
    }
    rs_hgb = RandomizedSearchCV(hgb, hgb_param_dist, n_iter=min(n_iter, 12), cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=random_state, verbose=0)
    rs_hgb.fit(X, y)
    results['hgb'] = rs_hgb.best_estimator_

    # XGB tuning
    xgb = XGBRegressor(random_state=random_state, verbosity=0)
    xgb_param_dist = {
        'n_estimators': [100, 200, 400],
        'max_depth': [3, 6, 8],
        'learning_rate': [0.01, 0.03, 0.05],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0]
    }
    rs_xgb = RandomizedSearchCV(xgb, xgb_param_dist, n_iter=min(n_iter, 12), cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=random_state, verbose=0)
    rs_xgb.fit(X, y)
    results['xgb'] = rs_xgb.best_estimator_

    # RF tuning
    rf = RandomForestRegressor(random_state=random_state)
    rf_param_dist = {
        'n_estimators': [100, 200, 400],
        'max_depth': [None, 6, 12],
        'max_features': ['auto', 'sqrt']
    }
    rs_rf = RandomizedSearchCV(rf, rf_param_dist, n_iter=min(n_iter, 8), cv=tscv, scoring='neg_mean_absolute_error', n_jobs=-1, random_state=random_state, verbose=0)
    rs_rf.fit(X, y)
    results['rf'] = rs_rf.best_estimator_

    return results

# ------------------------
# Time-series stacking helpers (manual OOF stacking with TimeSeriesSplit)
# ------------------------

def train_time_series_stacking(base_estimators, meta_estimator, X, y, cv):
    """
    base_estimators: list of (name, estimator) unfitted
    meta_estimator: estimator instance (unfitted)
    X: DataFrame or array (training)
    y: Series or array (training)
    cv: TimeSeriesSplit instance
    Returns dict with scaler, bases (fitted on full training), meta (fitted on oof_preds), oof_preds
    """
    # scale
    scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X)

    n_samples = Xs.shape[0]
    n_base = len(base_estimators)
    oof_preds = np.zeros((n_samples, n_base), dtype=float)

    # produce OOF predictions for each base with time-series CV
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(Xs)):
        X_tr, X_te = Xs[train_idx], Xs[test_idx]
        # handle y as Series or array
        if isinstance(y, pd.Series):
            y_tr = y.iloc[train_idx].values
        else:
            y_tr = y[train_idx]
        for i, (name, est) in enumerate(base_estimators):
            est_fold = clone(est)
            est_fold.fit(X_tr, y_tr)
            oof_preds[test_idx, i] = est_fold.predict(X_te)

    # fit meta on OOF predictions
    meta = clone(meta_estimator)
    y_array = y.values if isinstance(y, pd.Series) else y
    meta.fit(oof_preds, y_array)

    # fit bases on full training data (scaled)
    fitted_bases = []
    for name, est in base_estimators:
        est_full = clone(est)
        est_full.fit(Xs, y_array)
        fitted_bases.append((name, est_full))

    return {
        'scaler': scaler,
        'bases': fitted_bases,
        'meta': meta,
        'oof_preds': oof_preds
    }

def predict_time_series_stacking(stack_model, X):
    """
    Given model dict from train_time_series_stacking and X (DataFrame/array),
    returns meta-level predictions for X.
    """
    scaler = stack_model['scaler']
    Xs = scaler.transform(X)
    base_preds = np.column_stack([est.predict(Xs) for name, est in stack_model['bases']])
    return stack_model['meta'].predict(base_preds)

# ------------------------
# Training: classifier + time-series stacking + residual correction
# ------------------------

def train_models(df, tune_regressors=True, n_iter=20, cv_splits=5, save_models=False, model_dir='models', random_state=42, test_size=0.15):
    feature_cols = [
        'home_points_avg','away_points_avg','home_yards_avg','away_yards_avg',
        'home_tov_avg','away_tov_avg','home_rest','away_rest','home_is_favorite',
        'spread','over_under',
        'home_explosive_rate', 'away_explosive_rate',
        'home_rz_eff', 'away_rz_eff',
        'home_3rd_down', 'away_3rd_down',
        'home_sacks', 'away_sacks',
        'home_def_turnover_margin', 'away_def_turnover_margin'
    ]

    X = df[feature_cols].fillna(0).reset_index(drop=True)
    y_cls = df['home_win'].reset_index(drop=True)
    y_home = df['home_score'].reset_index(drop=True)
    y_away = df['away_score'].reset_index(drop=True)

    # Time-aware split
    X_train, X_test, [y_cls_train, y_home_train, y_away_train], [y_cls_test, y_home_test, y_away_test] = time_train_test_split(
        df.reset_index(drop=True),
        X,
        [y_cls, y_home, y_away],
        test_size=test_size
    )

    # Classifier (XGBoost) trained on X_train
    clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=300, max_depth=10, learning_rate=0.08, random_state=random_state, verbosity=0)
    clf.fit(X_train, y_cls_train)

    # Tune regressors on X_train (time-series aware)
    if tune_regressors:
        print("Tuning regressors for home score (this will take time)...")
        best_home = tune_regressors_mae(X_train, y_home_train, n_iter=n_iter, cv_splits=cv_splits, random_state=random_state)
        print("Tuning regressors for away score (this will take time)...")
        best_away = tune_regressors_mae(X_train, y_away_train, n_iter=n_iter, cv_splits=cv_splits, random_state=random_state+1)
    else:
        best_home = {
            'hgb': HistGradientBoostingRegressor(loss='absolute_error', max_iter=300, learning_rate=0.05, max_depth=6, random_state=random_state),
            'xgb': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=random_state, verbosity=0),
            'rf': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=random_state)
        }
        best_away = {
            'hgb': HistGradientBoostingRegressor(loss='absolute_error', max_iter=300, learning_rate=0.05, max_depth=6, random_state=random_state+1),
            'xgb': XGBRegressor(n_estimators=200, max_depth=6, learning_rate=0.05, random_state=random_state+1, verbosity=0),
            'rf': RandomForestRegressor(n_estimators=200, max_depth=12, random_state=random_state+1)
        }

    # Prepare base estimator lists (unfitted clones will be used in train_time_series_stacking)
    base_home = [('hgb', best_home['hgb']), ('xgb', best_home['xgb']), ('rf', best_home['rf'])]
    base_away = [('hgb', best_away['hgb']), ('xgb', best_away['xgb']), ('rf', best_away['rf'])]
    meta = Ridge(alpha=1.0)

    tscv = TimeSeriesSplit(n_splits=cv_splits)

    # Train time-series stacking for home
    print("Training time-series stacking ensemble for HOME...")
    home_stack = train_time_series_stacking(base_home, meta, X_train, y_home_train, tscv)

    # Compute OOF ensemble predictions and residuals for home, then train residual model
    home_oof_preds = home_stack['oof_preds']                     # shape (n_train, n_base)
    home_oof_ensemble = home_stack['meta'].predict(home_oof_preds)  # ensemble's OOF-level fitted predictions
    home_residuals = (y_home_train.values if isinstance(y_home_train, pd.Series) else y_home_train) - home_oof_ensemble

    home_resid_model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=200, learning_rate=0.05, max_depth=4, random_state=random_state+10)
    home_resid_pipeline = make_pipeline(StandardScaler(), home_resid_model)
    home_resid_pipeline.fit(X_train, home_residuals)

    home_model = {'stack': home_stack, 'resid': home_resid_pipeline}

    # Train time-series stacking for away
    print("Training time-series stacking ensemble for AWAY...")
    away_stack = train_time_series_stacking(base_away, meta, X_train, y_away_train, tscv)

    away_oof_preds = away_stack['oof_preds']
    away_oof_ensemble = away_stack['meta'].predict(away_oof_preds)
    away_residuals = (y_away_train.values if isinstance(y_away_train, pd.Series) else y_away_train) - away_oof_ensemble

    away_resid_model = HistGradientBoostingRegressor(loss='absolute_error', max_iter=200, learning_rate=0.05, max_depth=4, random_state=random_state+11)
    away_resid_pipeline = make_pipeline(StandardScaler(), away_resid_model)
    away_resid_pipeline.fit(X_train, away_residuals)

    away_model = {'stack': away_stack, 'resid': away_resid_pipeline}

    # Evaluate on holdout
    home_base_test = predict_time_series_stacking(home_model['stack'], X_test)
    home_resid_test = home_model['resid'].predict(X_test)
    home_pred_test = home_base_test + home_resid_test

    away_base_test = predict_time_series_stacking(away_model['stack'], X_test)
    away_resid_test = away_model['resid'].predict(X_test)
    away_pred_test = away_base_test + away_resid_test

    spread_pred = home_pred_test - away_pred_test
    total_pred = home_pred_test + away_pred_test

    spread_actual = y_home_test - y_away_test
    total_actual = y_home_test + y_away_test

    spread_mae = mean_absolute_error(spread_actual, spread_pred)
    total_mae = mean_absolute_error(total_actual, total_pred)
    cls_acc = accuracy_score(y_cls_test, clf.predict(X_test))

    print(f"[HOLDOUT] Classifier acc: {cls_acc:.3f} | Spread MAE: {spread_mae:.3f} | Total MAE: {total_mae:.3f}")

    if save_models:
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(clf, os.path.join(model_dir, 'clf.joblib'))
        joblib.dump(home_model, os.path.join(model_dir, 'home_model.joblib'))
        joblib.dump(away_model, os.path.join(model_dir, 'away_model.joblib'))
        print(f"Saved models to {model_dir}")

    holdout = (X_train, X_test, y_cls_train, y_cls_test, y_home_train, y_home_test, y_away_train, y_away_test)
    return clf, home_model, away_model, feature_cols, holdout

# ------------------------
# Prediction utilities
# ------------------------

def get_last_stats(team, features, is_home):
    games = features[(features['home_team'] == team) if is_home else (features['away_team'] == team)]
    if games.empty:
        return [0]*9
    last = games.iloc[-1]
    if is_home:
        return [
            last['home_points_avg'],
            last['home_yards_avg'],
            last['home_tov_avg'],
            last['home_rest'],
            last['home_explosive_rate'],
            last['home_rz_eff'],
            last['home_3rd_down'],
            last['home_sacks'],
            last['home_def_turnover_margin']
        ]
    else:
        return [
            last['away_points_avg'],
            last['away_yards_avg'],
            last['away_tov_avg'],
            last['away_rest'],
            last['away_explosive_rate'],
            last['away_rz_eff'],
            last['away_3rd_down'],
            last['away_sacks'],
            last['away_def_turnover_margin']
        ]

def predict_upcoming_games(games, features, clf, home_model, away_model, feature_cols):
    future_games = games[games['home_score'].isna() | games['away_score'].isna()]
    if future_games.empty:
        print("No unplayed games found in the schedule.")
        return

    if "week" in future_games.columns:
        next_week = future_games["week"].min()
        filtered_games = future_games[future_games["week"] == next_week].copy()
        print(f"\n--- Predictions for Upcoming Games: Week {next_week} ---")
    else:
        filtered_games = future_games.copy()
        print("\n--- Predictions for All Unplayed Games (no week info found) ---")

    for _, row in filtered_games.iterrows():
        home_team = row['home_team']
        away_team = row['away_team']
        spread = row.get('spread_line', 0.0)
        over_under = row.get('total_line', 44.0)
        home_stats = get_last_stats(home_team, features, True)
        away_stats = get_last_stats(away_team, features, False)
        (home_pts, home_yds, home_tov, home_rest, home_explosive, home_rz, home_3rd, home_sacks, home_def_to) = home_stats
        (away_pts, away_yds, away_tov, away_rest, away_explosive, away_rz, away_3rd, away_sacks, away_def_to) = away_stats
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
            'over_under': over_under,
            'home_explosive_rate': home_explosive,
            'away_explosive_rate': away_explosive,
            'home_rz_eff': home_rz,
            'away_rz_eff': away_rz,
            'home_3rd_down': home_3rd,
            'away_3rd_down': away_3rd,
            'home_sacks': home_sacks,
            'away_sacks': away_sacks,
            'home_def_turnover_margin': home_def_to,
            'away_def_turnover_margin': away_def_to
        }])
        # Ensure all expected feature cols exist and are ordered
        for c in feature_cols:
            if c not in input_df.columns:
                input_df[c] = 0
        input_df = input_df[feature_cols]

        win_prob = clf.predict_proba(input_df)[0,1]

        # stacking + residual correction for home
        home_base = predict_time_series_stacking(home_model['stack'], input_df.reshape(1, -1) if isinstance(input_df, np.ndarray) else input_df)
        home_resid = home_model['resid'].predict(input_df)[0]
        home_pred = float(home_base + home_resid)

        # stacking + residual correction for away
        away_base = predict_time_series_stacking(away_model['stack'], input_df.reshape(1, -1) if isinstance(input_df, np.ndarray) else input_df)
        away_resid = away_model['resid'].predict(input_df)[0]
        away_pred = float(away_base + away_resid)

        spread_pred = home_pred - away_pred
        total_pred = home_pred + away_pred

        if spread_pred > 0:
            spread_str = f"Home team ({home_team}) -{abs(spread_pred):.1f}"
        else:
            spread_str = f"Away team ({away_team}) -{abs(spread_pred):.1f}"
        week_str = f"Week {row['week']}" if "week" in row else ""
        print(f"{week_str} | {home_team} vs {away_team} | Prob {home_team} Wins: {win_prob:.1%} | Pred {home_team}: {home_pred:.1f} | Pred {away_team}: {away_pred:.1f} | Pred Total: {total_pred:.1f} | Pred Spread: {spread_str}")
        print("-----------------------------------------------")

# ------------------------
# Main
# ------------------------

def main():
    years = [2010,2011,2012,2013,2014,2015,2016,2017,2018,2019,2020,2021,2022,2023,2024,2025]  # adjust as needed
    print("Loading NFL data for years:", years)
    games = nfl.import_schedules(years)
    pbp = nfl.import_pbp_data(years)
    print(f"Loaded {len(games)} games, {len(pbp)} play-by-play rows.")
    features = build_features(games, pbp)

    # Train everything (tuning is expensive; reduce n_iter if you need speed)
    clf, home_model, away_model, feature_cols, holdout = train_models(features, tune_regressors=True, n_iter=20, cv_splits=5, save_models=False)

    # Predict upcoming games using ensembled + residual-corrected models
    predict_upcoming_games(games, features, clf, home_model, away_model, feature_cols)

if __name__ == "__main__":
    main()