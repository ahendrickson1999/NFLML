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
