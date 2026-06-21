"""
Demand Prediction Script
Metric: score = max(0, 100 * r2_score(actual, predicted))
Submission: 41778 x 2  →  Index | demand
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

# ── 1. Load Data ──────────────────────────────────────────────────────────────
train = pd.read_csv('train.csv')
test  = pd.read_csv('test.csv')
test_index = test['Index'].copy()

# ── 2. Feature Engineering ────────────────────────────────────────────────────
def engineer(df, geohash_stats=None, is_train=True):
    df = df.copy()

    # Parse timestamp → hour, minute, time slot (0–95)
    df['hour']       = df['timestamp'].str.split(':').str[0].astype(int)
    df['minute']     = df['timestamp'].str.split(':').str[1].astype(int)
    df['time_index'] = df['hour'] * 4 + df['minute'] // 15

    # Cyclical encoding (captures midnight wrap-around)
    df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24)
    df['time_sin'] = np.sin(2 * np.pi * df['time_index'] / 96)
    df['time_cos'] = np.cos(2 * np.pi * df['time_index'] / 96)

    # Binary encode
    df['LargeVehicles'] = (df['LargeVehicles'] == 'Allowed').astype(int)
    df['Landmarks']     = (df['Landmarks'] == 'Yes').astype(int)

    # Fill missing categoricals before one-hot
    df['Weather']  = df['Weather'].fillna('Unknown')
    df['RoadType'] = df['RoadType'].fillna('Unknown')

    # Flag + fill missing temperature
    df['temp_missing'] = df['Temperature'].isna().astype(int)
    if is_train:
        temp_median = df['Temperature'].median()
        geohash_stats = {'temp_median': temp_median}
    df['Temperature'] = df['Temperature'].fillna(geohash_stats['temp_median'])

    # One-hot encode low-cardinality categoricals
    df = pd.get_dummies(df, columns=['Weather', 'RoadType'], drop_first=False)

    # Target encode geohash (mean / median / std demand per location)
    if is_train:
        geo_mean   = df.groupby('geohash')['demand'].mean().rename('geo_mean_demand')
        geo_median = df.groupby('geohash')['demand'].median().rename('geo_median_demand')
        geo_std    = df.groupby('geohash')['demand'].std().fillna(0).rename('geo_std_demand')
        geohash_stats.update({
            'geo_mean': geo_mean, 'geo_median': geo_median, 'geo_std': geo_std,
            'global_mean': df['demand'].mean(), 'global_median': df['demand'].median()
        })
    df = df.merge(geohash_stats['geo_mean'],   on='geohash', how='left')
    df = df.merge(geohash_stats['geo_median'], on='geohash', how='left')
    df = df.merge(geohash_stats['geo_std'],    on='geohash', how='left')
    if not is_train:
        df['geo_mean_demand'].fillna(geohash_stats['global_mean'],   inplace=True)
        df['geo_median_demand'].fillna(geohash_stats['global_median'], inplace=True)
        df['geo_std_demand'].fillna(0, inplace=True)

    drop_cols = ['Index', 'timestamp', 'geohash']
    if 'demand' in df.columns:
        drop_cols.append('demand')
    df = df.drop(columns=drop_cols, errors='ignore')
    return df.astype(float), geohash_stats


# ── 3. Prepare Features ────────────────────────────────────────────────────────
X_raw, geo_stats = engineer(train, is_train=True)
y = train['demand'].values

X_test_raw, _ = engineer(test, geohash_stats=geo_stats, is_train=False)
X, X_test = X_raw.align(X_test_raw, join='left', axis=1, fill_value=0)
X_test = X_test.fillna(0)

print(f"Features: {X.shape[1]} | Train rows: {X.shape[0]} | Test rows: {X_test.shape[0]}\n")

# ── 4. Define Models ──────────────────────────────────────────────────────────
models = {
    'Ridge': Pipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('model',   Ridge(alpha=1.0))
    ]),
    'XGBoost': XGBRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbosity=0
    ),
    'LightGBM': LGBMRegressor(
        n_estimators=500, learning_rate=0.05, max_depth=6,
        subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    ),
    'RandomForest': RandomForestRegressor(
        n_estimators=300, max_depth=14,
        random_state=42, n_jobs=-1
    ),
}

# ── 5. Cross-Validation ────────────────────────────────────────────────────────
print("── 5-Fold Cross-Validation (R² × 100) ──")
kf = KFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = {}

for name, model in models.items():
    scores = cross_val_score(model, X, y, cv=kf, scoring='r2')
    score_pct = np.maximum(0, scores * 100)
    cv_scores[name] = score_pct.mean()
    print(f"  {name:15s}: {score_pct.mean():.2f} ± {score_pct.std():.2f}")

best_name = max(cv_scores, key=cv_scores.get)
print(f"\n✓ Best single model: {best_name} ({cv_scores[best_name]:.2f})\n")

# ── 6. Train on Full Data & Predict ───────────────────────────────────────────
print("── Training on full data ──")
preds = {}
for name, model in models.items():
    model.fit(X, y)
    preds[name] = np.clip(model.predict(X_test), 0, 1)
    train_score = max(0, r2_score(y, model.predict(X)) * 100)
    print(f"  {name:15s}: train R² = {train_score:.2f}")

# ── 7. Ensemble (average of top 3 tree models) ────────────────────────────────
ensemble_pred = np.mean([preds['XGBoost'], preds['LightGBM'], preds['RandomForest']], axis=0)

# ── 8. Save Submission Files ──────────────────────────────────────────────────
print("\n── Saving submissions ──")
for name, pred in preds.items():
    fname = f"submission_{name}.csv"
    pd.DataFrame({'Index': test_index, 'demand': pred}).to_csv(fname, index=False)
    print(f"  Saved {fname}")

pd.DataFrame({'Index': test_index, 'demand': ensemble_pred}).to_csv('submission_ensemble.csv', index=False)
print("  Saved submission_ensemble.csv  ← RECOMMENDED\n")
print("Done! Submit submission_ensemble.csv for best results.")
