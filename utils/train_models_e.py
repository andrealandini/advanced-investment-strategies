import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge, Lasso, ElasticNet
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping

# === Load dataset ===
features_df = pd.read_csv('../data/crsp_features_full_v.txt', sep='\t')
entropy_df = pd.read_csv('../data/entropy_target.txt', sep='\t')



# Merge on firm and date
df = features_df.merge(entropy_df, on=['permno', 'date'], how='inner')
print("✅ Loaded merged dataset:", df.shape)

# === Feature and target selection ===
features = [
    'ret_lag1', 'ret_excess_lag1', 'ret_lag2', 'ret_lag3',
    'ret_lag4', 'ret_lag5', 'skew_20d', 'mean_20d', 'sd_20d'
]
target = 'entropy_20d_forward'
df = df.dropna(subset=features + [target])

X = df[features].values
y = df[target].values

# === Standardize predictors ===
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# === Chronological split (80/20) ===
n_train = int(0.8 * len(df))
X_train, X_test = X_scaled[:n_train], X_scaled[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# === Evaluation function ===
def evaluate(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Bias': np.mean(y_pred - y_true),
        'Spearman': np.corrcoef(np.argsort(y_true), np.argsort(y_pred))[0,1],
        'DirAcc': np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    }

results = {}
preds_train, preds_test = {}, {}

# === Ridge ===
ridge = Ridge(alpha=1.0)
ridge.fit(X_train, y_train)
preds_train['ridge_pred'] = ridge.predict(X_train)
preds_test['ridge_pred'] = ridge.predict(X_test)
results['ridge'] = evaluate(y_test, preds_test['ridge_pred'])
print("ridge done")

# === LASSO ===
lasso = Lasso(alpha=0.001, max_iter=2000)
lasso.fit(X_train, y_train)
preds_train['lasso_pred'] = lasso.predict(X_train)
preds_test['lasso_pred'] = lasso.predict(X_test)
results['lasso'] = evaluate(y_test, preds_test['lasso_pred'])
print("lasso done")

# === Elastic Net ===
enet = ElasticNet(alpha=0.001, l1_ratio=0.5, max_iter=2000)
enet.fit(X_train, y_train)
preds_train['enet_pred'] = enet.predict(X_train)
preds_test['enet_pred'] = enet.predict(X_test)
results['enet'] = evaluate(y_test, preds_test['enet_pred'])
print("enet done")

# === Random Forest ===
rf = RandomForestRegressor(
    n_estimators=200, max_depth=6,
    max_features='sqrt', random_state=42, n_jobs=-1
)
rf.fit(X_train, y_train)
preds_train['rf_pred'] = rf.predict(X_train)
preds_test['rf_pred'] = rf.predict(X_test)
results['rf'] = evaluate(y_test, preds_test['rf_pred'])
print("rf done")


# === Gradient Boosting ===
gbrt = GradientBoostingRegressor(
    n_estimators=200, max_depth=3, learning_rate=0.05
)
gbrt.fit(X_train, y_train)
preds_train['gbrt_pred'] = gbrt.predict(X_train)
preds_test['gbrt_pred'] = gbrt.predict(X_test)
results['gbrt'] = evaluate(y_test, preds_test['gbrt_pred'])
print("gbrt done")

# === Neural Network ===
nn = Sequential([
    Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
    Dropout(0.3),
    Dense(16, activation='relu'),
    Dense(1)
])
nn.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
nn.fit(
    X_train, y_train,
    epochs=30, batch_size=128, verbose=0,
    validation_split=0.2,
    callbacks=[EarlyStopping(patience=5, restore_best_weights=True)]
)
preds_train['nn_pred'] = nn.predict(X_train, verbose=0).flatten()
preds_test['nn_pred'] = nn.predict(X_test, verbose=0).flatten()
results['nn'] = evaluate(y_test, preds_test['nn_pred'])
print("nn done")

# === Combine predictions ===
pred_df = pd.DataFrame({
    'permno': df['permno'].values,
    'date': df['date'].values,
    **{k: np.concatenate([preds_train[k], preds_test[k]]) for k in preds_train.keys()},
    'true': np.concatenate([y_train, y_test])
})
pred_df.to_csv('../data/entropy-output/ml_base_predictions_e.csv', index=False)
print("✅ Saved ../data/entropy-output/ml_base_predictions_e.csv")

# === Save metrics ===
metrics_df = pd.DataFrame(results).T
metrics_df.to_csv('../data/entropy-output/ml_model_metrics_e.csv')
print("✅ Saved ../data/entropy-output/ml_model_metrics_e.csv")

r2_df = metrics_df[['R2']].rename(columns={'R2': 'Test_R2'})
r2_df.to_csv('../data/entropy-output/ml_r2_e.csv')
print("✅ Saved ../data/entropy-output/ml_r2_e.csv")


