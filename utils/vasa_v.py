import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from tqdm import trange

# === Parameters ===
B = 100          # subsamples
k = 3            # number of base models per subsample
alphas = [0.1, 1.0, 10.0]
np.random.seed(42)

# === Helper: Evaluation metrics ===
def eval_metrics(y_true, y_pred, vasa_preds_all=None, best_base_r2=None):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    bias = np.mean(y_pred - y_true)
    spearman = spearmanr(y_true, y_pred).correlation
    dir_acc = np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    qlike = np.mean(y_true / y_pred - np.log(y_true / y_pred) - 1)
    tail_mask = y_true > np.quantile(y_true, 0.95)
    tail_rmse = np.sqrt(np.mean((y_true[tail_mask] - y_pred[tail_mask])**2))
    pred_var = np.var(vasa_preds_all, axis=0).mean() if vasa_preds_all is not None else np.nan
    improvement = (r2 - best_base_r2) / best_base_r2 if best_base_r2 else np.nan
    mean_ratio = np.mean(y_pred) / np.mean(y_true)
    return {
        'R2': r2, 'RMSE': rmse, 'MAE': mae, 'Bias': bias, 'Spearman': spearman,
        'DirAcc': dir_acc, 'QLIKE': qlike, 'TailRMSE': tail_rmse,
        'PredVar': pred_var, 'Improvement': improvement, 'MeanRatio': mean_ratio
    }

# === Load predictions ===
df = pd.read_csv('../data/ml_base_predictions_v.csv')
base_models = ['ridge_pred', 'lasso_pred', 'enet_pred', 'rf_pred', 'gbrt_pred', 'nn_pred']

# === Split train/test chronologically ===
n_train = int(0.8 * len(df))
X_train = df[base_models].iloc[:n_train].values
y_train = df['true'].iloc[:n_train].values
X_test  = df[base_models].iloc[n_train:].values
y_test  = df['true'].iloc[n_train:].values

# === Best base model for improvement ratio ===
r2_base = {m: r2_score(y_test, df[m].iloc[n_train:]) for m in base_models}
best_base_r2 = max(r2_base.values())
print(f"Best base model test R²: {best_base_r2:.4f}")

# === Run VASA ===
vasa_preds_train_all, vasa_preds_test_all = [], []

for b in trange(B, desc="Subsamples"):
    subset = np.random.choice(len(base_models), size=k, replace=False)
    Xb_train, Xb_test = X_train[:, subset], X_test[:, subset]
    model = RidgeCV(alphas=alphas)
    model.fit(Xb_train, y_train)
    vasa_preds_train_all.append(model.predict(Xb_train))
    vasa_preds_test_all.append(model.predict(Xb_test))

vasa_preds_train_all = np.array(vasa_preds_train_all)
vasa_preds_test_all = np.array(vasa_preds_test_all)
vasa_train_pred = vasa_preds_train_all.mean(axis=0)
vasa_test_pred = vasa_preds_test_all.mean(axis=0)

# === Evaluate ===
metrics_train = eval_metrics(y_train, vasa_train_pred, vasa_preds_train_all, best_base_r2)
metrics_test  = eval_metrics(y_test,  vasa_test_pred,  vasa_preds_test_all,  best_base_r2)

# === Save outputs ===
r2_df = pd.DataFrame({
    'Train_R2': [metrics_train['R2']],
    'Test_R2': [metrics_test['R2']]
}, index=['VASA'])
r2_df.to_csv('../data/vasa_r2_v.csv')
print("✅ Saved ../data/vasa_r2_v.csv")

metrics_df = pd.DataFrame([metrics_train, metrics_test], index=['Train', 'Test'])
metrics_df.to_csv('../data/vasa_metrics_v.csv')
print("✅ Saved ../data/vasa_metrics_v.csv")

print("\nR² summary:")
print(r2_df)
print("\nFull metrics:")
print(metrics_df)

