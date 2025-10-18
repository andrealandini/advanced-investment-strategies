import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy.stats import spearmanr
from tqdm import tqdm

# === Parameters ===
B = 100
k = 3
alphas = [0.1, 1.0, 10.0]
np.random.seed(42)

# === Helper: Evaluation metrics ===
def eval_metrics(y_true, y_pred):
    return {
        'R2': r2_score(y_true, y_pred),
        'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
        'MAE': mean_absolute_error(y_true, y_pred),
        'Bias': np.mean(y_pred - y_true),
        'Spearman': spearmanr(y_true, y_pred).correlation,
        'DirAcc': np.mean(np.sign(np.diff(y_true)) == np.sign(np.diff(y_pred)))
    }

# === Load base model predictions ===
df = pd.read_csv('../data/entropy-output/ml_base_predictions_e.csv')
base_models = ['ridge_pred', 'lasso_pred', 'enet_pred', 'rf_pred', 'gbrt_pred', 'nn_pred']

permnos = df['permno'].unique()
all_y_true, all_y_pred, all_permno = [], [], []

print(f"Running cross-sectional VASA on {len(permnos)} firms...")

for permno in tqdm(permnos):
    sub = df[df['permno'] == permno].dropna(subset=base_models + ['true'])
    if len(sub) < 60:
        continue

    n_train = int(0.8 * len(sub))
    X_train = sub[base_models].iloc[:n_train].values
    y_train = sub['true'].iloc[:n_train].values
    X_test  = sub[base_models].iloc[n_train:].values
    y_test  = sub['true'].iloc[n_train:].values

    vasa_preds_test_all = []
    for _ in range(B):
        subset_idx = np.random.choice(len(base_models), size=k, replace=False)
        Xb_train, Xb_test = X_train[:, subset_idx], X_test[:, subset_idx]
        model = RidgeCV(alphas=alphas)
        model.fit(Xb_train, y_train)
        vasa_preds_test_all.append(model.predict(Xb_test))

    vasa_preds_test_all = np.array(vasa_preds_test_all)
    vasa_pred = vasa_preds_test_all.mean(axis=0)

    all_y_true.append(y_test)
    all_y_pred.append(vasa_pred)
    all_permno.extend([permno] * len(y_test))

# === Combine and evaluate globally ===
y_true_all = np.concatenate(all_y_true)
y_pred_all = np.concatenate(all_y_pred)
permno_all = np.array(all_permno)

metrics = eval_metrics(y_true_all, y_pred_all)
metrics_df = pd.DataFrame([metrics])
metrics_df.to_csv('../data/entropy-output/vasa_cross_metrics_e.csv', index=False)
print("✅ Saved ../data/entropy-output/vasa_cross_metrics_e.csv")

r2_df = pd.DataFrame({'Train_R2': [np.nan], 'Test_R2': [metrics['R2']]}, index=['VASA_Cross'])
r2_df.to_csv('../data/entropy-output/vasa_cross_r2_e.csv')
print("✅ Saved ../data/entropy-output/vasa_cross_r2_e.csv")

print("\nCross-sectional R²:")
print(r2_df)
print("\nFull metrics:")
print(metrics_df)

