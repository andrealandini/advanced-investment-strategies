import pandas as pd
import numpy as np
from scipy.stats import entropy

# === Load your existing features file (with returns etc.) ===
df = pd.read_csv("../data/crsp_features_full_v.txt", sep="\t")
df = df.sort_values(['permno', 'date']).reset_index(drop=True)

# --- Rolling Shannon entropy helper ---
def rolling_entropy(x, bins=20):
    """Compute Shannon entropy (bits) of returns within window."""
    hist, _ = np.histogram(x, bins=bins, density=True)
    hist = hist[hist > 0]  # avoid log(0)
    return entropy(hist, base=2)

# --- Compute target for each firm ---
def add_entropy(group):
    group = group.sort_values('date').copy()
    group['entropy_20d_forward'] = (
        group['ret']
        .shift(-19)                # align t with t+1:t+20
        .rolling(20)
        .apply(rolling_entropy, raw=False)
    )
    return group[['permno', 'date', 'entropy_20d_forward']]

entropy_df = (
    df.groupby('permno', group_keys=False)
      .apply(add_entropy)
      .dropna(subset=['entropy_20d_forward'])
)

# --- Save result ---
entropy_df.to_csv("../data/entropy_target.txt", sep="\t", index=False)
print("✅ Saved ../data/entropy_target.txt with columns: permno, date, entropy_20d_forward")

