"""
Train Random Forest on extracted BDT features.
Run once for 125 GeV (model1), once for 250 GeV (model2).
"""

import os
import time
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


TRAIN_ENERGY = 250        

FEATURE_DIR  = r"E:/Python/MSc_Project_Upgrade/datasets/bdt"
RESULTS_DIR  = rf"E:/Python/MSc_Project_Upgrade/results_analysis/bdt/RF_model{1 if TRAIN_ENERGY==125 else 2}_{TRAIN_ENERGY}GeV"
MODEL_TAG    = f"model{1 if TRAIN_ENERGY == 125 else 2}"

os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42

RF_PARAMS = {
    "n_estimators":  700,
    "max_depth":     None,      
    "min_samples_leaf": 5,
    "max_features":  "sqrt",    
    "class_weight":  "balanced",
    "oob_score": True,
    "n_jobs":        10,
    "random_state":  RANDOM_SEED,
}

# LOAD DATA

print("=" * 60)
print(f"RANDOM FOREST {MODEL_TAG.upper()} — Training at {TRAIN_ENERGY} GeV")
print("=" * 60)

train_path = os.path.join(FEATURE_DIR, f"bdt_features_{TRAIN_ENERGY}GeV_train.npz")
d = np.load(train_path, allow_pickle=True)

X_all = d["features"].astype(np.float32)
y_all = d["labels"].astype(np.int32)

n_sig = int(np.sum(y_all == 1))
n_bkg = int(np.sum(y_all == 0))
print(f"  Total jets : {len(y_all):,}  |  signal={n_sig:,}  background={n_bkg:,}")
print(f"  sig:bkg    = 1 : {n_bkg/n_sig:.2f}")


X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all,
    test_size=0.2,
    stratify=y_all,
    random_state=RANDOM_SEED,
)
print(f"  Train: {len(y_train):,}   Val: {len(y_val):,}")

# StandardScaler 
scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)

# TRAIN

print(f"\nTraining Random Forest ({RF_PARAMS['n_estimators']} trees)...")
print("  (no early stopping — will train all trees)")
t0 = time.time()

model = RandomForestClassifier(**RF_PARAMS,verbose=1)
model.fit(X_train, y_train)

elapsed    = time.time() - t0
val_scores = model.predict_proba(X_val)[:, 1]
val_auc    = roc_auc_score(y_val, val_scores)

print(f"\n  Val AUC       : {val_auc:.5f}")
print(f"  Training time : {elapsed:.1f}s")

# SAVE MODEL

model_path  = os.path.join(RESULTS_DIR, f"rf_{MODEL_TAG}_{TRAIN_ENERGY}GeV.pkl")
scaler_path = os.path.join(RESULTS_DIR, f"rf_{MODEL_TAG}_scaler.npz")

joblib.dump(model, model_path, compress=3)
np.savez_compressed(
    scaler_path,
    mean  = scaler.mean_.astype(np.float32),
    scale = scaler.scale_.astype(np.float32),
)

print(f"\n  Model  saved → {model_path}")
print(f"  Scaler saved → {scaler_path}")

report = [
    f"Random Forest {MODEL_TAG} Training Report",
    "=" * 50,
    f"Train energy   : {TRAIN_ENERGY} GeV",
    f"Total jets     : {len(y_all):,}",
    f"  signal       : {n_sig:,}",
    f"  background   : {n_bkg:,}",
    f"Train / Val    : {len(y_train):,} / {len(y_val):,}",
    f"Val AUC        : {val_auc:.5f}",
    f"Training time  : {elapsed:.1f}s",
    "",
    "Random Forest Hyperparameters:",
]
for k, v in RF_PARAMS.items():
    report.append(f"  {k:<25}: {v}")

report_str = "\n".join(report)
print("\n" + report_str)

with open(os.path.join(RESULTS_DIR, f"rf_{MODEL_TAG}_train_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_str + "\n")
