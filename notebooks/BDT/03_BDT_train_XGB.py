"""
Train XGBoost BDT on extracted features.
Run once for 125 GeV (model1), once for 250 GeV (model2).
"""

import os
import time
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler


TRAIN_ENERGY = 250        

FEATURE_DIR  = r"E:/Python/MSc_Project_Upgrade/datasets/bdt"
RESULTS_DIR  = rf"E:/Python/MSc_Project_Upgrade/results_analysis/bdt/XGB_model{1 if TRAIN_ENERGY==125 else 2}_{TRAIN_ENERGY}GeV"
MODEL_TAG    = f"model{1 if TRAIN_ENERGY == 125 else 2}"

os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42

XGB_PARAMS = {
    "n_estimators":        2500,   
    "max_depth":           4,
    "learning_rate":       0.02,
    "subsample":           0.8,
    "colsample_bytree":    0.8,
    "min_child_weight":    1,
    "reg_alpha":           0.0,
    "reg_lambda":          1.0,
    "objective":           "binary:logistic",
    "eval_metric":         "auc",
    "random_state":        RANDOM_SEED,
    "n_jobs":              -1,
    "early_stopping_rounds": 40,
}

# LOAD DATA

print("=" * 60)
print(f"XGBOOST {MODEL_TAG.upper()} — Training at {TRAIN_ENERGY} GeV")
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

print(f"\nTraining XGBoost...")
t0 = time.time()

model = xgb.XGBClassifier(
    **{k: v for k, v in XGB_PARAMS.items() if k != "early_stopping_rounds"},
    scale_pos_weight=n_bkg / n_sig,
    early_stopping_rounds=XGB_PARAMS["early_stopping_rounds"],
)
model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=50,
)

elapsed    = time.time() - t0
best_iter  = model.best_iteration
val_scores = model.predict_proba(X_val)[:, 1]
val_auc    = roc_auc_score(y_val, val_scores)

print(f"\n  Best iteration : {best_iter}")
print(f"  Val AUC        : {val_auc:.5f}")
print(f"  Training time  : {elapsed:.1f}s")

# SAVE MODEL

model_path  = os.path.join(RESULTS_DIR, f"xgb_{MODEL_TAG}_{TRAIN_ENERGY}GeV.json")
scaler_path = os.path.join(RESULTS_DIR, f"xgb_{MODEL_TAG}_scaler.npz")

model.save_model(model_path)
np.savez_compressed(
    scaler_path,
    mean  = scaler.mean_.astype(np.float32),
    scale = scaler.scale_.astype(np.float32),
)

print(f"\n  Model  saved → {model_path}")
print(f"  Scaler saved → {scaler_path}")

report = [
    f"XGBoost {MODEL_TAG} Training Report",
    "=" * 50,
    f"Train energy   : {TRAIN_ENERGY} GeV",
    f"Total jets     : {len(y_all):,}",
    f"  signal       : {n_sig:,}",
    f"  background   : {n_bkg:,}",
    f"Train / Val    : {len(y_train):,} / {len(y_val):,}",
    f"Best iteration : {best_iter}",
    f"Val AUC        : {val_auc:.5f}",
    f"Training time  : {elapsed:.1f}s",
    "",
    "XGBoost Hyperparameters:",
]
for k, v in XGB_PARAMS.items():
    report.append(f"  {k:<25}: {v}")

report_str = "\n".join(report)
print("\n" + report_str)

with open(os.path.join(RESULTS_DIR, f"xgb_{MODEL_TAG}_train_report.txt"), "w", encoding="utf-8") as f:
    f.write(report_str + "\n")
