import os
import time
import numpy as np
import uproot
import awkward as ak

ROOT_DIR  = r"E:/Python/MSc_Project_Upgrade/generation/root_files"
SAVE_DIR  = r"E:/Python/MSc_Project_Upgrade/datasets/bdt"

os.makedirs(SAVE_DIR, exist_ok=True)

# pT windows
PT_WINDOWS = {
    100: (15.0,  50.0),
    125: (15.0,  60.0),
    150: (15.0,  75.0),
    200: (15.0, 100.0),
    250: (15.0, 125.0),
    300: (15.0, 150.0),
}

# Training ROOT file names
TRAIN_FILES = {
    125: {
        "tau": "ee_tau_125GeV_175k_train.root",
        "jj":  "ee_jj_125GeV_80k_train.root",
        "bb":  "ee_bb_125GeV_60k_train.root",
    },
    250: {
        "tau": "ee_tau_250GeV_175k_train.root",
        "jj":  "ee_jj_250GeV_80k_train.root",
        "bb":  "ee_bb_250GeV_60k_train.root",
    },
}

# Test ROOT file names (50k per process per energy)
TEST_ENERGIES = [100, 125, 150, 200, 250, 300]

PROCESS_LABEL    = {"tau": 1, "jj": 0, "bb": 0}
PROCESS_SAMPLEID = {"tau": 0, "jj": 1, "bb": 2}

FEATURE_NAMES = [
    "pt", "mass", "ncharged", "nneutrals",
    "ehad", "chf", "nef",
    "tau1", "tau21", "tau32",
]


def safe_ratio(num, den, fill=1.0):
    """Compute num/den; where den==0 return fill."""
    ratio = np.where(den > 0, num / np.where(den > 0, den, 1.0), fill)
    return ratio.astype(np.float32)


# FIX NOTES
# ---------
# Two bugs existed in the original event_id construction:
#
# Bug 1 — event_id was a LOCAL index within pT-masked events only
#   (ak.local_index applied AFTER masking). This resets to 0 for
#   every process file, so tau/jj/bb all share overlapping ids.
#   The CNN pipeline uses a GLOBAL index (loop variable ev over ALL
#   events) plus an event_offset that chains across files.
#   Fix: use ak.local_index on the FULL unmasked array, then mask,
#   then add event_offset — identical convention to CNN.
#
# Bug 2 — no jet_index stored
#   Within one event, multiple jets can pass the pT cut.
#   Sorting by (sample_id, event_id) leaves jet order within an
#   event undefined across the two pipelines.
#   Fix: store jet_index = position of the jet in its event
#   (Delphes-native order, before masking drops other jets).
#   Alignment key becomes (sample_id, event_id, jet_index).
#
# These changes affect ONLY metadata (event_id, jet_index).
# The feature matrix and labels are byte-for-byte identical
# to the original, so all trained BDT models remain valid.


def extract_jets_from_root(root_path, pt_min, pt_max, process,
                           energy_gev, event_offset=0):
    """
    Open a single Delphes ROOT file and extract the 10 BDT features
    for all jets passing the pT window.

    Parameters
    ----------
    event_offset : int
        Added to every event_id so ids are globally unique across
        files processed in sequence — matches CNN pipeline convention.

    Returns
    -------
    dict of numpy arrays, plus 'n_events_total' for offset chaining.
    """
    print(f"  Opening: {os.path.basename(root_path)}")
    t0 = time.time()

    with uproot.open(root_path) as f:
        tree = f["Delphes"]

        pt_raw        = tree["Jet/Jet.PT"].array(library="ak")
        mass_raw      = tree["Jet/Jet.Mass"].array(library="ak")
        ncharged_raw  = tree["Jet/Jet.NCharged"].array(library="ak")
        nneutrals_raw = tree["Jet/Jet.NNeutrals"].array(library="ak")
        ehad_raw      = tree["Jet/Jet.EhadOverEem"].array(library="ak")
        tau_raw       = tree["Jet/Jet.Tau[5]"].array(library="ak")

    n_events_total = len(pt_raw)   # ALL events before any cut

    # ── pT mask (ragged, per-jet) ──────────────────────────────────
    mask = (pt_raw >= pt_min) & (pt_raw <= pt_max)

    # ── FIX 1: event_id — global index matching CNN convention ─────
    # local_index on the FULL array (0..n_events_total-1), then mask
    global_ev_idx = ak.local_index(pt_raw, axis=0)
    event_id = ak.to_numpy(
        ak.flatten(ak.broadcast_arrays(global_ev_idx, pt_raw)[0][mask])
    ).astype(np.int64) + event_offset

    # ── FIX 2: jet_index — position of jet within its event ────────
    # local_index on axis=1 gives 0,1,2,... per event; mask selects
    # surviving jets but preserves their original Delphes position
    jet_idx_in_event = ak.local_index(pt_raw, axis=1)
    jet_index = ak.to_numpy(
        ak.flatten(jet_idx_in_event[mask])
    ).astype(np.int32)

    # ── apply mask to feature arrays ──────────────────────────────
    def apply_mask(arr):
        return ak.to_numpy(ak.flatten(arr[mask]))

    pt_sel    = apply_mask(pt_raw).astype(np.float32)
    mass_sel  = apply_mask(mass_raw).astype(np.float32)
    nch_sel   = apply_mask(ncharged_raw).astype(np.float32)
    nne_sel   = apply_mask(nneutrals_raw).astype(np.float32)
    ehad_sel  = apply_mask(ehad_raw).astype(np.float32)

    # tau sub-structure
    tau_np    = ak.to_numpy(ak.flatten(tau_raw[mask], axis=1))  # (N, 5)
    tau1_sel  = tau_np[:, 0].astype(np.float32)
    tau2_sel  = tau_np[:, 1].astype(np.float32)
    tau3_sel  = tau_np[:, 2].astype(np.float32)

    # derived features
    n_total   = nch_sel + nne_sel
    chf_sel   = safe_ratio(nch_sel,  n_total,  fill=0.0)
    nef_sel   = 1.0 - chf_sel
    tau21_sel = safe_ratio(tau2_sel, tau1_sel, fill=1.0)
    tau32_sel = safe_ratio(tau3_sel, tau2_sel, fill=1.0)

    # labels & sample_id
    n_jets    = len(pt_sel)
    labels    = np.full(n_jets, PROCESS_LABEL[process],    dtype=np.int32)
    sample_id = np.full(n_jets, PROCESS_SAMPLEID[process], dtype=np.int32)

    # feature matrix — unchanged from original
    features = np.stack([
        pt_sel, mass_sel, nch_sel, nne_sel,
        ehad_sel, chf_sel, nef_sel,
        tau1_sel, tau21_sel, tau32_sel,
    ], axis=1).astype(np.float32)   # (N, 10)

    elapsed = time.time() - t0
    print(f"    → {n_jets:,} jets | events in file: {n_events_total:,} | "
          f"event_offset: {event_offset:,}  ({elapsed:.1f}s)")

    return {
        "features":       features,
        "labels":         labels,
        "sample_id":      sample_id,
        "jet_pt":         pt_sel,
        "event_id":       event_id,       # FIXED — global, offset-chained
        "jet_index":      jet_index,      # NEW   — position within event
        "n_events_total": n_events_total, # NEW   — for offset chaining
    }


def merge_processes(results_list):
    """Concatenate outputs from multiple process files."""
    merged = {}
    # exclude the scalar helper key
    keys = [k for k in results_list[0] if k != "n_events_total"]
    for key in keys:
        merged[key] = np.concatenate([r[key] for r in results_list], axis=0)
    return merged


def save_npz(data, save_path):
    """Save merged feature dict to compressed npz."""
    np.savez_compressed(
        save_path,
        features      = data["features"],
        labels        = data["labels"],
        sample_id     = data["sample_id"],
        jet_pt        = data["jet_pt"],
        event_id      = data["event_id"],
        jet_index     = data["jet_index"],   # NEW
        feature_names = np.array(FEATURE_NAMES),
    )
    n   = len(data["labels"])
    sig = int(np.sum(data["labels"] == 1))
    bkg = int(np.sum(data["labels"] == 0))
    print(f"  Saved: {os.path.basename(save_path)}")
    print(f"    total={n:,}  signal={sig:,}  background={bkg:,}")


# ══════════════════════════════════════════════════════════════════
# MAIN: TRAINING SETS
# event_offset chains across tau → jj → bb within each energy
# ══════════════════════════════════════════════════════════════════

print("=" * 60)
print("BDT FEATURE EXTRACTION — TRAINING SETS")
print("=" * 60)

for energy in [125, 250]:
    pt_min, pt_max = PT_WINDOWS[energy]
    print(f"\n── Training {energy} GeV  (pT {pt_min}–{pt_max} GeV) ──")

    results      = []
    event_offset = 0

    for process, fname in TRAIN_FILES[energy].items():
        root_path = os.path.join(ROOT_DIR, fname)
        r = extract_jets_from_root(root_path, pt_min, pt_max,
                                   process, energy, event_offset)
        event_offset += r["n_events_total"]   # chain offset
        results.append(r)

    merged    = merge_processes(results)
    save_path = os.path.join(SAVE_DIR, f"bdt_features_{energy}GeV_train.npz")
    save_npz(merged, save_path)


# ══════════════════════════════════════════════════════════════════
# MAIN: TEST SETS
# event_offset resets to 0 per energy (each energy is independent)
# matches CNN pipeline where offset resets per output file
# ══════════════════════════════════════════════════════════════════

print("\n" + "=" * 60)
print("BDT FEATURE EXTRACTION — TEST SETS")
print("=" * 60)

for energy in TEST_ENERGIES:
    pt_min, pt_max = PT_WINDOWS[energy]
    print(f"\n── Test {energy} GeV  (pT {pt_min}–{pt_max} GeV) ──")

    results      = []
    event_offset = 0   # reset per energy

    for process in ["tau", "jj", "bb"]:
        fname     = f"ee_{process}_{energy}GeV_50k_test.root"
        root_path = os.path.join(ROOT_DIR, fname)
        r = extract_jets_from_root(root_path, pt_min, pt_max,
                                   process, energy, event_offset)
        event_offset += r["n_events_total"]   # chain offset
        results.append(r)

    merged    = merge_processes(results)
    save_path = os.path.join(SAVE_DIR, f"bdt_features_{energy}GeV_test.npz")
    save_npz(merged, save_path)

print("\n" + "=" * 60)
print("Features saved to:", SAVE_DIR)
print("Feature order:", FEATURE_NAMES)
print("=" * 60)