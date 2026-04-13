# Event Generation Pipeline

Full simulation chain: **MadGraph5 → Pythia8 → Delphes 3.5.1**

---

## Step 1 — Hard process with MadGraph5

Three processes were generated at each centre-of-mass energy:

```
# Signal
generate e+ e- > ta+ ta-

# Background 1
generate e+ e- > j j

# Background 2
generate e+ e- > b b~
```

Key run card settings used across all samples:

| Parameter | Value | Notes |
|---|---|---|
| `ebeam1`, `ebeam2` | √s / 2 | half beam energy per side |
| `maxjetflavor` | 4 | up to charm in j j sample |
| `bwcutoff` | 15 | Breit-Wigner cutoff |
| `use_syst` | False | no systematic variations |
| Random seed | varied per energy and train/test | ensures no overlap between sets |

Output: `.lhe` (Les Houches Event) files saved to `LHE_files/`.

---

## Step 2 — Parton shower and hadronisation with Pythia8

Each `.lhe` file is passed through a Pythia8 C++ script for showering,
hadronisation, and tau decay handling. Scripts are in `pythia_scripts/`:

| Script | Process |
|---|---|
| `tau.cc` | e⁺e⁻ → τ⁺τ⁻ (hadronic decays only) |
| `jj.cc` | e⁺e⁻ → jj |
| `bb.cc` | e⁺e⁻ → bb̄ |

Special Pythia configuration for the tau sample:

```cpp
// Force hadronic tau decays only (mode 2)
pythia.readString("TauDecays:mode = 2");
```

A safety filter is applied in the event loop to remove any residual leptonic
tau decay products (electrons/muons) that escape the mode-2 setting.

Output: `.hepmc` files saved to `hepmc_files/`.

---

## Step 3 — Detector simulation with Delphes 3.5.1

HepMC files are processed through Delphes using a modified ILD detector card:

```
DelphesHepMC2 delphes_card_ILD.tcl output.root input.hepmc
```

Key Delphes card settings:

| Parameter | Value |
|---|---|
| Jet algorithm | anti-kT |
| Jet radius R | 0.4 |
| `JetPTMin` | 15 GeV |
| `ComputeNsubjettiness` | 1 (enabled) |
| `Beta` | 1.0 |
| `AxisMode` | 1 |
| `ComputeTrimming` | enabled |
| `ComputePruning` | enabled |
| `ComputeSoftDrop` | enabled |

Output: `.root` files saved to `root_files/`.

---

## Generated samples summary

### Training sets

| Process | Label | √s | Events generated | ROOT file |
|---|---|---|---|---|
| e⁺e⁻ → τ⁺τ⁻ | Signal | 125 GeV | 175k | `ee_tau_125GeV_175k_train.root` |
| e⁺e⁻ → jj | Background | 125 GeV | 80k | `ee_jj_125GeV_80k_train.root` |
| e⁺e⁻ → bb̄ | Background | 125 GeV | 60k | `ee_bb_125GeV_60k_train.root` |
| e⁺e⁻ → τ⁺τ⁻ | Signal | 250 GeV | 175k | `ee_tau_250GeV_175k_train.root` |
| e⁺e⁻ → jj | Background | 250 GeV | 80k | `ee_jj_250GeV_80k_train.root` |
| e⁺e⁻ → bb̄ | Background | 250 GeV | 60k | `ee_bb_250GeV_60k_train.root` |

### Test sets

50k events per process at each energy, generated with **different random seeds**
from the training sets to ensure no event overlap.

| √s (GeV) | Processes | pT window |
|---|---|---|
| 100 | τ⁺τ⁻, jj, bb̄ | 15 – ~50 GeV |
| 125 | τ⁺τ⁻, jj, bb̄ | 15 – 60 GeV |
| 150 | τ⁺τ⁻, jj, bb̄ | 15 – ~75 GeV |
| 200 | τ⁺τ⁻, jj, bb̄ | 15 – ~100 GeV |
| 250 | τ⁺τ⁻, jj, bb̄ | 15 – 125 GeV |
| 300 | τ⁺τ⁻, jj, bb̄ | 15 – ~145 GeV |

> Raw files (`.lhe`, `.hepmc`, `.root`) are not stored in this repository
> due to size. The Pythia scripts and Delphes card needed to reproduce them
> are provided here.
