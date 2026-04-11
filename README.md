# Tau Jet Classification at a Future Lepton Collider

A deep learning study for hadronic tau jet identification at a future e⁺e⁻ collider (ILC/ILD),
using jet images and graph-based representations. This project is part of an MSc upgrade and
is **actively being developed — results, models, and documentation will be updated regularly**.

---

## Physics motivation

Precise identification of hadronic tau jets is critical for a wide range of physics processes
at future lepton colliders such as the ILC, including Higgs decay measurements (H → τ⁺τ⁻),
electroweak precision tests, and BSM searches. Unlike at hadron colliders, the clean e⁺e⁻
environment allows detailed study of jet substructure and classifier generalization across
center-of-mass energies. This project benchmarks modern deep learning architectures against
this task, using a full simulation pipeline from event generation to detector response.

---

## Pipeline

```
MadGraph5  →  Pythia8  →  HepMC3  →  Delphes 3.5.1 (ILD card)  →  ROOT  →  PyTorch
```

- Jet algorithm: anti-kT, R = 0.4, pT_min = 15 GeV
- Detector card: modified `delphes_card_ILD.tcl` with N-subjettiness enabled
- Jet images: 32×32, 3-channel (EFlowTrack / EFlowPhoton / EFlowNeutralHadron)
- Preprocessing: η-φ centering, PCA rotation, energy flip, L2 normalization

---

## Processes and samples

| Process | Role | Events (125 GeV training) |
|---|---|---|
| e⁺e⁻ → τ⁺τ⁻ | Signal | 175k |
| e⁺e⁻ → jj | Background | 80k |
| e⁺e⁻ → bb̄ | Background | 60k |

Test sets: 50k events per process, evaluated at **6 center-of-mass energies**:
100, 125, 150, 200, 250, and 300 GeV.

---

## Models implemented

| Model | Architecture | Params | Train √s | Val AUC |
|---|---|---|---|---|
| JetCNN (model1) | 4-layer CNN + FC head | 766k | 125 GeV | 0.9961 |
| JetCNN (model2) | 4-layer CNN + FC head | 766k | 250 GeV | 0.9988 |
| JetViT (model1) | Vision Transformer (depth=4) | 545k | 125 GeV | 0.9969 |
| JetViT (model2) | Vision Transformer (depth=4) | 545k | 250 GeV | 0.9988 |

### Cross-energy evaluation summary (overall AUC)

| Test energy | CNN@125 | CNN@250 | ViT@125 | ViT@250 |
|---|---|---|---|---|
| 100 GeV | 0.9944 | 0.9938 | 0.9957 | 0.9937 |
| 125 GeV | 0.9964 | 0.9962 | 0.9971 | 0.9963 |
| 150 GeV | 0.9975 | 0.9975 | 0.9978 | 0.9976 |
| 200 GeV | 0.9983 | 0.9986 | 0.9984 | 0.9986 |
| 250 GeV | 0.9986 | 0.9989 | 0.9986 | 0.9989 |
| 300 GeV | 0.9988 | 0.9992 | 0.9987 | 0.9991 |

Key observation: all models improve with test energy — higher √s produces more
collimated tau jets with more visually distinct 3-prong substructure, making the
classification task intrinsically easier. The CNN@125 model is the highest-precision
classifier; the ViT@125 model achieves the highest recall and event-tagging efficiency,
particularly at lower energies out-of-distribution.

---

## Repository structure

```
tau-jet-classification/
├── generation/
│   ├── pythia_scripts/       ← Pythia8 config files for each process
│   └── delphes_card_ILD.tcl  ← modified ILD detector card
├── utils/
│   ├── dataset.py            ← JetDataset, event-wise split
│   └── modelarch.py          ← JetCNN, JetViT definitions
├── notebooks/
│   ├── cnn/                  ← CNN training and evaluation notebooks
│   ├── vit/                  ← ViT training and evaluation notebooks
│   ├── bdt/                  ← (planned)
│   └── gnn/                  ← (planned)
├── docs/                     ← notes and architecture diagrams
├── .gitignore
├── requirements.txt
├── LICENSE
└── README.md
```

> Large files (datasets `.npz`, ROOT files, model checkpoints `.pt`, plots) are excluded
> from the repository via `.gitignore`. Model weights are available on request.

---

## Roadmap

- [x] Event generation pipeline (MadGraph5 + Pythia8 + Delphes)
- [x] Jet image dataset construction with preprocessing
- [x] JetCNN — trained at 125 GeV and 250 GeV
- [x] JetViT — trained at 125 GeV and 250 GeV
- [x] Cross-energy evaluation (100–300 GeV) for CNN and ViT
- [ ] BDT baseline using tabular jet features
- [ ] Graph Neural Network (GNN) using particle-level inputs
- [ ] Late-fusion ensemble (CNN + ViT + BDT + GNN)
- [ ] Full results writeup and comparison

---

## Requirements

```
python >= 3.9
torch >= 2.0
numpy
uproot
awkward
scikit-learn
matplotlib
scipy
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Citation / acknowledgement

If you use any part of this project, please cite this repository and acknowledge
the use of MadGraph5, Pythia8, and Delphes with the ILD detector card.

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.

---

*This repository is part of an ongoing MSc project. Expect regular updates.*
