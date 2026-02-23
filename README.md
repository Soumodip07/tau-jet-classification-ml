# Identifying Hadronically Decaying Tau Leptons Using Machine Learning

This repository presents the complete workflow for classifying hadronically decaying tau jets versus light quark jets in high-energy physics using **jet image-based representation** and **Convolutional Neural Networks (CNNs)**. The analysis leverages simulated $e^+e^-$ collision events and evaluates performance across multiple center-of-mass energies.

---

## 📘 Project Overview

**Tau lepton identification** is crucial for both **precision Standard Model (SM) measurements** and **searches for physics beyond the Standard Model (BSM)**.

A central motivation stems from **Higgs boson analyses**:
- The Higgs decays to tau lepton pairs ($H \to \tau^+ \tau^-$) with a branching ratio of ~**6.3%**, making it a key channel for probing **Yukawa couplings**.
- Hadronic tau decays, constituting ~**65%** of all tau decays, are challenging to isolate from quark-induced jets.
- Effective tau tagging is critical for **$H \to \tau\tau$** reconstruction, as well as identifying **new physics signatures** involving tau leptons.

This project investigates CNN-based image classification to distinguish **tau jets** from **light quark jets**, using simulated data from:
- $e^+e^- \to \tau^+ \tau^-$ (tau jets)
- $e^+e^- \to jj$ (light quark jets, $j = u,d,s,c,\bar{u},\bar{d},\bar{s},\bar{c}$)

---

## 🎯 Project Goals

- Construct jet images in the $(\eta, \phi)$ plane from final-state particles.
- Apply a full preprocessing pipeline: centering, rotating, and normalizing.
- Train CNN models to classify tau vs. quark jets.
- Evaluate model generalization across various center-of-mass energies.

---

## Dataset Details

- **Event generation**:
  - **MadGraph5_aMC@NLO** for parton-level events
  - **Pythia8** for hadronization and decay
  - **FastJet** with anti-$k_T$ algorithm (R = 0.4) for jet clustering

- **Jet images**:
  - Format: $32 \times 32$ grayscale images
  - Plane: $(\eta, \phi)$
  - **Labels**:  
    - Tau jet → `1`  
    - Quark jet → `0`

- **Energy configurations**:
  - Center-of-mass energies: **100, 200, 300 GeV**
  - Each dataset contains ~220k–240k jets

- **Core jet images**:
  - Also generated with **R = 0.15** to capture compact energy deposits

---

## 🧠 CNN Architecture

Developed using **TensorFlow / Keras**:

- **Convolutional Backbone**:
  - 4 blocks: `Conv2D → BatchNorm → MaxPooling`
- **Classifier Head**:
  - `Flatten → Dense(128, ReLU) → Dropout(0.5) → Dense(1, Sigmoid)`

- **Training Config**:
  - Loss: `BinaryCrossentropy`
  - Metrics: `Accuracy`, `AUC`, `Precision`, `Recall`
  - Optimizer: `Adam` with exponential learning rate decay
  - Class imbalance handled using `class_weight`

---

## 📊 Results Summary

- **Model A** (Trained @ 100 GeV):  
  - High accuracy at 100 GeV, but limited generalization at higher energies.

- **Model B** (Trained @ 200 GeV):  
  - Strong performance across multiple test sets with high AUC.

- **Model C** (Trained @ 300 GeV):  
  - Best overall classifier with AUC up to **99.78%** and consistent generalization.

- **Evaluation Range**:  
  - Test energies: 100, 150, 200, 250, 300, 350, 400 GeV  
  - Performance includes: score distributions, ROC curves, tagging/mistagging vs $p_T$

---

## 📂 Repository Structure

```text
├── LHE_Files/           # MadGraph-generated parton-level events
├── Tau_Pipeline/        # Tau jet simulation and image generation
├── JJ_Pipeline/         # Quark jet simulation and image processing
├── NPY_Files/           # Saved jet image arrays and labels
├── CNN_Model/           # CNN architecture, training, and evaluation scripts
├── Results/             # Plots, metrics, and final performance evaluations
