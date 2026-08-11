## Spiking Neural Network Experiments on BACH and MHIST

This directory contains the PyTorch/SNNTorch implementations used to evaluate an **anytime Spiking Neural Network (SNN)** on two publicly available breast histopathology datasets: **BACH** and **MHIST**.

The experiments investigate temporal behavior in a residual SNN architecture, including the effect of the LIF membrane parameter ( \beta ), its learnability, and the number of simulation timesteps (T). The same `SpikingResNetLite` architecture and 160×160 input configuration are used across the datasets to support cross-dataset comparison.

### Files

| File           | Dataset | Purpose                                                           |
| -------------- | ------- | ----------------------------------------------------------------- |
| `SNN-BACH.py`  | BACH    | Trains and evaluates the SNN on the BACH breast histology dataset |
| `SNN-MHIST.py` | MHIST   | Trains and evaluates the SNN on the MHIST dataset                 |

---

## Model Architecture

Both implementations use a lightweight residual spiking architecture, `SpikingResNetLite`.

The network consists of:

* A convolutional stem with Batch Normalization and a LIF neuron
* Four residual spiking stages
* Two LIF neurons within each residual block
* Residual/shortcut connections
* Global Average Pooling
* Dropout (`p=0.5`)
* A single-output linear classification head

The network maintains membrane states across timesteps and produces a classification logit at every timestep. Cumulative logits are computed using a running average across timesteps, enabling **anytime prediction** at intermediate computational budgets.

### Temporal Prediction

For a simulation with (T) timesteps, the model produces:

1. Per-timestep logits
2. Cumulative logits
3. Intermediate feature representations

The cumulative prediction at timestep (t) is calculated from the average of the logits observed up to that timestep.

This allows the model to be evaluated at different temporal budgets rather than relying exclusively on the final timestep.

---

## Experimental Configurations

Three experimental configurations are implemented for the main BACH experiments:

### `baseline_learn`

Uses a uniform membrane parameter:

```text
β = 0.95
```

with learnable LIF parameters:

```text
learn_beta = True
```

This configuration evaluates whether allowing the membrane parameter to adapt during training affects temporal representation and performance.

### `baseline_frozen`

Uses the same uniform:

```text
β = 0.95
```

but keeps the membrane parameters fixed:

```text
learn_beta = False
```

This isolates the effect of learning the membrane parameters from the effect of the underlying β schedule.

### `matches`

Uses a heterogeneous, layer-wise β schedule with frozen membrane parameters:

```text
stem    = 1.00

stage 1 = 0.90, 0.98
stage 2 = 0.93, 0.97
stage 3 = 0.92, 0.96
stage 4 = 0.90, 0.94
```

This configuration tests a manually specified heterogeneous temporal schedule.

---

## Datasets

### BACH

The BACH experiments use the binary **benign vs malignant** classification setup.

The implementation uses a predefined 60/20/20 train/validation/test split:

| Split      | Total | Benign | Malignant |
| ---------- | ----: | -----: | --------: |
| Train      |   240 |    120 |       120 |
| Validation |    80 |     40 |        40 |
| Test       |    80 |     40 |        40 |

The code explicitly verifies both the class balance and the absence of overlap between the three splits.

### MHIST

The MHIST implementation uses the **HP** and **SSA** classes:

* HP — Hyperplastic Polyp
* SSA — Sessile Serrated Adenoma

The verified split sizes are:

| Split      | Total |    HP | SSA |
| ---------- | ----: | ----: | --: |
| Train      | 1,740 | 1,236 | 504 |
| Validation |   435 |   309 | 126 |
| Test       |   977 |   617 | 360 |

The implementation also verifies that there is no overlap between training, validation, and test samples.

---

## Data Preprocessing

Images are resized to **160×160** pixels.

Training augmentation includes:

* Random cropping
* Random horizontal flipping
* Random vertical flipping
* Random rotation
* Color jitter
* Random grayscale conversion
* Random erasing
* ImageNet normalization

Validation and test images are resized directly to 160×160 and normalized using ImageNet statistics.

The MHIST implementation uses the same 160×160 preprocessing configuration for cross-dataset comparability.

---

## Training Configuration

The common training configuration is:

| Parameter                 |            Value |
| ------------------------- | ---------------: |
| Image size                |        160 × 160 |
| Batch size                |               24 |
| Maximum epochs            |              120 |
| Optimizer                 |            AdamW |
| Learning rate             |         1 × 10⁻³ |
| Weight decay              |         5 × 10⁻⁴ |
| Gradient clipping         |              1.0 |
| LR scheduler              | Cosine Annealing |
| Early stopping patience   |        75 epochs |
| Surrogate gradient        |       Arctangent |
| Default simulation length |     16 timesteps |

The BACH implementation uses the same core configuration.

For MHIST, class imbalance is additionally handled through a positive-class weight in the binary cross-entropy loss.

---

## Anytime Training Objective

The model is trained using an **anytime loss** that evaluates the cumulative prediction at every timestep.

Later timesteps receive greater weight than earlier timesteps. For a simulation length (T), the timestep weights are proportional to:

[
T+1,;T+2,;\ldots,;2T
]

and are normalized to sum to one.

## The loss is therefore a weighted combination of the binary classification losses across all available timesteps.

## Experimental Seeds and Timesteps

The BACH script evaluates multiple random seeds and temporal configurations, including:

```text
Seeds:
42
123
2024

Timesteps:
T = 16
T = 4
```

The main BACH run matrix includes `baseline_learn`, `matches`, and `baseline_frozen`, with the additional `T=4` run used to examine the effect of a shorter temporal budget.

The MHIST implementation evaluates `baseline_learn` and `matches` using the three seeds:

```text
42
123
2024
```

with:

```text
T = 16
```

---

## Evaluation and Temporal Analysis

In addition to test-set ROC-AUC, the scripts collect several temporal diagnostics.

### Test ROC-AUC

Classification performance is evaluated at intermediate and final timesteps:

```text
t = 1
t = 4
t = 8
t = 12
t = 16
```

when those timesteps are available.

### Feature Probing

The learned feature representations at different timesteps are evaluated using a 5-fold stratified logistic-regression probe. This provides an additional measure of how much class-discriminative information is present in the intermediate representations.

### Temporal Representation Similarity

The scripts calculate consecutive and pairwise cosine similarity between class-averaged feature representations across timesteps.

For BACH, this is calculated separately for:

* Benign
* Malignant

For MHIST, it is calculated separately for:

* HP
* SSA

These measurements are used to characterize how representations evolve over time.

### Spike Activity and Synaptic Operations

The implementations also measure:

* Per-layer spike rates
* Static convolutional MACs
* Per-convolution SynOps
* Cumulative SynOps
* Layer-wise cumulative SynOps

This provides a temporal estimate of computational activity as the simulation proceeds.

---

## Saved Outputs

Each experimental run creates a separate output directory containing the trained model and analysis artifacts.

Typical outputs include:

```text
best_val_state.pt
sidecar.json
history.csv
beta_history.csv
analysis.npz
```

The files contain:

* `best_val_state.pt` — model parameters selected using validation performance
* `sidecar.json` — experiment configuration and metadata
* `history.csv` — training loss and validation AUC history
* `beta_history.csv` — evolution of LIF β parameters
* `analysis.npz` — temporal features, probe results, cosine similarities, test AUCs, bootstrap confidence intervals, spike rates, MACs, SynOps, and final predictions

The BACH implementation explicitly saves these analysis artifacts after each completed run. The MHIST implementation follows the same artifact structure.

---

## Reproducibility

The experiments explicitly seed Python, NumPy, PyTorch, and CUDA random-number generators. Deterministic cuDNN behavior is also enabled:

```python
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
```

## Each run records its experiment name, seed, timestep budget, β configuration, dataset, and training hyperparameters in `sidecar.json`.

## Data Availability

All datasets analysed in this study are publicly available:

* **BACH:** https://www.kaggle.com/datasets/truthisneverlinear/bach-breast-cancer-histology-images
* **MHIST:** https://bmirds.github.io/MHIST/

The datasets are obtained from their respective original sources cited in the manuscript. **No new data were generated in this study.**

---

## Requirements

The main dependencies are:

```text
Python
PyTorch
Torchvision
SNNTorch
NumPy
Pandas
scikit-learn
Pillow
Matplotlib
```

The scripts install `snntorch` automatically when executed in environments such as Kaggle notebooks.

## Usage

Before running the scripts, configure the dataset path according to the local environment.

For example:

```bash
python SNN-BACH.py
```

and:

```bash
python SNN-MHIST.py
```

The scripts automatically execute the configurations specified in their respective `RUN_MATRIX` definitions and save the resulting checkpoints and analysis artifacts under the working directory.

> **Note:** The dataset paths currently present in the scripts are environment-specific Kaggle paths. Update `DATA_ROOT` before running the code outside the original Kaggle environment.
