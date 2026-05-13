# CareWear Study Data Processing

Pre requisite:
-- Data access
1. Concat_File/
2. merged_lables/

## Visualizer
Run following:

visualizer_dash_raw.py

Plots dasboard for entire dataset Smartwatch (HR), Belt (ECG, Respiration)

```bash
pip install -r requirements.txt
python3 visualizer_dash_raw.py
```

---

## Machine Learning & Deep Learning Pipelines

This repository contains two primary pipelines for stress detection using wearable data: a traditional Machine Learning (ML) benchmarking tool and a Deep Learning (DL) Fusion network.

### 1. Deep Fusion Network with Attention
**File:** `machine_learning/dl_run_stratified_5fold_fusion_attention_auto.py`

This script implements a multimodal Deep Learning architecture that fuses Accelerometer (ACC) and Heart Rate (HR) data using a combination of CNNs, LSTMs, and Temporal Attention.

#### Model Architecture
The `DeepFusionNet` architecture follows a late-fusion approach:

```mermaid
graph TD
    subgraph "Accelerometer Branch (Spatial-Temporal)"
        A1[Raw ACC - 3 channels] --> B1[3x Conv1D + ReLU + MaxPool]
        B1 --> C1[2-Layer LSTM - 128 Hidden]
        C1 --> D1[Temporal Self-Attention]
        D1 --> E1[Context Vector - 128 units]
    end
    
    subgraph "Heart Rate Branch (Physiological)"
        A2[Raw HR - 1 channel] --> B2[Conv1D + ReLU + MaxPool]
        B2 --> C2[1-Layer LSTM - 32 Hidden]
        C2 --> D2[Last Time-Step - 32 units]
    end
    
    E1 --> F[Late Fusion - Concatenation - 160 units]
    D2 --> F
    
    F --> G[Dropout 0.5]
    G --> H[Fully Connected - 64 units]
    H --> I[ReLU]
    I --> J[Output Layer - Softmax - 2 Classes]

    style A1 fill:#f9f,stroke:#333,stroke-width:2px
    style A2 fill:#f9f,stroke:#333,stroke-width:2px
    style J fill:#bfb,stroke:#333,stroke-width:2px
```

#### Evaluation Pipeline
- **Cross-Validation:** 5-Fold Stratified Group K-Fold (ensures no data leakage between participants).
- **Windowing:** Defaults to 60-second sliding windows with 50% overlap.
- **Normalization:** Subject-wise Z-score standardization.
- **Training:** Uses AdamW optimizer, `ReduceLROnPlateau` scheduler, and early stopping.
- **Outputs:** Confusion matrices, gradient flow logs, and probability distribution plots per fold.

---

### 2. Automated ML Benchmarking
**File:** `machine_learning/run_stratified_5fold_auto.py`

This script provides an automated framework to benchmark traditional ML classifiers across multiple scientific feature sets.

#### ML Pipeline Structure
The pipeline uses `imblearn` and `sklearn` to handle class imbalance and hyperparameter optimization.

```mermaid
graph LR
    Input[Feature CSVs] --> Pre[Subject+Activity Standardization]
    Pre --> CV[Stratified Group 5-Fold CV]
    
    subgraph "Per-Fold Training Pipeline"
        CV --> Impute[Median Imputer]
        Impute --> US[Random Under-Sampler]
        US --> Search[RandomizedSearchCV - 20 iter]
        Search --> Models{Classifier Selection}
        Models --> |RF| RF[Random Forest]
        Models --> |XGB| XGB[XGBoost]
        Models --> |GB| GB[Grad Boosting]
        Models --> |LR| LR[Log Reg]
    end
    
    RF & XGB & GB & LR --> Eval[Global Performance Metrics]
    Eval --> Master[Master Summary CSV]

    style Input fill:#f9f,stroke:#333,stroke-width:2px
    style Master fill:#bfb,stroke:#333,stroke-width:2px
```

#### Evaluation Pipeline
- **Feature Sets:** Automatically benchmarks sets including Time Domain Stats, HRV Proxies, ACC Kinematics, and Spectral Band Power.
- **Balancing:** Implements a "mild under-sampling" strategy to handle class imbalance while preserving data.
- **Hyperparameters:** Conducts `RandomizedSearchCV` on a pre-defined grid for each fold.
- **Metrics:** Calculates Accuracy, Balanced Accuracy, F1-Score, Sensitivity, Specificity, and ROC-AUC.
- **Deployment:** After validation, trains a final model on the full dataset and saves it as a `.pkl` file.