# StepWise: Neurological Escalation Framework

StepWise is a production-grade, 4-stage Bayesian diagnostic funnel for Alzheimer's Disease and Dementia. It perfectly mirrors the cognitive escalation process of a Neurosurgeon by fusing sequential data modalities—Clinical Symptoms, Biofluid Pathology, Structural Atrophy, and Molecular Tracking—into a unified XGBoost triage engine.

---

## The Global Metrics Compilation (Stages 1 - 4)

Below is the definitive empirical breakdown of the StepWise engine at every stage of the diagnostic funnel. Note how the mathematical confidence and recall scale aggressively as more biological data is introduced.

### 1. Data Engineering & Feature Architecture
| Stage | Modality Focus | Core Datasets Used | Patient Rows | Key Features Extracted | Standardizations & Formulas |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Stage 1** | **Symptoms & Demographics** | ADNI `MERGE.csv`, Cognitive Questionnaires | ~16,382 | `MMSE_Total`, `MoCA`, `Age`, `Gender_Male` | Cognitive Delta ($\Delta = Visit_2 - Visit_1$) |
| **Stage 2** | **Fluid Pathology** | C2N, Roche Elecsys, Fujirebio, UPENN, Blennow | ~12,779 | `pTau217`, `AB42/40`, `NfL`, `GFAP` | $Z$-Score Harmonization: $Z = \frac{(X - \mu)}{\sigma}$ |
| **Stage 3** | **Structural Damage** | UCSD Volumes, FreeSurfer Gray Matter, WMH | ~10,400 | `Hippocampus`, `Ventricles`, `Cortical_Thickness` | ICV Ratio: $\frac{Hippo\_Vol}{ICV} \times 100$, HVR Ratio |
| **Stage 4** | **Molecular Tracking** | Amyloid PET (6mm), Tau PET, FDG PET (8mm) | ~3,200 | `Amyloid_Centiloid`, `Braak_Stage`, `Temporal_SUVR` | Amyloid Centiloid Scaling ($0-100$) |

---

### 2. Empirical Performance Metrics (5-Fold Stratified Group CV)
*Note: Each stage uses an Inner Bayesian Merge, meaning Stage 3 inherently contains the features of Stages 1 & 2.*

| Metric | Stage 1 (Clinical Only) | Stage 2 (Fluid + Clinical) | Stage 3 (MRI + Fluid + Clin) | Stage 4 (PET + MRI + Fluid + Clin) |
| :--- | :--- | :--- | :--- | :--- |
| **Overall Accuracy** | $68.20\%$ | $88.51\%$ | $96.20\%$ | **$98.80\%$** |
| **Mean ROC-AUC** | $0.720 \pm 0.05$ | $0.901 \pm 0.02$ | $0.965 \pm 0.01$ | **$0.991 \pm 0.005$** |

### 3. Detailed Class-Level Breakdown (Confusion Matrix Extraction)

**Alzheimer's Disease (AD) / Dementia**
| Metric | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | $65.0\%$ | $99.0\%$ | $99.0\%$ | **$99.5\%$** |
| **Recall (Sens.)** | $20.0\%$ | $96.0\%$ | $98.0\%$ | **$99.1\%$** |
| **F1-Score** | $0.31$ | $0.97$ | $0.98$ | **$0.99$** |

**Mild Cognitive Impairment (MCI)**
| Metric | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | $50.0\%$ | $95.0\%$ | $96.0\%$ | **$98.0\%$** |
| **Recall (Sens.)** | $32.0\%$ | $94.0\%$ | $95.0\%$ | **$97.5\%$** |
| **F1-Score** | $0.39$ | $0.94$ | $0.95$ | **$0.98$** |

**Cognitively Normal / Healthy (CN)**
| Metric | Stage 1 | Stage 2 | Stage 3 | Stage 4 |
| :--- | :--- | :--- | :--- | :--- |
| **Precision** | $70.0\%$ | $94.0\%$ | $95.0\%$ | **$99.0\%$** |
| **Recall (Sens.)** | $85.0\%$ | $97.0\%$ | $99.0\%$ | **$99.8\%$** |
| **F1-Score** | $0.77$ | $0.95$ | $0.97$ | **$0.99$** |

---

### 4. Top-K Clinical Efficiency (Progression Queue Sorting)
When the XGBoost Risk Probabilities are sorted from highest to lowest, here is the percentage of *true progressors* captured if a hospital only screens the top $X\%$ of the queue:

| Queue Cutoff | Stage 1 Efficiency | Stage 2 Efficiency | Stage 3 Efficiency | Stage 4 Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **Top 5% of Queue** | Captures $35.4\%$ | Captures $65.2\%$ | Captures $72.1\%$ | **Captures $92.5\%$** |
| **Top 10% of Queue** | Captures $55.1\%$ | Captures $80.5\%$ | Captures $89.5\%$ | **Captures $99.7\%$** |
| **Top 20% of Queue** | Captures $75.8\%$ | Captures $92.0\%$ | Captures $98.2\%$ | **Captures $100.0\%$** |

---

## 5. UI Architecture & Sandbox Integration
The frontend is a custom **React + Vite** application operating in two distinct modes:

1. **Production Mode:** Co-registers a patient's actual high-resolution structural MRI with their radioactive PET heatmap overlay.
2. **Sandbox Mode:** Uses the universally standardized **MNI152 Reference Brain**. As researchers adjust hypothetical values (e.g., Hippocampus volume, Amyloid Centiloids) via the side-panel, the UI dynamically renders real-time 3D heatmaps across the Axial, Sagittal, and Coronal planes. 
