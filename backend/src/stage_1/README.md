# StepWise Stage 1: Clinical Assessments, Questionnaires & Progression Prioritizer

## Executive Summary
Stage 1 is the **zero-cost, 100% non-invasive** entry point for every patient visiting a Primary Health Centre (PHC), memory clinic, or outpatient department.

Its clinical mandate is **NOT to definitively diagnose Alzheimer's Disease**, but to accurately compute a **Prioritization Risk Score** that determines whether the patient must be escalated to **Stage 2 (Plasma p-tau217 Blood Biomarkers)** or safely managed within the **Routine Primary Care Loop**. 

---

## 1. Approach & Methodology
Our architecture relies on a **Unified XGBoost Triage Engine** that processes multi-stream clinical data. We discarded legacy unsupervised GMM methods in favor of a clean, robust supervised approach.

### Key Innovations:
1. **Flexible Doctor Workflow & Native Missing-Value Routing:** In real-world clinics, doctors administer only one screening test (e.g., MoCA *or* MMSE). Our XGBoost Engine natively handles `NaN` inputs via default branch split routing. If a doctor inputs MoCA only, XGBoost automatically routes the patient through MoCA decision paths without crashing or needing artificial imputation.
2. **Domain Harmonization:** Universal domain ratios (`Domain_Delayed_Recall_Ratio`, `Domain_Orientation_Ratio`, `Domain_Executive_Ratio`) mathematically translate any administered test into universal 0.0–1.0 domain pillars.
3. **Longitudinal Velocities ($\Delta$):** We compute the differential rate of cognitive decline per year (e.g., $\Delta\text{MoCA}/\text{yr}$, $\Delta\text{CDRSB}/\text{yr}$). These are the strongest drivers for distinguishing progressive MCI from stable status.
4. **Leakage-Free Validation:** We use 5-Fold `StratifiedGroupKFold` grouped on Patient ID (`RID`) to prevent data leakage between visits of the same patient.

---

## 2. Empirical Model Validation & Matrices

Our Stage 1 Unified Engine delivers exceptional performance, maximizing triage efficiency.

### A. 24-Month Clinical Progression Predictor (Risk Model)
* **Validation:** 5-Fold `StratifiedGroupKFold` on Patient ID (`RID`).
* **Mean CV ROC-AUC:** **$0.7628 \pm 0.0207$**
* **Cohort:** 11,641 prospectively followed patient visits.
* **Top-K Clinical Efficiency:** Reviewing the top 20% of the risk queue captures over **$51\%$ of all clinical progressors**, vastly optimizing clinical resource allocation.

### B. 3-Class Triage Classifier (Healthy CN / MCI / Dementia)
* **Overall Accuracy:** **$85.0\%$** across 16,382 patient visits.
* **MCI Detection:** Precision **$86\%$** | Recall **$76\%$** | F1-Score **$0.81$** (Significantly boosted by MoCA integration).
* **Healthy (CN) Detection:** Precision **$88\%$** | Recall **$92\%$** | F1-Score **$0.90$**.
* **Dementia Detection:** Precision **$80\%$** | Recall **$88\%$** | F1-Score **$0.84$**.

---

## 3. The Multi-Stream Clinical Dataset Architecture

Stage 1 harmonizes raw clinical data across **16,382 visits** and **3,700 unique patients** in the ADNI cohort. Below are the precise datasets and columns utilized:

### Dataset 1: MMSE (Mini-Mental State Exam)
* **Columns:** `MMSCORE` (Total 0-30), `MMORIENTATION` (Date/place), `MMRECALL` (3-word delay), `MMATTENTION` (Serial 7s), `MMREGISTRATION`, `MMLANGUAGE`.
* **Rationale:** Classic bedside screening tool. `MMRECALL` is highly specific for AD.

### Dataset 2: MoCA (Montreal Cognitive Assessment)
* **Columns:** Reconstructed Total (0-30), Visuospatial/Executive, Trails, Clock, 5-Word Delayed Recall, Orientation.
* **Rationale:** Gold-standard for early MCI detection. Fixes the MMSE ceiling effect.

### Dataset 3: CDR & CDR-SB (Clinical Dementia Rating)
* **Columns:** `CDGLOBAL` (Global staging 0-3), `CDSOB` (Sum of Boxes 0-18), `CDMEMORY`, `CDORIENT`, `CDJUDGE`, `CDSOCIAL`, `CDHOME`, `CDCARE`.
* **Rationale:** Clinical staging anchor. Sum of Boxes provides granular tracking of progression. 

### Dataset 4: ADAS-COG
* **Columns:** `TOTSCORE` (Total errors), `WORDRECALL`, `WORDRECOG`, `ORIENT`.
* **Rationale:** Catches early MCI that MMSE misses; extremely validated in clinical trials.

### Dataset 5: FAQ (Functional Assessment Questionnaire)
* **Columns:** `FAQTOTAL` (0-30), `FAQFINAN` (Finances), `FAQTRAVE`, `FAQMEAL`, `FAQNEWS`, `FAQREM` (Appointments).
* **Rationale:** Informant-reported loss of daily autonomy. `FAQFINAN` is often the first functional skill to fail.

### Dataset 6: NPI-Q (Neuropsychiatric Inventory)
* **Columns:** `NPISCORE` (Total), `NPID` (Depression), `NPIF` (Apathy), `NPIB` (Agitation), `NPIE` (Anxiety).
* **Rationale:** Apathy and depression precede cognitive decline by 3–5 years, providing our earliest possible signal.

### Dataset 7: Demographics & Vitals
* **Columns:** `AGE`, `PTGENDER`, `PTEDUCAT` (Education years), `PTETHCAT`/`PTRACCAT` (Ancestry).
* **Columns (Vitals):** `VSBPSYS` (Systolic BP), `VSBPDIA` (Diastolic BP), `VSBMI`, plus engineered `Pulse_Pressure` (vascular brain damage risk).

### Dataset 8: Medical History & Medications
* **Columns (History):** `MHDIAB` (Diabetes), `MHHYPTEN` (Hypertension), `MHCVD`, `MHSTROKE`, `MHSMOK`, `MHFAMAD` (Family History), `MHSINUS`.
* **Columns (Meds):** Engineered flags `On_AntiDiabetic`, `On_Antihypertensive`, `On_Anticholinergic` (confounds cognition), `On_Antidepressant`.

### Dataset 9: Longitudinal Velocities (Derived)
* **Columns:** $\Delta\text{MoCA}/\text{yr}$, $\Delta\text{MMSE}/\text{yr}$, $\Delta\text{CDRSB}/\text{yr}$, $\Delta\text{FAQ}/\text{yr}$.
* **Rationale:** Differential rate of cognitive decline.

### Dataset 10: Ground Truth Targets
* **Columns:** `Target_Progression24m` (24-Month Future Progression flag) + `DX_Label` / `DIAGNOSIS` (CN / MCI / Dementia).
* **Rationale:** XGBoost learning targets for triage and clinical conversion milestones.

---

## 4. Stage 1 File Structure

```
backend/src/stage_1/
├── 01_dataset_harmonizer.py           # Reconstructs MoCA rows, engineers velocities & harmonizes features
├── 02_train_xgboost_triage_engine.py  # Trains calibrated progression & triage XGBoost models (5-Fold CV)
├── 03_sandbox_clinical_simulator.py   # Simulates real-world clinical patient cases
├── README.md                          # Stage 1 documentation & clinical rationale
├── cleaned_dataset/
│   ├── stage1_harmonized_dataset.csv
│   └── stage1_harmonized_dataset.parquet
└── visualizations/
    ├── stage1_progression_24m_evaluation_curves.png
    └── stage1_progression_24m_feature_importance.png
```

---

## 5. Output Gating & Next Stage Escalation

$$\text{Prioritization Risk Score } R_1 \in [0.0, 1.0]$$

1. **High / Moderate Risk ($R_1 \ge 0.40$ or Rapid Declinor $\Delta\text{MoCA} \le -2.0/\text{yr}$):**
   * **Action:** Generate 1-Click HL7 FHIR `ServiceRequest` order for **Stage 2 (Plasma p-tau217 + $A\beta_{42/40}$ Phlebotomy)**.
2. **Low Risk ($R_1 < 0.40$ & Stable Velocity):**
   * **Action:** Direct to **Primary Care Loop** (Screen for sleep apnea, B12, thyroid, depression; schedule 12-month re-check).
