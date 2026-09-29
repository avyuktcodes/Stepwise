"""
==============================================================================
STEPWISE STAGE 1: CLINICAL SANDBOX & TRIAGE INFERENCE SIMULATOR
File: 03_sandbox_clinical_simulator.py
Purpose: Simulates the Doctor's Sandbox Interface:
         1. Single-Test Selection: Doctor chooses administered test (MoCA / MMSE / CDR)
            while unadministered tests default to Not Applicable (N/A).
         2. Automatic 6-Domain Cognitive Harmonization.
         3. FAQ Functional Questions (Yes / No / Unknown) & Medical History.
         4. Real-time inference through calibrated Stage 1 XGBoost engines.
==============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np

class StepWiseStage1Sandbox:
    def __init__(self):
        models_dir = "backend/models"
        self.prog_path = os.path.join(models_dir, "stage1_progression_xgb.joblib")
        self.triage_path = os.path.join(models_dir, "stage1_triage_3class_xgb.joblib")
        
        if not os.path.exists(self.prog_path) or not os.path.exists(self.triage_path):
            raise FileNotFoundError("Trained models not found. Please run 02_train_xgboost_triage_engine.py first.")
            
        self.prog_payload = joblib.load(self.prog_path)
        self.triage_payload = joblib.load(self.triage_path)
        
        self.prog_model = self.prog_payload['model']
        self.triage_model = self.triage_payload['model']
        self.feature_cols = self.prog_payload['feature_names']

    def evaluate_patient(self, intake_dict):
        """
        Receives raw clinician input and maps into 6 cognitive domains and XGBoost feature vector.
        """
        # Extract Demographics & Vitals
        age = intake_dict.get('age', 72.0)
        education = intake_dict.get('education_years', 14.0)
        gender_male = 1 if intake_dict.get('gender', 'M').upper() == 'M' else 0
        sbp = intake_dict.get('systolic_bp', 135.0)
        dbp = intake_dict.get('diastolic_bp', 82.0)
        pp = sbp - dbp
        bmi = intake_dict.get('bmi', 25.5)

        # Selected Test Type
        test_selected = intake_dict.get('selected_test', 'MOCA').upper() # 'MOCA', 'MMSE', 'CDR', or 'ALL'
        moca_data = intake_dict.get('moca', {})
        mmse_data = intake_dict.get('mmse', {})
        cdr_data = intake_dict.get('cdr', {})
        faq_data = intake_dict.get('faq', {}) # Yes/No/Unknown for 10 functional daily activities
        npi_data = intake_dict.get('npiq', {})
        history_data = intake_dict.get('medical_history', {})
        longitudinal_data = intake_dict.get('longitudinal', {})

        # Compute FAQ Total from Yes/No/Unknown answers
        faq_score = 0
        for k, v in faq_data.items():
            if str(v).lower() == 'yes':
                faq_score += 3 # Dependent / Requires Assistance
            elif str(v).lower() == 'no':
                faq_score += 0 # Fully Independent
            elif str(v).lower() == 'unknown':
                faq_score += 1 # Borderline / Suspected
        faq_total = faq_score if len(faq_data) > 0 else np.nan

        # Harmonize into 6 Universal Clinical Domains
        # Domain 1: Orientation
        d1_orient = np.nan
        if test_selected in ['MOCA', 'ALL'] and 'orientation' in moca_data:
            d1_orient = moca_data['orientation'] / 6.0
        elif test_selected in ['MMSE', 'ALL'] and 'orientation' in mmse_data:
            d1_orient = mmse_data['orientation'] / 10.0

        # Domain 2: Registration
        d2_reg = 1.0 if test_selected == 'MOCA' else (mmse_data.get('registration', 3.0)/3.0 if test_selected == 'MMSE' else 1.0)

        # Domain 3: Attention & Calculation
        d3_att = np.nan
        if test_selected in ['MOCA', 'ALL'] and 'visuospatial_executive' in moca_data:
            d3_att = moca_data['visuospatial_executive'] / 5.0
        elif test_selected in ['MMSE', 'ALL'] and 'attention' in mmse_data:
            d3_att = mmse_data['attention'] / 5.0

        # Domain 4: Delayed Recall (Key Episodic Memory AD Driver)
        d4_recall = np.nan
        if test_selected in ['MOCA', 'ALL'] and 'delayed_recall' in moca_data:
            d4_recall = moca_data['delayed_recall'] / 5.0
        elif test_selected in ['MMSE', 'ALL'] and 'delayed_recall' in mmse_data:
            d4_recall = mmse_data['delayed_recall'] / 3.0

        # Domain 5: Language & Praxis
        d5_lang = np.nan
        if test_selected in ['MMSE', 'ALL'] and 'language' in mmse_data:
            d5_lang = mmse_data['language'] / 9.0

        # Domain 6: Functional & Behavioral
        d6_func = (faq_total / 30.0) if pd.notna(faq_total) else (cdr_data.get('cdrsb', 0.0)/18.0 if 'cdrsb' in cdr_data else np.nan)

        # Longitudinal Velocities (if revisit)
        is_revisit = longitudinal_data.get('is_revisit', False)
        dt = longitudinal_data.get('years_since_prior', 1.0) if is_revisit else np.nan
        v_moca = (moca_data.get('total') - longitudinal_data.get('prior_moca')) / dt if (is_revisit and 'total' in moca_data and 'prior_moca' in longitudinal_data) else np.nan
        v_mmse = (mmse_data.get('total') - longitudinal_data.get('prior_mmse')) / dt if (is_revisit and 'total' in mmse_data and 'prior_mmse' in longitudinal_data) else np.nan
        v_cdrsb = (cdr_data.get('cdrsb') - longitudinal_data.get('prior_cdrsb')) / dt if (is_revisit and 'cdrsb' in cdr_data and 'prior_cdrsb' in longitudinal_data) else np.nan
        v_faq = (faq_total - longitudinal_data.get('prior_faq')) / dt if (is_revisit and pd.notna(faq_total) and 'prior_faq' in longitudinal_data) else np.nan

        # Construct Master Inference Vector (Populate selected, leave unselected as NaN)
        feat_dict = {
            'Age': age,
            'Education_Years': education,
            'Gender_Male': gender_male,
            'Cognitive_Reserve_Ratio': education / (age + 1e-5),
            
            # MoCA Battery
            'MoCA_Total': moca_data.get('total', np.nan) if test_selected in ['MOCA', 'ALL'] else np.nan,
            'MoCA_Visuospatial': moca_data.get('visuospatial_executive', np.nan) if test_selected in ['MOCA', 'ALL'] else np.nan,
            'MoCA_Executive_Trails': moca_data.get('trails_pass', np.nan) if test_selected in ['MOCA', 'ALL'] else np.nan,
            'MoCA_Delayed_Recall': moca_data.get('delayed_recall', np.nan) if test_selected in ['MOCA', 'ALL'] else np.nan,
            'MoCA_Orientation': moca_data.get('orientation', np.nan) if test_selected in ['MOCA', 'ALL'] else np.nan,

            # MMSE Battery
            'MMSE_Total': mmse_data.get('total', np.nan) if test_selected in ['MMSE', 'ALL'] else np.nan,
            'MMSE_Orientation': mmse_data.get('orientation', np.nan) if test_selected in ['MMSE', 'ALL'] else np.nan,
            'MMSE_Registration': mmse_data.get('registration', np.nan) if test_selected in ['MMSE', 'ALL'] else np.nan,
            'MMSE_Attention': mmse_data.get('attention', np.nan) if test_selected in ['MMSE', 'ALL'] else np.nan,
            'MMSE_Delayed_Recall': mmse_data.get('delayed_recall', np.nan) if test_selected in ['MMSE', 'ALL'] else np.nan,
            'MMSE_Language': mmse_data.get('language', np.nan) if test_selected in ['MMSE', 'ALL'] else np.nan,

            # CDR & FAQ & NPI-Q
            'CDR_SB': cdr_data.get('cdrsb', np.nan) if test_selected in ['CDR', 'ALL'] else (cdr_data.get('cdrsb', np.nan)),
            'CDR_Memory': cdr_data.get('memory', np.nan),
            'CDR_Orientation': cdr_data.get('orientation', np.nan),
            'CDR_Judgment': cdr_data.get('judgment', np.nan),
            'FAQ_Total': faq_total,
            'NPIQ_Total_Severity': npi_data.get('total_severity', np.nan),
            'NPIQ_Depression': 1 if history_data.get('depression', False) else npi_data.get('depression', 0),
            'NPIQ_Anxiety': npi_data.get('anxiety', 0),

            # Vitals
            'Systolic_BP': sbp,
            'Diastolic_BP': dbp,
            'Pulse_Pressure': pp,
            'BMI': bmi,

            # 6 Universal Domain Scores
            'Domain_1_Orientation': d1_orient,
            'Domain_2_Registration': d2_reg,
            'Domain_3_Attention_Calc': d3_att,
            'Domain_4_Delayed_Recall': d4_recall,
            'Domain_5_Language_Praxis': d5_lang,
            'Domain_6_Functional_Behavior': d6_func,

            # Velocities
            'Velocity_MoCA_per_year': v_moca,
            'Velocity_MMSE_per_year': v_mmse,
            'Velocity_CDRSB_per_year': v_cdrsb,
            'Velocity_FAQ_per_year': v_faq,
            'Time_Since_Prior_Visit_Years': dt
        }

        # Run Predictions
        X_df = pd.DataFrame([feat_dict], columns=self.feature_cols)
        prog_prob = float(self.prog_model.predict_proba(X_df)[0, 1])
        triage_probs = self.triage_model.predict_proba(X_df)[0]
        pred_class_idx = int(np.argmax(triage_probs))
        class_labels = ['Healthy (CN)', 'Mild Cognitive Impairment (MCI)', 'Dementia (Alzheimer\'s)']

        # Prioritization & Decision Gating
        is_escalate = (prog_prob >= 0.40) or (triage_probs[1] >= 0.50) or (triage_probs[2] >= 0.50)
        
        return {
            'triage_diagnosis': class_labels[pred_class_idx],
            'triage_probabilities': {
                'Healthy_CN': round(float(triage_probs[0]) * 100, 1),
                'MCI': round(float(triage_probs[1]) * 100, 1),
                'Dementia': round(float(triage_probs[2]) * 100, 1)
            },
            'progression_risk_24m': round(prog_prob * 100, 1),
            'decision_gating': "ESCALATE TO STAGE 2 (BLOOD BIOMARKERS)" if is_escalate else "ROUTINE PRIMARY CARE MONITORING",
            'six_domains': {
                'Orientation': round(d1_orient * 100, 1) if pd.notna(d1_orient) else 'N/A',
                'Registration': round(d2_reg * 100, 1) if pd.notna(d2_reg) else 'N/A',
                'Attention_Calc': round(d3_att * 100, 1) if pd.notna(d3_att) else 'N/A',
                'Delayed_Recall': round(d4_recall * 100, 1) if pd.notna(d4_recall) else 'N/A',
                'Language_Praxis': round(d5_lang * 100, 1) if pd.notna(d5_lang) else 'N/A',
                'Functional_Daily': round(d6_func * 100, 1) if pd.notna(d6_func) else 'N/A'
            },
            'test_administered': test_selected
        }

def run_doctor_sandbox_demo():
    print("=" * 80)
    print("STEPWISE STAGE 1 CLINICIAN SANDBOX DEMO (INTERACTIVE WORKFLOW)")
    print("=" * 80)

    sandbox = StepWiseStage1Sandbox()

    # DEMO 1: Doctor administers ONLY MoCA (MMSE is left N/A) + FAQ checklist
    patient_1 = {
        'name': "Patient A: Suresh Menon (Age 73, 14y Education)",
        'age': 73,
        'education_years': 14,
        'gender': 'M',
        'systolic_bp': 148,
        'diastolic_bp': 88,
        'bmi': 26.8,
        'selected_test': 'MOCA', # Only MoCA taken
        'moca': {
            'total': 21.0,
            'visuospatial_executive': 2.0, # Clock failed, Trails failed
            'trails_pass': 0,
            'delayed_recall': 1.0,         # 1/5 Delayed Recall (Severe Memory Flag)
            'orientation': 5.0
        },
        # MMSE left completely empty
        'mmse': {},
        # FAQ functional assessment
        'faq': {
            'managing_finances': 'yes',     # Dependent
            'remembering_appointments': 'yes',
            'cooking_meals': 'unknown',
            'driving_traveling': 'no'
        },
        'medical_history': {
            'hypertension': True,
            'depression': False,
            'diabetes': True,
            'stroke': False
        },
        'longitudinal': {
            'is_revisit': True,
            'years_since_prior': 1.0,
            'prior_moca': 25.0 # Dropped 4 points in 1 year!
        }
    }

    # DEMO 2: Doctor administers ONLY MMSE (MoCA is left N/A) - Normal Stable Patient
    patient_2 = {
        'name': "Patient B: Anita Roy (Age 67, 16y Education)",
        'age': 67,
        'education_years': 16,
        'gender': 'F',
        'systolic_bp': 120,
        'diastolic_bp': 78,
        'bmi': 23.4,
        'selected_test': 'MMSE', # Only MMSE taken
        'moca': {},              # MoCA is N/A
        'mmse': {
            'total': 29.0,
            'orientation': 10.0,
            'registration': 3.0,
            'attention': 5.0,
            'delayed_recall': 3.0,
            'language': 8.0
        },
        'faq': {
            'managing_finances': 'no',
            'remembering_appointments': 'no',
            'cooking_meals': 'no'
        },
        'medical_history': {
            'hypertension': False,
            'depression': False,
            'diabetes': False
        },
        'longitudinal': {
            'is_revisit': True,
            'years_since_prior': 1.0,
            'prior_mmse': 29.0 # Stable
        }
    }

    for p in [patient_1, patient_2]:
        res = sandbox.evaluate_patient(p)
        print("\n" + "-" * 80)
        print(f"🩺 PATIENT: {p['name']}")
        print(f"   Selected Test: [{res['test_administered']}] | Other Tests: [NOT APPLICABLE (N/A)]")
        print("-" * 80)
        print(f"🧠 6 UNIVERSAL COGNITIVE DOMAIN BREAKDOWN:")
        for dom, score in res['six_domains'].items():
            print(f"   • {dom:<20}: {str(score)+'%':<8}")
        print(f"\n📊 3-CLASS TRIAGE DIAGNOSIS: {res['triage_diagnosis']}")
        print(f"   Confidence Breakdown: CN: {res['triage_probabilities']['Healthy_CN']}% | MCI: {res['triage_probabilities']['MCI']}% | Dementia: {res['triage_probabilities']['Dementia']}%")
        print(f"🔮 24-MONTH PROGRESSION RISK: {res['progression_risk_24m']}%")
        print(f"🚦 CLINICAL GATING DECISION: {res['decision_gating']}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    run_doctor_sandbox_demo()
