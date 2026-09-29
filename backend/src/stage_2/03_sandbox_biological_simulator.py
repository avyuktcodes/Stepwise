"""
==============================================================================
STEPWISE STAGE 2: BIOLOGICAL SANDBOX SIMULATOR
File: 03_sandbox_biological_simulator.py
Purpose: Simulates real-world clinical inference using the Stage 2 XGBoost 
         models based on blood biomarker inputs (p-tau217, APS2, NfL).
==============================================================================
"""

import os
import xgboost as xgb
import pandas as pd
import numpy as np

def run_biological_sandbox():
    print("==========================================================")
    print("STEPWISE STAGE 2: BLOOD BIOMARKER DIAGNOSTIC SANDBOX")
    print("==========================================================\n")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.abspath(os.path.join(script_dir, "../../../backend/models"))
    
    prog_model_path = os.path.join(model_dir, "stage2_biofluid_progression_xgb.json")
    diag_model_path = os.path.join(model_dir, "stage2_biofluid_triage_xgb.json")
    
    if not os.path.exists(prog_model_path) or not os.path.exists(diag_model_path):
        print("Error: Models not found. Run 02_train_biomarker_engine.py first.")
        return
        
    model_prog = xgb.XGBClassifier()
    model_prog.load_model(prog_model_path)
    
    model_diag = xgb.XGBClassifier()
    model_diag.load_model(diag_model_path)
    
    print("Patient Profile: 72-year-old female, APOE4 carrier.")
    print("Blood Draw Results (C2N PrecivityAD2 & UPENN Quanterix):")
    print(" - p-tau217 (C2N): 3.2 pg/mL (Elevated)")
    print(" - APS2 Score (C2N): 85 / 100 (High Amyloid Probability)")
    print(" - NfL (UPENN): 25.4 pg/mL (Neurodegeneration active)")
    
    # Feature order: ['pT217_C2N', 'AB42_AB40_C2N', 'APS2_C2N', 'NfL_Q', 'GFAP_Q', 'PLASMATAU', 'APOE4_Count', 'Age', 'Gender_Male']
    patient_data = pd.DataFrame([{
        'pT217_C2N': 3.2,
        'AB42_AB40_C2N': 0.08,
        'APS2_C2N': 85.0,
        'NfL_Q': 25.4,
        'GFAP_Q': 150.0,
        'PLASMATAU': 3.5,
        'APOE4_Count': 1.0,
        'Age': 72.0,
        'Gender_Male': 0.0
    }])
    
    prog_risk = model_prog.predict_proba(patient_data)[0][1]
    diag_probs = model_diag.predict_proba(patient_data)[0]
    
    print("\n--- STAGE 2 BIOLOGICAL ENGINE RESULTS ---")
    print(f"24-Month Clinical Progression Risk: {prog_risk*100:.1f}%")
    
    print("\nClinical Diagnosis Probability (Despite Biological AD):")
    print(f" - Clinically Normal (CN): {diag_probs[0]*100:.1f}%")
    print(f" - Mild Cognitive Impairment (MCI): {diag_probs[1]*100:.1f}%")
    print(f" - Clinical Dementia: {diag_probs[2]*100:.1f}%")
    
    print("\n[CLINICAL INSIGHT]")
    print("Notice that while the blood tests show DEFINITIVE biological Alzheimer's (APS2=85),")
    print("the model might still predict a high chance of being Clinically Normal (CN).")
    print("This is the 'Preclinical Alzheimer's' phase: The brain has plaques, but the clinical")
    print("symptoms haven't fully manifested yet. This is exactly why we need blood tests!")
    print("==========================================================")

if __name__ == "__main__":
    run_biological_sandbox()
