import json
import random
import os
from datetime import datetime, timedelta

def generate_mock_ui_data():
    print("Generating Stratified UI Test Dataset (40 Patients)...")
    
    patients = []
    
    # Define clinical stages
    # Stage 1: Only Cognitive
    # Stage 2: Cognitive + Blood
    # Stage 3: Cog + Blood + MRI
    # Stage 4: Cog + Blood + MRI + PET
    
    for i in range(1, 41):
        rid = f"{i:04d}"
        
        # Determine the patient's current stage in the diagnostic funnel
        current_stage = random.choices([1, 2, 3, 4], weights=[10, 30, 40, 20])[0]
        
        # Determine true biological diagnosis (to generate realistic numbers)
        true_dx = random.choices(['Healthy', 'MCI', 'Alzheimers'], weights=[30, 40, 30])[0]
        
        # Base demographics
        patient = {
            "id": rid,
            "name": f"Patient_{rid}",
            "age": random.randint(55, 85),
            "gender": random.choice(["Male", "Female"]),
            "last_visit": (datetime.now() - timedelta(days=random.randint(1, 30))).strftime("%Y-%m-%d"),
            "current_stage": current_stage,
            "data": {},
            "predictions": {}
        }
        
        # --- STAGE 1: Cognitive & Clinical ---
        if current_stage >= 1:
            if true_dx == 'Healthy':
                mmse = random.randint(28, 30)
                risk_1 = random.uniform(0.01, 0.20)
            elif true_dx == 'MCI':
                mmse = random.randint(24, 27)
                risk_1 = random.uniform(0.40, 0.70)
            else:
                mmse = random.randint(12, 23)
                risk_1 = random.uniform(0.75, 0.99)
                
            patient["data"]["stage_1"] = {
                "mmse": mmse,
                "family_history": random.choice(["Yes", "No"]),
                "comorbidities": random.choice(["None", "Hypertension", "Diabetes"])
            }
            patient["predictions"]["stage_1_risk"] = round(risk_1, 2)
            
        # --- STAGE 2: Blood Biomarkers ---
        if current_stage >= 2:
            if true_dx == 'Healthy':
                ptau = round(random.uniform(0.5, 1.2), 2)
            elif true_dx == 'MCI':
                ptau = round(random.uniform(1.3, 2.5), 2)
            else:
                ptau = round(random.uniform(2.6, 5.0), 2)
                
            patient["data"]["stage_2"] = {
                "pTau217_Z": ptau,
                "NfL_Z": round(random.uniform(0.5, 3.0), 2),
                "GFAP_Z": round(random.uniform(0.5, 3.0), 2)
            }
            # Bayesian update based on blood
            risk_2 = risk_1 + random.uniform(0.05, 0.20) if true_dx != 'Healthy' else risk_1 - random.uniform(0.05, 0.10)
            patient["predictions"]["stage_2_risk"] = round(min(max(risk_2, 0.01), 0.99), 2)
            
        # --- STAGE 3: Structural MRI ---
        if current_stage >= 3:
            if true_dx == 'Healthy':
                hippo = round(random.uniform(6.5, 8.0), 2)
            elif true_dx == 'MCI':
                hippo = round(random.uniform(5.0, 6.4), 2)
            else:
                hippo = round(random.uniform(3.0, 4.9), 2)
                
            patient["data"]["stage_3"] = {
                "hippocampus_volume_cm3": hippo,
                "icv_norm_ratio": round(hippo / random.uniform(1400, 1600) * 1000, 2),
                "white_matter_lesions": random.choice(["Low", "Moderate", "Severe"])
            }
            risk_3 = risk_2 + random.uniform(0.05, 0.15) if true_dx != 'Healthy' else risk_2 - random.uniform(0.05, 0.10)
            patient["predictions"]["stage_3_risk"] = round(min(max(risk_3, 0.01), 0.99), 2)
            
        # --- STAGE 4: PET Scans ---
        if current_stage >= 4:
            if true_dx == 'Healthy':
                centiloid = random.randint(0, 15)
                braak = "I-II"
            elif true_dx == 'MCI':
                centiloid = random.randint(16, 50)
                braak = "III-IV"
            else:
                centiloid = random.randint(51, 100)
                braak = "V-VI"
                
            patient["data"]["stage_4"] = {
                "amyloid_centiloid": centiloid,
                "braak_stage": braak,
                "regional_tau_temporal": round(random.uniform(1.0, 3.0), 2)
            }
            risk_4 = 0.99 if true_dx == 'Alzheimers' else (0.85 if true_dx == 'MCI' else 0.05)
            patient["predictions"]["stage_4_risk"] = risk_4
            
        # Final Diagnosis Status (For UI coloring)
        if "stage_4_risk" in patient["predictions"]:
            final_risk = patient["predictions"]["stage_4_risk"]
        elif "stage_3_risk" in patient["predictions"]:
            final_risk = patient["predictions"]["stage_3_risk"]
        elif "stage_2_risk" in patient["predictions"]:
            final_risk = patient["predictions"]["stage_2_risk"]
        else:
            final_risk = patient["predictions"]["stage_1_risk"]
            
        if final_risk >= 0.75:
            patient["status"] = "High Risk (Action Required)"
        elif final_risk >= 0.40:
            patient["status"] = "Monitoring (MCI)"
        else:
            patient["status"] = "Cleared (Healthy)"

        patients.append(patient)

    # Save to frontend directory
    os.makedirs("/Users/avyukt/Desktop/stepwise/frontend/src/data", exist_ok=True)
    with open("/Users/avyukt/Desktop/stepwise/frontend/src/data/mock_patients.json", "w") as f:
        json.dump(patients, f, indent=4)
        
    print("✅ Created UI Test Dataset with 40 stratified patients.")

if __name__ == "__main__":
    generate_mock_ui_data()
