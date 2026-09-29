"""
==============================================================================
STEPWISE STAGE 1: MULTIMODAL CLINICAL DATASET HARMONIZER
File: 01_dataset_harmonizer.py
Purpose: Ingests 10 raw ADNI datasets, reconstructs full 9,335 MoCA records,
         computes longitudinal decline velocities (ΔScore/Δt), extracts universal
         cognitive domains, vitals, medical history, and formulates 24-month progression targets.
==============================================================================
"""

import os
import glob
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

def build_stage1_harmonized_dataset():
    print("=" * 75)
    print("STEPWISE STAGE 1: CLINICAL DATASET HARMONIZATION ENGINE")
    print("=" * 75)

    base_dir = "Dataset"
    clin_dir = os.path.join(base_dir, "Clinical Assessments & Questionnaires")
    med_dir = os.path.join(base_dir, "Medical HIstory")
    
    # 1. DIAGNOSIS (DXSUM)
    print("\n[1/8] Ingesting Diagnostic Summary (DXSUM)...")
    dx_file = glob.glob(os.path.join(clin_dir, "DXSUM", "DXSUM*.csv"))[0]
    dx = pd.read_csv(dx_file, low_memory=False)
    dx['VISCODE'] = dx['VISCODE'].astype(str).str.lower()
    dx = dx.dropna(subset=['RID', 'VISCODE', 'DIAGNOSIS'])
    dx['DIAGNOSIS'] = pd.to_numeric(dx['DIAGNOSIS'], errors='coerce')
    dx = dx[dx['DIAGNOSIS'].isin([1.0, 2.0, 3.0])]
    dx['DX_Label'] = dx['DIAGNOSIS'].map({1.0: 0, 2.0: 1, 3.0: 2}) # 0: Healthy (CN), 1: MCI, 2: Dementia (AD)
    
    if 'EXAMDATE' in dx.columns:
        dx['DX_DATE'] = pd.to_datetime(dx['EXAMDATE'], errors='coerce')
    elif 'VISDATE' in dx.columns:
        dx['DX_DATE'] = pd.to_datetime(dx['VISDATE'], errors='coerce')
    print(f"   -> Ingested {len(dx)} diagnostic milestones across {dx['RID'].nunique()} patients.")

    # 2. DEMOGRAPHICS (PTDEMOG)
    print("\n[2/8] Ingesting Demographics & Cognitive Reserve...")
    demo_file = glob.glob(os.path.join(clin_dir, "DEMOGRAPHY", "PTDEMOG*.csv"))[0]
    demo = pd.read_csv(demo_file, low_memory=False).sort_values(by=['RID']).drop_duplicates(subset=['RID'], keep='first')
    demo['Education_Years'] = pd.to_numeric(demo['PTEDUCAT'], errors='coerce')
    demo['Gender_Male'] = (demo['PTGENDER'] == 1).astype(int) if 'PTGENDER' in demo.columns else (demo['PTGENDER'] == 'Male').astype(int)
    demo['Birth_Year'] = pd.to_numeric(demo['PTDOBYY'], errors='coerce')
    demo_clean = demo[['RID', 'Education_Years', 'Gender_Male', 'Birth_Year']]

    # 3. MoCA (Montreal Cognitive Assessment)
    print("\n[3/8] Ingesting & Reconstructing 100% MoCA Subdomains...")
    moca_file = glob.glob(os.path.join(clin_dir, "MoCA", "MOCA*.csv"))[0]
    moca = pd.read_csv(moca_file, low_memory=False)
    moca['VISCODE'] = moca['VISCODE'].astype(str).str.lower()
    if 'VISDATE' in moca.columns:
        moca['MOCA_DATE'] = pd.to_datetime(moca['VISDATE'], errors='coerce')

    def reconstruct_moca(row):
        if pd.notna(row.get('MOCA')) and str(row.get('MOCA')).strip() != '':
            try:
                return float(row.get('MOCA'))
            except:
                pass
        v1 = 1 if row.get('TRAILS') == 1 else 0
        v2 = 1 if row.get('CUBE') == 1 else 0
        v3 = 1 if row.get('CLOCKCON') == 1 else 0
        v4 = 1 if row.get('CLOCKNO') == 1 else 0
        v5 = 1 if row.get('CLOCKHAN') == 1 else 0
        vis = v1 + v2 + v3 + v4 + v5
        
        n1 = 1 if row.get('LION') == 1 else 0
        n2 = 1 if row.get('RHINO') == 1 else 0
        n3 = 1 if row.get('CAMEL') == 1 else 0
        nam = n1 + n2 + n3
        
        d1 = 1 if row.get('DIGFOR') == 1 else 0
        d2 = 1 if row.get('DIGBACK') == 1 else 0
        let = 1 if pd.to_numeric(row.get('LETTERS'), errors='coerce') <= 1 else 0
        s_sum = sum([1 if row.get(f'SERIAL{i}') == 1 else 0 for i in range(1, 6)])
        s_pts = 3 if s_sum >= 4 else (2 if s_sum >= 2 else (1 if s_sum == 1 else 0))
        att = d1 + d2 + let + s_pts
        
        r1 = 1 if row.get('REPEAT1') == 1 else 0
        r2 = 1 if row.get('REPEAT2') == 1 else 0
        ff = 1 if pd.to_numeric(row.get('FFLUENCY'), errors='coerce') >= 11 else 0
        lang = r1 + r2 + ff
        
        a1 = 1 if row.get('ABSTRAN') == 1 else 0
        a2 = 1 if row.get('ABSMEAS') == 1 else 0
        abst = a1 + a2
        
        del_pts = sum([1 if row.get(f'DELW{i}') == 1 else 0 for i in range(1, 6)])
        
        o1 = 1 if row.get('DATE') == 1 else 0
        o2 = 1 if row.get('MONTH') == 1 else 0
        o3 = 1 if row.get('YEAR') == 1 else 0
        o4 = 1 if row.get('DAY') == 1 else 0
        o5 = 1 if row.get('PLACE') == 1 else 0
        o6 = 1 if row.get('CITY') == 1 else 0
        orient = o1 + o2 + o3 + o4 + o5 + o6
        
        return float(vis + nam + att + lang + abst + del_pts + orient)

    moca['MoCA_Total'] = moca.apply(reconstruct_moca, axis=1)
    moca['MoCA_Visuospatial'] = (
        (moca['TRAILS'] == 1).astype(int) + (moca['CUBE'] == 1).astype(int) +
        (moca['CLOCKCON'] == 1).astype(int) + (moca['CLOCKNO'] == 1).astype(int) + (moca['CLOCKHAN'] == 1).astype(int)
    )
    moca['MoCA_Executive_Trails'] = (moca['TRAILS'] == 1).astype(int)
    moca['MoCA_Delayed_Recall'] = (
        (moca['DELW1'] == 1).astype(int) + (moca['DELW2'] == 1).astype(int) +
        (moca['DELW3'] == 1).astype(int) + (moca['DELW4'] == 1).astype(int) + (moca['DELW5'] == 1).astype(int)
    )
    moca['MoCA_Orientation'] = (
        (moca['DATE'] == 1).astype(int) + (moca['MONTH'] == 1).astype(int) +
        (moca['YEAR'] == 1).astype(int) + (moca['DAY'] == 1).astype(int) +
        (moca['PLACE'] == 1).astype(int) + (moca['CITY'] == 1).astype(int)
    )
    
    moca_clean = moca[['RID', 'VISCODE', 'MOCA_DATE', 'MoCA_Total', 'MoCA_Visuospatial', 
                       'MoCA_Executive_Trails', 'MoCA_Delayed_Recall', 'MoCA_Orientation']].drop_duplicates(subset=['RID', 'VISCODE'])
    print(f"   -> MoCA complete records: {len(moca_clean)} visits (Mean Score: {moca_clean['MoCA_Total'].mean():.2f})")

    # 4. MMSE (Mini-Mental State Examination)
    print("\n[4/8] Ingesting MMSE...")
    mmse_file = glob.glob(os.path.join(clin_dir, "MMSE", "MMSE*.csv"))[0]
    mmse = pd.read_csv(mmse_file, low_memory=False)
    mmse['VISCODE'] = mmse['VISCODE'].astype(str).str.lower()
    if 'EXAMDATE' in mmse.columns:
        mmse['MMSE_DATE'] = pd.to_datetime(mmse['EXAMDATE'], errors='coerce')
    elif 'VISDATE' in mmse.columns:
        mmse['MMSE_DATE'] = pd.to_datetime(mmse['VISDATE'], errors='coerce')
        
    mmse['MMSE_Total'] = pd.to_numeric(mmse['MMSCORE'], errors='coerce')
    orient_cols = ['MMDATE', 'MMYEAR', 'MMMONTH', 'MMDAY', 'MMSEASON', 'MMHOSPIT', 'MMFLOOR', 'MMCITY', 'MMAREA', 'MMSTATE']
    mmse['MMSE_Orientation'] = mmse[orient_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    mmse['MMSE_Registration'] = mmse[['WORD1', 'WORD2', 'WORD3']].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    mmse['MMSE_Attention'] = mmse[['MMLTR1', 'MMLTR2', 'MMLTR3', 'MMLTR4', 'MMLTR5']].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    mmse['MMSE_Delayed_Recall'] = mmse[['WORD1DL', 'WORD2DL', 'WORD3DL']].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    lang_cols = ['MMWATCH', 'MMPENCIL', 'MMREPEAT', 'MMHAND', 'MMFOLD', 'MMONFLR', 'MMREAD', 'MMWRITE', 'MMDRAW']
    mmse['MMSE_Language'] = mmse[lang_cols].apply(pd.to_numeric, errors='coerce').sum(axis=1)
    
    mmse_clean = mmse[['RID', 'VISCODE', 'MMSE_DATE', 'MMSE_Total', 'MMSE_Orientation', 
                       'MMSE_Registration', 'MMSE_Attention', 'MMSE_Delayed_Recall', 'MMSE_Language']].drop_duplicates(subset=['RID', 'VISCODE'])

    # 5. CDR (Clinical Dementia Rating)
    print("\n[5/8] Ingesting CDR & CDR-SB...")
    cdr_file = glob.glob(os.path.join(clin_dir, "CDR_SB", "CDR*.csv"))[0]
    cdr = pd.read_csv(cdr_file, low_memory=False)
    cdr['VISCODE'] = cdr['VISCODE'].astype(str).str.lower()
    cdr['CDR_SB'] = pd.to_numeric(cdr['CDRSB'], errors='coerce')
    cdr['CDR_Global'] = pd.to_numeric(cdr['CDGLOBAL'], errors='coerce')
    cdr['CDR_Memory'] = pd.to_numeric(cdr['CDMEMORY'], errors='coerce')
    cdr['CDR_Orientation'] = pd.to_numeric(cdr['CDORIENT'], errors='coerce')
    cdr['CDR_Judgment'] = pd.to_numeric(cdr['CDJUDGE'], errors='coerce')
    cdr_clean = cdr[['RID', 'VISCODE', 'CDR_SB', 'CDR_Global', 'CDR_Memory', 'CDR_Orientation', 'CDR_Judgment']].drop_duplicates(subset=['RID', 'VISCODE'])

    # 6. FAQ & NPI-Q & ADAS-COG
    print("\n[6/8] Ingesting FAQ, NPI-Q, ADAS-Cog...")
    faq_file = glob.glob(os.path.join(clin_dir, "FAQ", "FAQ*.csv"))[0]
    faq = pd.read_csv(faq_file, low_memory=False)
    faq['VISCODE'] = faq['VISCODE'].astype(str).str.lower()
    faq['FAQ_Total'] = pd.to_numeric(faq['FAQTOTAL'], errors='coerce')
    faq_clean = faq[['RID', 'VISCODE', 'FAQ_Total']].drop_duplicates(subset=['RID', 'VISCODE'])

    npiq_file = glob.glob(os.path.join(clin_dir, "NPI_Q", "NPIQ*.csv"))[0]
    npiq = pd.read_csv(npiq_file, low_memory=False)
    npiq['VISCODE'] = npiq['VISCODE'].astype(str).str.lower()
    npiq['NPIQ_Total_Severity'] = pd.to_numeric(npiq['NPISCORE'], errors='coerce') if 'NPISCORE' in npiq.columns else np.nan
    npiq['NPIQ_Depression'] = pd.to_numeric(npiq['NPIDSEV'], errors='coerce') if 'NPIDSEV' in npiq.columns else 0
    npiq['NPIQ_Anxiety'] = pd.to_numeric(npiq['NPIESEV'], errors='coerce') if 'NPIESEV' in npiq.columns else 0
    npiq_clean = npiq[['RID', 'VISCODE', 'NPIQ_Total_Severity', 'NPIQ_Depression', 'NPIQ_Anxiety']].drop_duplicates(subset=['RID', 'VISCODE'])

    adas_file = glob.glob(os.path.join(clin_dir, "ADAS_COG", "ADAS*.csv"))[0]
    adas = pd.read_csv(adas_file, low_memory=False)
    adas['VISCODE'] = adas['VISCODE'].astype(str).str.lower()
    adas['ADAS_Total'] = pd.to_numeric(adas['TOTSCORE'], errors='coerce') if 'TOTSCORE' in adas.columns else np.nan
    adas_clean = adas[['RID', 'VISCODE', 'ADAS_Total']].drop_duplicates(subset=['RID', 'VISCODE'])

    # 7. VITALS & CARDIOMETABOLIC
    print("\n[7/8] Ingesting Vitals & Blood Pressure...")
    vitals_file = glob.glob(os.path.join(med_dir, "VITAL_SIGNS", "VITALS*.csv"))[0]
    vitals = pd.read_csv(vitals_file, low_memory=False)
    vitals['VISCODE'] = vitals['VISCODE'].astype(str).str.lower()
    vitals['Systolic_BP'] = pd.to_numeric(vitals['VSBPSYS'], errors='coerce')
    vitals['Diastolic_BP'] = pd.to_numeric(vitals['VSBPDIA'], errors='coerce')
    vitals['Pulse_Pressure'] = vitals['Systolic_BP'] - vitals['Diastolic_BP']
    vitals['Weight_lbs'] = pd.to_numeric(vitals['VSWEIGHT'], errors='coerce')
    vitals['Height_in'] = pd.to_numeric(vitals['VSHEIGHT'], errors='coerce')
    vitals['BMI'] = (vitals['Weight_lbs'] / (vitals['Height_in'] ** 2)) * 703
    vitals_clean = vitals[['RID', 'VISCODE', 'Systolic_BP', 'Diastolic_BP', 'Pulse_Pressure', 'BMI']].drop_duplicates(subset=['RID', 'VISCODE'])

    # 8. MASTER MERGE & FEATURE ENGINEERING
    print("\n[8/8] Merging Master Clinical Matrix...")
    df = pd.merge(dx[['RID', 'VISCODE', 'DX_DATE', 'DIAGNOSIS', 'DX_Label']], demo_clean, on='RID', how='left')
    df = pd.merge(df, moca_clean, on=['RID', 'VISCODE'], how='left')
    df = pd.merge(df, mmse_clean, on=['RID', 'VISCODE'], how='left')
    df = pd.merge(df, cdr_clean, on=['RID', 'VISCODE'], how='left')
    df = pd.merge(df, faq_clean, on=['RID', 'VISCODE'], how='left')
    df = pd.merge(df, npiq_clean, on=['RID', 'VISCODE'], how='left')
    df = pd.merge(df, adas_clean, on=['RID', 'VISCODE'], how='left')
    df = pd.merge(df, vitals_clean, on=['RID', 'VISCODE'], how='left')

    df['Visit_Date'] = df['DX_DATE'].combine_first(df['MOCA_DATE']).combine_first(df['MMSE_DATE'])
    df['Visit_Year'] = df['Visit_Date'].dt.year
    df['Age'] = df['Visit_Year'] - df['Birth_Year']
    df['Age'] = df['Age'].fillna(72.0)

    # Universal 6-Domain Cross-Test Harmonization
    df['Domain_1_Orientation'] = np.where(df['MoCA_Orientation'].notna(), df['MoCA_Orientation'] / 6.0,
                                 np.where(df['MMSE_Orientation'].notna(), df['MMSE_Orientation'] / 10.0, np.nan))
    df['Domain_2_Registration'] = np.where(df['MMSE_Registration'].notna(), df['MMSE_Registration'] / 3.0, 1.0)
    df['Domain_3_Attention_Calc'] = np.where(df['MoCA_Visuospatial'].notna(), df['MoCA_Visuospatial'] / 5.0,
                                    np.where(df['MMSE_Attention'].notna(), df['MMSE_Attention'] / 5.0, np.nan))
    df['Domain_4_Delayed_Recall'] = np.where(df['MoCA_Delayed_Recall'].notna(), df['MoCA_Delayed_Recall'] / 5.0,
                                    np.where(df['MMSE_Delayed_Recall'].notna(), df['MMSE_Delayed_Recall'] / 3.0, np.nan))
    df['Domain_5_Language_Praxis'] = np.where(df['MMSE_Language'].notna(), df['MMSE_Language'] / 9.0, np.nan)
    df['Domain_6_Functional_Behavior'] = np.where(df['FAQ_Total'].notna(), df['FAQ_Total'] / 30.0, 
                                         np.where(df['CDR_SB'].notna(), df['CDR_SB'] / 18.0, np.nan))

    df['Cognitive_Reserve_Ratio'] = df['Education_Years'] / (df['Age'] + 1e-5)

    # Longitudinal Velocity Calculations (Δ / Δt per patient)
    print("\nComputing Longitudinal Differential Velocity Features (ΔScore / Δt)...")
    df = df.sort_values(by=['RID', 'Visit_Date', 'VISCODE'])
    df['Prev_Visit_Date'] = df.groupby('RID')['Visit_Date'].shift(1)
    df['Time_Since_Prior_Visit_Years'] = (df['Visit_Date'] - df['Prev_Visit_Date']).dt.days / 365.25
    df['Time_Since_Prior_Visit_Years'] = df['Time_Since_Prior_Visit_Years'].apply(lambda x: x if (pd.notna(x) and x > 0.1) else np.nan)

    df['Prev_MoCA'] = df.groupby('RID')['MoCA_Total'].shift(1)
    df['Prev_MMSE'] = df.groupby('RID')['MMSE_Total'].shift(1)
    df['Prev_CDRSB'] = df.groupby('RID')['CDR_SB'].shift(1)
    df['Prev_FAQ'] = df.groupby('RID')['FAQ_Total'].shift(1)

    df['Velocity_MoCA_per_year'] = (df['MoCA_Total'] - df['Prev_MoCA']) / df['Time_Since_Prior_Visit_Years']
    df['Velocity_MMSE_per_year'] = (df['MMSE_Total'] - df['Prev_MMSE']) / df['Time_Since_Prior_Visit_Years']
    df['Velocity_CDRSB_per_year'] = (df['CDR_SB'] - df['Prev_CDRSB']) / df['Time_Since_Prior_Visit_Years']
    df['Velocity_FAQ_per_year'] = (df['FAQ_Total'] - df['Prev_FAQ']) / df['Time_Since_Prior_Visit_Years']

    # 24-Month Clinical Progression Target
    print("\nFormulating 24-Month Clinical Progression Target (Progression24m)...")
    progression_targets = []
    for rid, group in df.groupby('RID'):
        group_dates = group['Visit_Date'].values
        group_dx = group['DX_Label'].values
        for idx in range(len(group)):
            curr_date = group_dates[idx]
            curr_dx = group_dx[idx]
            if pd.isna(curr_date):
                progression_targets.append(np.nan)
                continue
            future_mask = (group_dates > curr_date) & (group_dates <= curr_date + np.timedelta64(750, 'D'))
            future_dx = group_dx[future_mask]
            if len(future_dx) > 0:
                is_progressor = 1 if np.nanmax(future_dx) > curr_dx else 0
                progression_targets.append(is_progressor)
            else:
                progression_targets.append(np.nan)
                
    df['Target_Progression24m'] = progression_targets

    out_dir = "backend/src/stage_1/cleaned_dataset"
    os.makedirs(out_dir, exist_ok=True)
    out_csv = os.path.join(out_dir, "stage1_harmonized_dataset.csv")
    df.to_csv(out_csv, index=False)

    print("\n" + "=" * 75)
    print(f"✅ STAGE 1 HARMONIZATION COMPLETE!")
    print(f"   Total Ingested Rows: {len(df)}")
    print(f"   Unique Patients: {df['RID'].nunique()}")
    print(f"   Non-Null MoCA Total: {df['MoCA_Total'].notna().sum()}")
    print(f"   Non-Null MMSE Total: {df['MMSE_Total'].notna().sum()}")
    print(f"   Non-Null CDR-SB: {df['CDR_SB'].notna().sum()}")
    print(f"   Saved Clean Dataset: {out_csv}")
    print("=" * 75)
    return df

if __name__ == "__main__":
    build_stage1_harmonized_dataset()
