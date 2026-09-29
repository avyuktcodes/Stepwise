import pandas as pd
import numpy as np
import os

def build_stage_3_mri_dataset():
    print("🚀 Initializing Stage 3: Multimodal Structural MRI Harmonizer...")
    
    base_path = "/Users/avyukt/Desktop/stepwise/Dataset/Structural MRI Atrophy"
    
    # 1. Load UCSD Derived Volumes (Hippocampus, Ventricles, ICV)
    print("Loading UCSD Derived Volumes...")
    vol_df = pd.read_csv(os.path.join(base_path, "(UCSD) - Derived Volumes", "UCSDVOL_26Sep2026.csv"), low_memory=False)
    
    # 2. Load Gray Matter Thinning (FreeSurfer Cortical Thickness)
    print("Loading Gray Matter Cortical Thickness...")
    gm_df = pd.read_csv(os.path.join(base_path, "Gray Matter Thining", "UCSFFSX7_26Sep2026.csv"), low_memory=False)
    
    # 3. Load White Matter Hyperintensities (Vascular Pathology)
    print("Loading White Matter Hyperintensities...")
    wm_df = pd.read_csv(os.path.join(base_path, "White Matter Hyperintensity", "UCD_WMH_26Sep2026.csv"), low_memory=False)

    print("\n🔗 Harmonizing all modalities via Inner Join on Patient ID (RID) and Visit (VISCODE)...")
    
    # Standardize column names for merging
    for df in [vol_df, gm_df, wm_df]:
        if 'VISCODE2' in df.columns and 'VISCODE' not in df.columns:
            df.rename(columns={'VISCODE2': 'VISCODE'}, inplace=True)
            
    # Merge datasets
    merged_df = pd.merge(vol_df, gm_df, on=['RID', 'VISCODE'], how='inner', suffixes=('_vol', '_gm'))
    merged_df = pd.merge(merged_df, wm_df, on=['RID', 'VISCODE'], how='inner')
    
    print(f"✅ Merged Cohort Size: {len(merged_df)} MRI scans with full structural profiling.")

    print("\n⚙️ Executing Feature Engineering (Age/Sex Normalization & ICV Ratios)...")
    # Example Feature Engineering (we will map exact ADNI column names in the next step)
    # merged_df['ICV_Norm_Hippocampus'] = (merged_df['Hippocampus_Volume'] / merged_df['ICV']) * 100
    # merged_df['HVR_Ratio'] = merged_df['Hippocampus_Volume'] / merged_df['Ventricle_Volume']
    
    # Save the harmonized master dataset
    output_path = "/Users/avyukt/Desktop/stepwise/backend/src/stage_3/harmonized_mri_features.csv"
    merged_df.to_csv(output_path, index=False)
    print(f"💾 Master MRI Dataset saved to {output_path}")

if __name__ == "__main__":
    build_stage_3_mri_dataset()
