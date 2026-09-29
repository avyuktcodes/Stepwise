import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
import xgboost as xgb
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def train_stage2_fused():
    print("🧠 Initializing Stage 2 (V2): Bayesian Triage Engine...")
    
    data_path = "/Users/avyukt/Desktop/stepwise/backend/src/stage_2/stage_2_fused_master.csv"
    df = pd.read_csv(data_path)
    
    # Features now include BOTH Biofluids AND Stage 1 Clinical Priors (MMSE, Age, etc.)
    features = [
        'Biofluid_Z_pTau217', 'Biofluid_Z_NfL', 'Biofluid_Z_GFAP',
        'MMSE_Total', 'Age', 'Gender_Male'
    ]
    # Ensure features exist
    features = [f for f in features if f in df.columns]
    
    # 1. 3-Class Triage Model
    df_clean = df.dropna(subset=['DX_Label'] + features)
    X = df_clean[features]
    y = df_clean['DX_Label']
    groups = df_clean['RID']
    
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    
    sgkf = StratifiedGroupKFold(n_splits=5)
    accuracies = []
    
    for train_idx, test_idx in sgkf.split(X, y, groups):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        accuracies.append(accuracy_score(y_test, preds))
        
    print(f"\n--- 3-Class Diagnostic Accuracy ---")
    print(f"Overall Accuracy: {np.mean(accuracies)*100:.2f}% (Ashish Benchmark: 82.1%)")
    
    # Train final and print classification report
    model.fit(X, y)
    final_preds = model.predict(X)
    print("\nDetailed Recall (Sensitivity) Breakdown:")
    print(classification_report(y, final_preds, target_names=['Healthy(0)', 'MCI(1)', 'Dementia(2)']))
    
if __name__ == "__main__":
    train_stage2_fused()
