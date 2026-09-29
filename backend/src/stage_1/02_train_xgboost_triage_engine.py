"""
==============================================================================
STEPWISE STAGE 1: UNIFIED XGBOOST TRIAGE & PROGRESSION ENGINE
File: 02_train_xgboost_triage_engine.py
Purpose: Trains 5-fold patient-grouped XGBoost models for:
         1. 24-Month Future Progression Risk (Model A) with Top-K queue metrics (5%, 10%, 20%, 30%)
         2. 3-Class Triage Classification (Model B: Healthy vs MCI vs Dementia)
         3. Generates Confusion Matrices, ROC/PR curves, and Feature Importance charts.
==============================================================================
"""

import os
import joblib
import pandas as pd
import numpy as np
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score, brier_score_loss, 
    classification_report, confusion_matrix, roc_curve, precision_recall_curve
)
from sklearn.utils.class_weight import compute_sample_weight
import warnings
warnings.filterwarnings('ignore')

class Stage1XGBoostTriageEngine:
    def __init__(self, data_path="backend/src/stage_1/cleaned_dataset/stage1_harmonized_dataset.csv"):
        self.data_path = data_path
        self.models_dir = "backend/models"
        self.viz_dir = "backend/src/stage_1/visualizations"
        os.makedirs(self.models_dir, exist_ok=True)
        os.makedirs(self.viz_dir, exist_ok=True)
        
        # Clinical feature space
        self.feature_cols = [
            # Demographics & Reserve
            'Age', 'Education_Years', 'Gender_Male', 'Cognitive_Reserve_Ratio',
            # MoCA Battery (Totals & Granular Subdomains)
            'MoCA_Total', 'MoCA_Visuospatial', 'MoCA_Executive_Trails', 
            'MoCA_Delayed_Recall', 'MoCA_Orientation',
            # MMSE Battery (Totals & Subdomains)
            'MMSE_Total', 'MMSE_Orientation', 'MMSE_Registration', 
            'MMSE_Attention', 'MMSE_Delayed_Recall', 'MMSE_Language',
            # CDR & Functional FAQ & Neuropsychiatric
            'CDR_SB', 'CDR_Memory', 'CDR_Orientation', 'CDR_Judgment',
            'FAQ_Total', 'NPIQ_Total_Severity', 'NPIQ_Depression', 'NPIQ_Anxiety',
            # Vitals & Arterial Stiffness
            'Systolic_BP', 'Diastolic_BP', 'Pulse_Pressure', 'BMI',
            # Universal 6-Domain Harmonized Indices
            'Domain_1_Orientation', 'Domain_2_Registration', 'Domain_3_Attention_Calc',
            'Domain_4_Delayed_Recall', 'Domain_5_Language_Praxis', 'Domain_6_Functional_Behavior',
            # Longitudinal Decline Velocity (per year)
            'Velocity_MoCA_per_year', 'Velocity_MMSE_per_year', 
            'Velocity_CDRSB_per_year', 'Velocity_FAQ_per_year',
            'Time_Since_Prior_Visit_Years'
        ]

    def load_data(self):
        print(f"Loading harmonized dataset from {self.data_path}...")
        df = pd.read_csv(self.data_path, low_memory=False)
        print(f"Loaded {len(df)} total visits across {df['RID'].nunique()} unique patients.")
        return df

    def evaluate_top_k_queue(self, y_true, y_probs, k_percentages=[5, 10, 20, 30, 50]):
        """
        Computes Top-K Prioritization Queue Performance:
        Shows what percentage of future progressors are captured when reviewing only the Top K% highest-risk patients.
        """
        n_total = len(y_true)
        n_pos = (y_true == 1).sum()
        df_eval = pd.DataFrame({'true': y_true, 'prob': y_probs}).sort_values(by='prob', ascending=False).reset_index(drop=True)
        
        print("\n" + "=" * 80)
        print("🎯 TOP-K RISK PRIORITIZATION QUEUE METRICS (CLINICAL TRIAGE WORKLOAD EFFICIENCY)")
        print("=" * 80)
        print(f"Total Cohort Size: {n_total:,} visits | Total 24-Month Progressors: {n_pos:,} ({n_pos/n_total*100:.1f}%)")
        print("-" * 80)
        print(f"{'Queue Cutoff':<15} | {'Patients Reviewed':<18} | {'Progressors Captured':<22} | {'Capture Rate (Recall)':<22} | {'Enrichment Precision':<20}")
        print("-" * 80)

        results = []
        for k in k_percentages:
            cutoff_count = int(np.ceil(n_total * (k / 100.0)))
            top_k_subset = df_eval.iloc[:cutoff_count]
            captured_pos = (top_k_subset['true'] == 1).sum()
            capture_rate = (captured_pos / n_pos) * 100.0
            precision_in_k = (captured_pos / cutoff_count) * 100.0
            
            print(f"Top {k:>2}% of Queue | {cutoff_count:>6,} patients ({k:>2}%) | {captured_pos:>6,} progressors     | {capture_rate:>6.2f}% of all progressors | {precision_in_k:>6.2f}% progressors")
            results.append({
                'k_percent': k,
                'patients_reviewed': cutoff_count,
                'progressors_captured': captured_pos,
                'capture_rate': capture_rate,
                'precision': precision_in_k
            })
        print("-" * 80)
        return pd.DataFrame(results)

    def train_progression_model(self, df):
        print("\n" + "=" * 80)
        print("STAGE 1 MODEL A: 24-MONTH CLINICAL PROGRESSION PREDICTOR")
        print("=" * 80)

        df_target = df.dropna(subset=['Target_Progression24m']).copy()
        X = df_target[self.feature_cols]
        y = df_target['Target_Progression24m'].astype(int)
        groups = df_target['RID']

        pos_ratio = (y == 0).sum() / (y == 1).sum()
        print(f"Cohort Size: {len(X)} visits | Non-Progressors: {(y==0).sum()} | Progressors: {(y==1).sum()}")
        print(f"scale_pos_weight: {pos_ratio:.2f}")

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros(len(X))
        fold_aucs, fold_praucs, fold_briers = [], [], []

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            model = xgb.XGBClassifier(
                n_estimators=180,
                max_depth=4,
                learning_rate=0.03,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=pos_ratio,
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False
            )
            model.fit(X_train, y_train)
            val_probs = model.predict_proba(X_val)[:, 1]
            oof_preds[val_idx] = val_probs

            auc = roc_auc_score(y_val, val_probs)
            prauc = average_precision_score(y_val, val_probs)
            brier = brier_score_loss(y_val, val_probs)
            
            fold_aucs.append(auc)
            fold_praucs.append(prauc)
            fold_briers.append(brier)
            print(f"   Fold {fold} - ROC-AUC: {auc:.4f} | PR-AUC: {prauc:.4f} | Brier: {brier:.4f}")

        overall_auc = roc_auc_score(y, oof_preds)
        overall_prauc = average_precision_score(y, oof_preds)
        overall_brier = brier_score_loss(y, oof_preds)

        print("\n" + "-" * 80)
        print(f"✅ 5-FOLD PATIENT-GROUPED PROGRESSION MODEL BENCHMARKS:")
        print(f"   Mean ROC-AUC: {np.mean(fold_aucs):.4f} ± {np.std(fold_aucs):.4f}")
        print(f"   Mean PR-AUC:  {np.mean(fold_praucs):.4f} ± {np.std(fold_praucs):.4f}")
        print(f"   Mean Brier:   {np.mean(fold_briers):.4f} ± {np.std(fold_briers):.4f}")
        print(f"   Overall Out-of-Fold ROC-AUC: {overall_auc:.4f}")
        print(f"   Overall Out-of-Fold PR-AUC:  {overall_prauc:.4f}")
        print("-" * 80)

        # Evaluate Top-K Queue
        self.evaluate_top_k_queue(y.values, oof_preds)

        # Train final production model
        final_model = xgb.XGBClassifier(
            n_estimators=180,
            max_depth=4,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=pos_ratio,
            random_state=42,
            eval_metric="logloss",
            use_label_encoder=False
        )
        final_model.fit(X, y)

        model_payload = {
            'model': final_model,
            'feature_names': self.feature_cols,
            'metrics': {'roc_auc': float(overall_auc), 'pr_auc': float(overall_prauc), 'brier': float(overall_brier)}
        }
        joblib.dump(model_payload, os.path.join(self.models_dir, "stage1_progression_xgb.joblib"))
        
        # Plot evaluation curves & Top-K chart
        self._plot_progression_evaluations(y.values, oof_preds, final_model)
        return model_payload

    def train_3class_triage_model(self, df):
        print("\n" + "=" * 80)
        print("STAGE 1 MODEL B: 3-CLASS TRIAGE CLASSIFIER (HEALTHY / MCI / DEMENTIA)")
        print("=" * 80)

        df_target = df.dropna(subset=['DX_Label']).copy()
        X = df_target[self.feature_cols]
        y = df_target['DX_Label'].astype(int)
        groups = df_target['RID']

        print(f"Cohort Size: {len(X)} visits across {df_target['RID'].nunique()} patients.")
        print(f"Class Breakdown: Healthy (CN): {(y==0).sum()} | MCI: {(y==1).sum()} | Dementia: {(y==2).sum()}")

        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        oof_preds = np.zeros((len(X), 3))

        for fold, (train_idx, val_idx) in enumerate(sgkf.split(X, y, groups=groups), 1):
            X_train, y_train = X.iloc[train_idx], y.iloc[train_idx]
            X_val, y_val = X.iloc[val_idx], y.iloc[val_idx]

            sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

            model = xgb.XGBClassifier(
                objective='multi:softprob',
                num_class=3,
                n_estimators=180,
                max_depth=4,
                learning_rate=0.04,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric="mlogloss",
                use_label_encoder=False
            )
            model.fit(X_train, y_train, sample_weight=sample_weights)
            oof_preds[val_idx] = model.predict_proba(X_val)
            print(f"   Fold {fold} complete.")

        y_pred_classes = np.argmax(oof_preds, axis=1)
        target_names = ['Healthy (CN)', 'MCI (Mild Impairment)', 'Dementia (Alzheimer\'s)']

        print("\n" + "-" * 80)
        print("✅ 5-FOLD PATIENT-GROUPED CLASSIFICATION REPORT (3-CLASS TRIAGE):")
        print("-" * 80)
        print(classification_report(y, y_pred_classes, target_names=target_names, digits=4))
        print("-" * 80)

        # Confusion Matrix
        cm = confusion_matrix(y, y_pred_classes)
        print("\n📊 CONFUSION MATRIX (COUNTS):")
        print(pd.DataFrame(cm, index=[f"True: {c}" for c in target_names], columns=[f"Pred: {c}" for c in target_names]))

        # Normalized Confusion Matrix
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("\n📊 NORMALIZED CONFUSION MATRIX (% RECALL):")
        print(pd.DataFrame(np.round(cm_norm*100, 2), index=[f"True: {c}" for c in target_names], columns=[f"Pred: {c} (%)" for c in target_names]))

        # Plot Confusion Matrix
        self._plot_confusion_matrix(cm, cm_norm, target_names)

        # Train final production multi-class model
        final_weights = compute_sample_weight(class_weight='balanced', y=y)
        final_model_3class = xgb.XGBClassifier(
            objective='multi:softprob',
            num_class=3,
            n_estimators=180,
            max_depth=4,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            eval_metric="mlogloss",
            use_label_encoder=False
        )
        final_model_3class.fit(X, y, sample_weight=final_weights)

        model_payload = {
            'model': final_model_3class,
            'feature_names': self.feature_cols,
            'class_names': target_names,
            'confusion_matrix': cm.tolist()
        }
        joblib.dump(model_payload, os.path.join(self.models_dir, "stage1_triage_3class_xgb.joblib"))
        print(f"\nSaved Stage 1 3-Class Triage Model to backend/models/stage1_triage_3class_xgb.joblib")
        return model_payload

    def _plot_progression_evaluations(self, y_true, y_probs, model):
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        # 1. ROC
        fpr, tpr, _ = roc_curve(y_true, y_probs)
        auc = roc_auc_score(y_true, y_probs)
        axes[0].plot(fpr, tpr, color='#0284c7', lw=2.5, label=f'Progression ROC (AUC = {auc:.3f})')
        axes[0].plot([0, 1], [0, 1], color='#94a3b8', linestyle='--')
        axes[0].set_title('5-Fold Patient-Grouped ROC Curve', fontsize=12, fontweight='bold')
        axes[0].set_xlabel('False Positive Rate')
        axes[0].set_ylabel('True Positive Rate (Sensitivity)')
        axes[0].legend(loc='lower right')
        axes[0].grid(alpha=0.2)

        # 2. PR Curve
        precision, recall, _ = precision_recall_curve(y_true, y_probs)
        prauc = average_precision_score(y_true, y_probs)
        axes[1].plot(recall, precision, color='#10b981', lw=2.5, label=f'PR Curve (PR-AUC = {prauc:.3f})')
        axes[1].set_title('Precision-Recall Curve', fontsize=12, fontweight='bold')
        axes[1].set_xlabel('Recall (Sensitivity)')
        axes[1].set_ylabel('Precision')
        axes[1].legend(loc='upper right')
        axes[1].grid(alpha=0.2)

        # 3. Top-K Prioritization Gain Curve
        df_k = pd.DataFrame({'true': y_true, 'prob': y_probs}).sort_values(by='prob', ascending=False)
        cum_pos = np.cumsum(df_k['true'])
        total_pos = df_k['true'].sum()
        pct_reviewed = np.linspace(0, 100, len(df_k))
        pct_captured = (cum_pos / total_pos) * 100.0

        axes[2].plot(pct_reviewed, pct_captured, color='#8b5cf6', lw=2.5, label='StepWise Triage Priority')
        axes[2].plot([0, 100], [0, 100], color='#94a3b8', linestyle='--', label='Random Unprioritized Baseline')
        axes[2].scatter([20], [pct_captured.iloc[int(len(df_k)*0.2)]], color='#ef4444', s=80, zorder=5)
        axes[2].annotate(f"Top 20% Queue:\n{pct_captured.iloc[int(len(df_k)*0.2)]:.1f}% Progressors", 
                         (20, pct_captured.iloc[int(len(df_k)*0.2)]), textcoords="offset points", xytext=(15,-10),
                         arrowprops=dict(arrowstyle="->", color='#ef4444'), fontweight='bold')
        axes[2].set_title('Top-K Prioritization Capture Curve', fontsize=12, fontweight='bold')
        axes[2].set_xlabel('% of Patient Queue Reviewed')
        axes[2].set_ylabel('% of 24m Progressors Captured')
        axes[2].legend(loc='lower right')
        axes[2].grid(alpha=0.2)

        plt.tight_layout()
        out_path = os.path.join(self.viz_dir, "stage1_progression_comprehensive_eval.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved Comprehensive Evaluation Chart to {out_path}")

    def _plot_confusion_matrix(self, cm, cm_norm, class_names):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Raw Counts
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names, ax=ax1)
        ax1.set_title('Confusion Matrix (Raw Counts)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Predicted Label')
        ax1.set_ylabel('True Clinical Label')

        # Normalized %
        sns.heatmap(cm_norm * 100, annot=True, fmt='.1f', cmap='Greens', xticklabels=class_names, yticklabels=class_names, ax=ax2)
        ax2.set_title('Confusion Matrix (% Recall by Class)', fontsize=12, fontweight='bold')
        ax2.set_xlabel('Predicted Label')
        ax2.set_ylabel('True Clinical Label')

        plt.tight_layout()
        out_path = os.path.join(self.viz_dir, "stage1_confusion_matrix.png")
        plt.savefig(out_path, dpi=300)
        plt.close()
        print(f"Saved Confusion Matrix Plot to {out_path}")

if __name__ == "__main__":
    engine = Stage1XGBoostTriageEngine()
    df = engine.load_data()
    engine.train_progression_model(df)
    engine.train_3class_triage_model(df)
