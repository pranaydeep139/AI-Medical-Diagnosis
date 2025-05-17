# ml_training/conceptual_train_model.py
import pandas as pd
import numpy as np
import json
import joblib # For saving the model
from sklearn.preprocessing import LabelBinarizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsRestClassifier
from collections import defaultdict
import os

ARTIFACTS_DIR = 'app_artifacts' # Directory to save artifacts for Streamlit app
MODEL_OUTPUT_DIR = 'aml_model_files' # Directory for potential AML deployment files

def train_and_save_all(data_path='Training.csv', artifacts_output_dir=ARTIFACTS_DIR, model_output_dir=MODEL_OUTPUT_DIR):
    if not os.path.exists(artifacts_output_dir):
        os.makedirs(artifacts_output_dir)
    if not os.path.exists(model_output_dir):
        os.makedirs(model_output_dir)

    print(f"Loading data from {data_path}...")
    try:
        df = pd.read_csv(data_path)
        df = df.drop(columns=['Unnamed: 132', 'fluid_overload', 'fluid_overload.1'], errors='ignore') # Drop if exists
        print(f"Data loaded. Shape: {df.shape}")
    except FileNotFoundError:
        print(f"Error: {data_path} not found. Please ensure Training.csv is in the same directory or provide the correct path.")
        return

    print("Preprocessing data...")
    numerical_cols = df.select_dtypes(include=np.number).columns.tolist()
    binary_cols = [col for col in numerical_cols if df[col].nunique() == 2 and col != 'prognosis']
    
    if 'prognosis' in binary_cols: # Should not happen if 'prognosis' isn't just 0/1
        binary_cols.remove('prognosis')

    cols_to_keep = binary_cols + ['prognosis']
    df_processed = df[cols_to_keep].copy()

    for col in binary_cols:
        if df_processed[col].isnull().any():
            df_processed[col] = df_processed[col].fillna(0)

    if 'prognosis' not in df_processed.columns:
        print("Error: 'prognosis' column missing in the processed data.")
        return
    df_processed.dropna(subset=['prognosis'], inplace=True)

    if df_processed.empty or len(df_processed['prognosis'].unique()) < 2:
        print("Error: Dataset empty or lacks variety after processing.")
        return

    y = df_processed['prognosis']
    lb = LabelBinarizer()
    Y = lb.fit_transform(y)
    disease_names = lb.classes_.tolist()

    X = df_processed[binary_cols]
    original_feature_order = X.columns.tolist() 

    if X.empty or len(Y) == 0:
        print("Error: Issue preparing symptom data (X) or outcomes (Y).")
        return

    print("Training model...")
    base_rf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
    ovr_model = OneVsRestClassifier(base_rf, n_jobs=-1)
    ovr_model.fit(X, Y)
    print("Model training complete.")

    # --- Save Model for potential Azure ML ---
    aml_model_save_path = os.path.join(model_output_dir, "model.pkl")
    joblib.dump(ovr_model, aml_model_save_path)
    print(f"Trained model for AML saved to {aml_model_save_path}")
    with open(os.path.join(model_output_dir, 'model_binary_cols.json'), 'w') as f:
        json.dump(original_feature_order, f)
    with open(os.path.join(model_output_dir, 'model_disease_names.json'), 'w') as f:
        json.dump(disease_names, f)
    print(f"Supporting files for AML model saved in {model_output_dir}/")


    print("Calculating feature importances for app logic...")
    importances = pd.DataFrame(
        {disease: clf.feature_importances_ for disease, clf in zip(disease_names, ovr_model.estimators_)},
        index=binary_cols # Use binary_cols as it matches X's columns
    ).T
    thresh = importances.quantile(0.90, axis=1)
    key_features = {disease: importances.columns[importances.loc[disease] > thresh[disease]].tolist()
                    for disease in disease_names}
    disease_to_feats = {disease: list(feats) for disease, feats in key_features.items()}

    symptom_to_diseases_map_defaultdict = defaultdict(list)
    for disease, feats_list in disease_to_feats.items():
        for feat in feats_list:
            symptom_to_diseases_map_defaultdict[feat].append(disease)
    symptom_to_diseases_map = dict(symptom_to_diseases_map_defaultdict)

    # --- Save Artifacts for Streamlit App ---
    print(f"Saving artifacts for Streamlit app to {artifacts_output_dir}/")
    
    # Save model.pkl also to artifacts_output_dir for Streamlit app
    streamlit_model_save_path = os.path.join(artifacts_output_dir, "model.pkl")
    joblib.dump(ovr_model, streamlit_model_save_path)
    print(f"Trained model for Streamlit app saved to {streamlit_model_save_path}")

    with open(os.path.join(artifacts_output_dir, 'model_binary_cols.json'), 'w') as f:
        json.dump(original_feature_order, f)
    with open(os.path.join(artifacts_output_dir, 'model_disease_names.json'), 'w') as f:
        json.dump(disease_names, f)
    with open(os.path.join(artifacts_output_dir, 'model_disease_to_feats.json'), 'w') as f:
        json.dump(disease_to_feats, f)
    with open(os.path.join(artifacts_output_dir, 'model_symptom_to_diseases.json'), 'w') as f:
        json.dump(symptom_to_diseases_map, f) # Save the plain dict
    
    symptom_defaults = {col: 0 for col in original_feature_order}
    with open(os.path.join(artifacts_output_dir, 'symptom_defaults.json'), 'w') as f:
        json.dump(symptom_defaults, f)
    
    print("All artifacts saved.")

if __name__ == '__main__':
    if not os.path.exists('Training.csv'):
        print("Error: Training.csv not found in the current directory.")
        print("Please place Training.csv here or update the path in train_and_save_all().")
    else:
        train_and_save_all()
        print("\n--- Conceptual Training and Artifact Generation Complete ---")
        print(f"Streamlit artifacts (including model.pkl) are in: ./{ARTIFACTS_DIR}")
        print(f"Files for potential Azure ML model deployment are in: ./{MODEL_OUTPUT_DIR}")