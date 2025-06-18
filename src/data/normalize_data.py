import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import joblib
import os
import sys

def normalize_data(train_path: str, test_path: str, output_dir: str, scaler_path: str = "models/scaler.pkl"):
    print("Chargement des données...")
    try:
        X_train = pd.read_csv(train_path)
        X_test = pd.read_csv(test_path)
        
        print(f"X_train shape: {X_train.shape}")
        print(f"X_test shape: {X_test.shape}")
        print(f"Features: {list(X_train.columns)}")
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        sys.exit(1)
    
    print("\nAnalyse des données avant normalisation:")
    print("X_train - Statistiques descriptives:")
    print(X_train.describe())
    
    print("\nInitialisation et ajustement du StandardScaler...")
    scaler = StandardScaler()
    
    scaler.fit(X_train)
    
    print("Paramètres du scaler:")
    print(f"  Moyennes: {scaler.mean_}")
    print(f"  Écarts-types: {scaler.scale_}")
    
    print("\nTransformation des données...")
    X_train_scaled = scaler.transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Conversion en DataFrame pour conserver les noms de colonnes
    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)
    
    print(f"X_train_scaled shape: {X_train_scaled_df.shape}")
    print(f"X_test_scaled shape: {X_test_scaled_df.shape}")
    
    print("\nVérification de la normalisation (données d'entraînement):")
    print("Moyennes :")
    print(X_train_scaled_df.mean())
    print("\nÉcarts-types :")
    print(X_train_scaled_df.std())
    
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
    
    print("\nSauvegarde des données normalisées...")
    try:
        X_train_scaled_df.to_csv(os.path.join(output_dir, 'X_train_scaled.csv'), index=False)
        X_test_scaled_df.to_csv(os.path.join(output_dir, 'X_test_scaled.csv'), index=False)
        
        print("Données normalisées sauvegardées avec succès")
        print(f"  - X_train_scaled.csv: {X_train_scaled_df.shape}")
        print(f"  - X_test_scaled.csv: {X_test_scaled_df.shape}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des données: {e}")
        sys.exit(1)
    
    print("\nSauvegarde du scaler...")
    try:
        joblib.dump(scaler, scaler_path)
        print(f"Scaler sauvegardé dans: {scaler_path}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde du scaler: {e}")
        sys.exit(1)
    
    print("\nNormalisation terminée avec succès")


if __name__ == "__main__":
    # Paramètres par défaut
    train_file = "data/processed/X_train.csv"
    test_file = "data/processed/X_test.csv"
    output_directory = "data/processed"
    scaler_file = "models/scaler.pkl"
    
    print("=== NORMALISATION DES DONNÉES ===")
    print(f"Fichier X_train: {train_file}")
    print(f"Fichier X_test: {test_file}")
    print(f"Répertoire de sortie: {output_directory}")
    print(f"Scaler: {scaler_file}")
    
    normalize_data(train_file, test_file, output_directory, scaler_file)
    
    print("=== SCRIPT TERMINÉ ===")