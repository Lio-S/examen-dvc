import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os
import sys

def split_data(input_path: str, output_dir: str, test_size: float = 0.2, random_state: int = 42):    
    print("Chargement des données brutes...")
    try:
        df = pd.read_csv(input_path)
        print(f"Données chargées avec succès. Shape: {df.shape}")
        
        print(f"Colonnes disponibles: {list(df.columns)}")
        print(f"Types de données:\n{df.dtypes}")
        print(f"Valeurs manquantes: {df.isnull().sum().sum()}")
        print(f"Doublons: {df.duplicated().sum()}")
        print(f"Suppression colonne date..")
        df = df.drop(columns = "date")
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        sys.exit(1)
    
    print("Séparation des features et de la variable cible...")
    
    target_column = df.columns[-1]  # silica_concentrate
    print(f"Variable cible: {target_column}")
    
    X = df.iloc[:, :-1]  # Toutes les colonnes sauf la dernière
    y = df.iloc[:, -1]   # Dernière colonne (silica_concentrate)
    
    print(f"Features shape: {X.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Features: {list(X.columns)}")
    
    print(f"Division des données (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=None  
    )
    
    print(f"   - X_train: {X_train.shape}")
    print(f"   - X_test: {X_test.shape}")
    print(f"   - y_train: {y_train.shape}")
    print(f"   - y_test: {y_test.shape}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("Sauvegarde des datasets...")
    try:
        X_train.to_csv(os.path.join(output_dir, 'X_train.csv'), index=False)
        X_test.to_csv(os.path.join(output_dir, 'X_test.csv'), index=False)
        y_train.to_csv(os.path.join(output_dir, 'y_train.csv'), index=False)
        y_test.to_csv(os.path.join(output_dir, 'y_test.csv'), index=False)
        
        print("Sauvegarde terminée avec succès")
        print(f"Fichiers sauvegardés dans: {output_dir}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        sys.exit(1)


if __name__ == "__main__":
    input_file = "data/raw/raw.csv"
    output_directory = "data/processed"
    
    print("=== SPLIT DES DONNÉES ===")
    print(f"Fichier d'entrée: {input_file}")
    print(f"Répertoire de sortie: {output_directory}")
    
    split_data(input_file, output_directory)
    
    print("=== SCRIPT TERMINÉ ===")