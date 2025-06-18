import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import os
import sys
import time
from sklearn.metrics import mean_squared_error, r2_score

def prepare_trained_model(X_train_path: str, y_train_path: str, optuna_results_path: str, output_model_path: str):
    """
    Charge les résultats d'Optuna et prépare le modèle final pour l'évaluation.
    
    Args:
        X_train_path (str): Chemin vers X_train_scaled.csv
        y_train_path (str): Chemin vers y_train.csv
        optuna_results_path (str): Chemin vers les résultats Optuna
        output_model_path (str): Chemin pour sauvegarder le modèle final
    """
    
    print("Chargement des données d'entraînement...")
    try:
        X_train = pd.read_csv(X_train_path)
        y_train = pd.read_csv(y_train_path).squeeze()
        
        print(f"X_train shape: {X_train.shape}")
        print(f"y_train shape: {y_train.shape}")
        print(f"Features: {list(X_train.columns)}")
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        sys.exit(1)
    
    print("\nChargement des résultats d'optimisation Optuna...")
    try:
        optuna_results = joblib.load(optuna_results_path)
        
        model = optuna_results['final_model']
        best_params = optuna_results['best_params']
        final_metrics = optuna_results['final_metrics']
        
        print("Résultats Optuna chargés avec succès:")
        print(f"Temps d'optimisation: {optuna_results['optimization_time']:.2f} secondes")
        print(f"Meilleur RMSE trouvé: {optuna_results['best_rmse']:.6f}")
        
        print(f"\nMeilleurs hyperparamètres:")
        for param, value in best_params.items():
            print(f"  {param}: {value}")
        
        print(f"\nMétriques sur les données d'entraînement:")
        print(f"  - RMSE: {final_metrics['rmse']:.6f}")
        print(f"  - R²: {final_metrics['r2']:.6f}")
        
    except Exception as e:
        print(f"Erreur lors du chargement des résultats Optuna: {e}")
        sys.exit(1)
    
    print("\nVérification du modèle...")
    try:
        dtrain = xgb.DMatrix(X_train)
        
        y_pred = model.predict(dtrain)
        
        train_mse = mean_squared_error(y_train, y_pred)
        train_rmse = np.sqrt(train_mse)
        train_r2 = r2_score(y_train, y_pred)
        
        print(f"Vérification des métriques:")
        print(f"  - MSE: {train_mse:.6f}")
        print(f"  - RMSE: {train_rmse:.6f}")
        print(f"  - R²: {train_r2:.6f}")
        
        residuals = y_train - y_pred
        print(f"\nStatistiques des résidus:")
        print(f"  - Moyenne: {residuals.mean():.6f}")
        print(f"  - Std: {residuals.std():.6f}")
        print(f"  - Min: {residuals.min():.6f}")
        print(f"  - Max: {residuals.max():.6f}")
        
    except Exception as e:
        print(f"Erreur lors de la vérification: {e}")
        sys.exit(1)
    
    model_info = {
        'model': model,  
        'model_type': 'XGBoost Native API',
        'optimization_method': 'Optuna Bayesian Optimization',
        'best_params': best_params,
        'final_params': optuna_results['final_params'],
        'train_metrics': {
            'mse': float(train_mse),
            'rmse': float(train_rmse),
            'r2': float(train_r2)
        },
        'optuna_info': {
            'best_rmse': optuna_results['best_rmse'],
            'optimization_time': optuna_results['optimization_time']
        },
        'feature_names': list(X_train.columns),
        'training_samples': len(X_train),
        'validation_split': optuna_results.get('val_size', 0.2)
    }
    
    os.makedirs(os.path.dirname(output_model_path), exist_ok=True)
    
    print(f"\nSauvegarde du modèle final...")
    try:
        joblib.dump(model_info, output_model_path)
        print(f"Modèle sauvegardé dans: {output_model_path}")
        
        file_size = os.path.getsize(output_model_path) / (1024 * 1024)  # MB
        print(f"Taille du fichier: {file_size:.2f} MB")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        sys.exit(1)
    
    print("\nPréparation du modèle terminée avec succès!")
    
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ DU MODÈLE ENTRAÎNÉ")
    print(f"{'='*60}")
    print(f"Méthode d'optimisation: Optuna (Bayesian)")
    print(f"Temps d'optimisation: {optuna_results['optimization_time']:.1f}s")
    print(f"Échantillons d'entraînement: {len(X_train)}")
    print(f"Features: {len(X_train.columns)}")
    print(f"RMSE final: {train_rmse:.6f}")
    print(f"R² final: {train_r2:.6f}")
    print(f"{'='*60}")
    
    return model


if __name__ == "__main__":
    X_train_file = "data/processed/X_train_scaled.csv"
    y_train_file = "data/processed/y_train.csv"
    optuna_results_file = "models/best_params.pkl"
    output_model_file = "models/trained_model.pkl"
    
    print("=== PRÉPARATION DU MODÈLE ENTRAÎNÉ ===")
    print(f"Fichier X_train: {X_train_file}")
    print(f"Fichier y_train: {y_train_file}")
    print(f"Résultats Optuna: {optuna_results_file}")
    print(f"Modèle de sortie: {output_model_file}")
    
    prepare_trained_model(X_train_file, y_train_file, optuna_results_file, output_model_file)
    
    print("=== SCRIPT TERMINÉ ===")