import pandas as pd
import numpy as np
import xgboost as xgb
import joblib
import json
import os
import sys
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(X_test_path: str, y_test_path: str, model_path: str, predictions_output: str, metrics_output: str):    
    print("Chargement des données de test...")
    try:
        X_test = pd.read_csv(X_test_path)
        y_test = pd.read_csv(y_test_path).squeeze()  # squeeze pour convertir en Series
        
        print(f"X_test shape: {X_test.shape}")
        print(f"y_test shape: {y_test.shape}")
        print(f"Features: {list(X_test.columns)}")
        
        print(f"\nStatistiques de y_test:")
        print(f"  - Min: {y_test.min():.3f}")
        print(f"  - Max: {y_test.max():.3f}")
        print(f"  - Mean: {y_test.mean():.3f}")
        print(f"  - Std: {y_test.std():.3f}")
        
    except Exception as e:
        print(f"Erreur lors du chargement des données de test: {e}")
        sys.exit(1)
    
    print("\nChargement du modèle entraîné...")
    try:
        model_info = joblib.load(model_path)
        model = model_info['model']
        best_params = model_info['best_params']
        train_metrics = model_info['train_metrics']
        
        print("Modèle chargé avec succès!")
        print(f"Échantillons d'entraînement: {model_info['training_samples']}")
        print(f"Features utilisées: {len(model_info['feature_names'])}")
        
        print(f"\nMétriques d'entraînement:")
        print(f"  - RMSE: {train_metrics['rmse']:.6f}")
        print(f"  - R²: {train_metrics['r2']:.6f}")
        
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        sys.exit(1)
    
    print("\nGénération des prédictions sur les données de test...")
    try:
        dtest = xgb.DMatrix(X_test)

        y_pred = model.predict(dtest)
        
        print(f"Prédictions générées: {len(y_pred)}")
        print(f"Prédictions - Min: {y_pred.min():.3f}, Max: {y_pred.max():.3f}, Mean: {y_pred.mean():.3f}")
        
    except Exception as e:
        print(f"Erreur lors de la génération des prédictions: {e}")
        sys.exit(1)
    
    print("\nCalcul des métriques d'évaluation...")
    
    # Métriques 
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    residuals = y_test - y_pred
    mean_residual = residuals.mean()
    std_residual = residuals.std()
    
    mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
    
    var_y_test = np.var(y_test)
    explained_variance = 1 - (np.var(residuals) / var_y_test)
    
    print(f"Métriques d'évaluation sur les données de test:")
    print(f"  - MSE: {mse:.6f}")
    print(f"  - RMSE: {rmse:.6f}")
    print(f"  - MAE: {mae:.6f}")
    print(f"  - R²: {r2:.6f}")
    print(f"  - MAPE: {mape:.2f}%")
    print(f"  - Variance expliquée: {explained_variance:.6f}")
    
    print(f"\nAnalyse des résidus:")
    print(f"  - Moyenne des résidus: {mean_residual:.6f}")
    print(f"  - Std des résidus: {std_residual:.6f}")
    print(f"  - Min résidu: {residuals.min():.6f}")
    print(f"  - Max résidu: {residuals.max():.6f}")
    
    print(f"\nComparaison entraînement vs test:")
    print(f"  - RMSE train: {train_metrics['rmse']:.6f}")
    print(f"  - RMSE test:  {rmse:.6f}")
    rmse_diff = rmse - train_metrics['rmse']
    print(f"  - Différence: {rmse_diff:.6f}")
    print(f"  - R² train: {train_metrics['r2']:.6f}")
    print(f"  - R² test:  {r2:.6f}")
    r2_diff = train_metrics['r2'] - r2
    print(f"  - Différence: {r2_diff:.6f}")
    
    # Détection du sur/sous-apprentissage
    if rmse_diff > 0.1 or r2_diff > 0.1:
        print(f"\nAvertissement: Possible sur-apprentissage détecté")
        print(f"  - RMSE dégradation: {rmse_diff:.6f}")
        print(f"  - R² dégradation: {r2_diff:.6f}")
    else:
        print(f"\nModèle semble bien généralisé")
    
    predictions_df = pd.DataFrame({
        'y_true': y_test,
        'y_pred': y_pred,
        'residuals': residuals,
        'abs_residuals': np.abs(residuals),
        'relative_error': np.abs(residuals) / y_test * 100
    })
    
    predictions_df.index = y_test.index
    
    print(f"\nDataFrame des prédictions créé: {predictions_df.shape}")
    
    metrics_dict = {
        "model_type": "XGBoost Native API",
        "test_samples": int(len(y_test)),
        "features_count": int(len(X_test.columns)),
        "test_metrics": {
            "mse": float(mse),
            "rmse": float(rmse),
            "mae": float(mae),
            "r2_score": float(r2),
            "mape": float(mape),
            "explained_variance": float(explained_variance)
        },
        "residuals_analysis": {
            "mean_residual": float(mean_residual),
            "std_residual": float(std_residual),
            "min_residual": float(residuals.min()),
            "max_residual": float(residuals.max())
        },
        "train_vs_test": {
            "train_rmse": float(train_metrics['rmse']),
            "test_rmse": float(rmse),
            "rmse_difference": float(rmse_diff),
            "train_r2": float(train_metrics['r2']),
            "test_r2": float(r2),
            "r2_difference": float(r2_diff)
        },
        "target_statistics": {
            "min": float(y_test.min()),
            "max": float(y_test.max()),
            "mean": float(y_test.mean()),
            "std": float(y_test.std())
        },
        "best_params": best_params
    }
    
    os.makedirs(os.path.dirname(predictions_output), exist_ok=True)
    os.makedirs(os.path.dirname(metrics_output), exist_ok=True)
    
    print(f"\nSauvegarde des prédictions...")
    try:
        predictions_df.to_csv(predictions_output, index=False)
        print(f"Prédictions sauvegardées dans: {predictions_output}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des prédictions: {e}")
        sys.exit(1)
    
    print(f"\nSauvegarde des métriques...")
    try:
        with open(metrics_output, 'w') as f:
            json.dump(metrics_dict, f, indent=4)
        print(f"Métriques sauvegardées dans: {metrics_output}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde des métriques: {e}")
        sys.exit(1)
    
    print("\nÉvaluation terminée avec succès")
    
    print(f"\n{'='*60}")
    print(f"RÉSUMÉ DE L'ÉVALUATION")
    print(f"{'='*60}")
    print(f"Modèle: XGBoost (API native) + Optuna")
    print(f"Échantillons de test: {len(y_test)}")
    print(f"RMSE: {rmse:.6f}")
    print(f"R²: {r2:.6f}")
    print(f"MAPE: {mape:.2f}%")
    print(f"{'='*60}")


if __name__ == "__main__":
    X_test_file = "data/processed/X_test_scaled.csv"
    y_test_file = "data/processed/y_test.csv"
    model_file = "models/trained_model.pkl"
    predictions_file = "data/predictions.csv"
    metrics_file = "metrics/scores.json"
    
    print("=== ÉVALUATION DU MODÈLE ===")
    print(f"Fichier X_test: {X_test_file}")
    print(f"Fichier y_test: {y_test_file}")
    print(f"Modèle: {model_file}")
    print(f"Prédictions: {predictions_file}")
    print(f"Métriques: {metrics_file}")
    
    evaluate_model(X_test_file, y_test_file, model_file, predictions_file, metrics_file)
    
    print("=== SCRIPT TERMINÉ ===")