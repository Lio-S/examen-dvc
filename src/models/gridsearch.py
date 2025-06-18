import pandas as pd
import numpy as np
import xgboost as xgb
import optuna
import optuna.exceptions
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os
import sys
import time

def objective(trial, X_train, y_train, X_val, y_val):
    """
    Définit l'espace de recherche des hyperparamètres.
    
    Args:
        trial: Objet trial d'Optuna
        X_train, y_train: Données d'entraînement
        X_val, y_val: Données de validation
    
    Returns:
        float: Score RMSE à minimiser
    """
    
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'eta': trial.suggest_float('eta', 0.01, 0.3, log=True),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 1, 10),
        'min_child_weight' : trial.suggest_int('min_child_weight', 3, 10),
        'random_state': 42,
        'seed': 42,
        'verbosity': 0 
    }
    
    n_estimators = trial.suggest_int('n_estimators', 200, 600)
    
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    try:
        callbacks = [
            xgb.callback.EarlyStopping(rounds=20, save_best=True, maximize=False)
        ]
        
        model = xgb.train(
            params,
            dtrain,
            num_boost_round=n_estimators,
            evals=[(dtrain, 'train'), (dval, 'val')],
            callbacks=callbacks,
            verbose_eval=False
        )
        
        y_pred = model.predict(dval)
        
        rmse = np.sqrt(mean_squared_error(y_val, y_pred))
        
        # Report du score intermédiaire pour le pruning Optuna
        trial.report(rmse, model.num_boosted_rounds())
        
        # Vérification si l'essai doit être arrêté (pruning)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
        
        return rmse
        
    except optuna.exceptions.TrialPruned:
        raise  # Re-raise pour Optuna
    except Exception as e:
        print(f"Erreur dans l'essai {trial.number}: {e}")
        return float('inf')


def optuna_gridsearch(X_train_path: str, y_train_path: str, output_path: str, 
                     n_trials: int = 500, val_size: float = 0.2):
    """
    Effectue une optimisation bayésienne avec Optuna pour XGBoost.
    
    Args:
        X_train_path (str): Chemin vers X_train_scaled.csv
        y_train_path (str): Chemin vers y_train.csv
        output_path (str): Chemin pour sauvegarder les meilleurs paramètres
        n_trials (int): Nombre d'essais Optuna
        val_size (float): Proportion des données pour la validation
    """
    
    print("Chargement des données d'entraînement...")
    try:
        X_train_full = pd.read_csv(X_train_path)
        y_train_full = pd.read_csv(y_train_path).squeeze()
        
        print(f"Données complètes - X shape: {X_train_full.shape}, y shape: {y_train_full.shape}")
        print(f"Features: {list(X_train_full.columns)}")
        print(f"Target stats: min={y_train_full.min():.3f}, max={y_train_full.max():.3f}, mean={y_train_full.mean():.3f}")
        
    except Exception as e:
        print(f"Erreur lors du chargement des données: {e}")
        sys.exit(1)
    
    print(f"\nDivision train/validation (validation = {val_size*100}%)...")
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full, y_train_full,
        test_size=val_size,
        random_state=42
    )
    
    print(f"Train set: X={X_train.shape}, y={y_train.shape}")
    print(f"Validation set: X={X_val.shape}, y={y_val.shape}")
    
    print(f"\nConfiguration de l'étude Optuna avec {n_trials} essais...")
    
    study = optuna.create_study(
        direction='minimize',
        study_name='xgboost_optimization',
        sampler=optuna.samplers.TPESampler(seed=42),  # Algorithme bayésien
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10, n_warmup_steps=5)  # Pruning des essais non prometteurs
    )
    
    objective_with_data = lambda trial: objective(trial, X_train, y_train, X_val, y_val)
    
    print(f"\nDémarrage de l'optimisation Optuna...")
    start_time = time.time()
    
    try:
        study.optimize(
            objective_with_data,
            n_trials=n_trials,
            show_progress_bar=True,
            callbacks=[
                lambda study, trial: print(f"Essai {trial.number}: RMSE = {trial.value:.6f}")
                if trial.number % 20 == 0 else None
            ]
        )
        
        end_time = time.time()
        print(f"\nOptimisation terminée en {(end_time - start_time):.2f} secondes")
        
    except Exception as e:
        print(f"Erreur lors de l'optimisation: {e}")
        sys.exit(1)
    
    print("\n=== RÉSULTATS DE L'OPTIMISATION OPTUNA ===")
    print(f"Nombre d'essais terminés: {len(study.trials)}")
    print(f"Meilleur RMSE: {study.best_value:.6f}")
    
    print("\nMeilleurs hyperparamètres:")
    for param, value in study.best_params.items():
        print(f"  {param}: {value}")
    
    completed_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]
    
    if len(completed_trials) > 1:
        values = [t.value for t in completed_trials]
        print(f"\nStatistiques des essais:")
        print(f"  - Meilleur RMSE: {min(values):.6f}")
        print(f"  - RMSE médian: {np.median(values):.6f}")
        print(f"  - RMSE moyen: {np.mean(values):.6f}")
        print(f"  - Écart-type: {np.std(values):.6f}")
    
    print(f"\nEntraînement final avec les meilleurs paramètres...")
    
    best_params = study.best_params.copy()
    n_estimators = best_params.pop('n_estimators')
    
    final_params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'random_state': 42,
        'seed': 42,
        'verbosity': 0,
        **best_params 
    }
    
    dtrain_full = xgb.DMatrix(X_train_full, label=y_train_full)
    
    final_model = xgb.train(
        final_params,
        dtrain_full,
        num_boost_round=n_estimators
    )
    
    y_train_pred = final_model.predict(dtrain_full)
    final_rmse = np.sqrt(mean_squared_error(y_train_full, y_train_pred))
    final_r2 = r2_score(y_train_full, y_train_pred)
    
    print(f"Métriques finales sur toutes les données d'entraînement:")
    print(f"  - RMSE: {final_rmse:.6f}")
    print(f"  - R²: {final_r2:.6f}")
    
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    print(f"\nSauvegarde des meilleurs paramètres...")
    try:
        optuna_results = {
            'best_params': study.best_params,
            'best_rmse': study.best_value,
            'final_params': final_params,
            'final_model': final_model,
            'final_metrics': {
                'rmse': final_rmse,
                'r2': final_r2
            },
            'feature_names': list(X_train_full.columns),
            'n_trials': len(completed_trials),
            'optimization_time': end_time - start_time,
            'val_size': val_size
        }
        
        joblib.dump(optuna_results, output_path)
        print(f"Résultats Optuna sauvegardés dans: {output_path}")
        
    except Exception as e:
        print(f"Erreur lors de la sauvegarde: {e}")
        sys.exit(1)
    
    print("\nOptimisation Optuna terminée avec succès!")
    return study.best_params


if __name__ == "__main__":
    X_train_file = "data/processed/X_train_scaled.csv"
    y_train_file = "data/processed/y_train.csv"
    output_file = "models/best_params.pkl"
    
    print(f"Fichier X_train: {X_train_file}")
    print(f"Fichier y_train: {y_train_file}")
    print(f"Sortie: {output_file}")
    
    optuna_gridsearch(X_train_file, y_train_file, output_file)
    
    print("=== SCRIPT TERMINÉ ===")