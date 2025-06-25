from config_loader import ConfigLoader
from model_adapters import (
    RandomForestAdapter, 
    LogisticRegressionAdapter, 
    MLPClassifierAdapter, 
    SVMAdapter,
    AdaBoostAdapter,
    GradientBoostingAdapter
)
from model_manager import ModelManager
import numpy as np
import pandas as pd
import pickle
import os
import time
import argparse
from utilities import preprocess_data, split_data, preprocess_data_naive, split_data_clientTrajectory, scoring_with_bootstrap, cross_validation_scoring, get_target_column, save_results_to_file, print_cv_results, load_and_preprocess_data
from sklearn.pipeline import make_pipeline
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import make_scorer

# File paths
SCORING_FILE_PATH = """outputs/scorings/scorings_training_{model_name}_{dataset}.txt"""
SHAP_FILE_PATH = """outputs/shap_values/shap_explanation_{model_name}_{dataset}.pkl"""
LIME_FILE_PATH = """outputs/lime_values/lime_explanation_{model_name}_{dataset}.pkl"""
GRID_SEARCH_FILE_PATH = """outputs/best_params/grid_search_{model_name}_{dataset}.txt"""
MODEL_SAVE_PATH = """outputs/presaved_models/{model_name}_{dataset}_model.joblib"""

# Ensure output directories exist
os.makedirs(os.path.dirname(SCORING_FILE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(SHAP_FILE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(LIME_FILE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(GRID_SEARCH_FILE_PATH), exist_ok=True)
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)

def parse_args():
    parser = argparse.ArgumentParser(description='Run machine learning models on different datasets')
    parser.add_argument('--dataset', type=str, choices=['placement', 'trajectory'], required=True,
                      help='Which dataset to use: "placement" for demographics dataset or "trajectory" for client trajectory dataset')
    parser.add_argument('--model', type=str, choices=['rf', 'logreg', 'mlp', 'svm', 'adaboost', 'gb'],
                      help='Which model to use')
    parser.add_argument('--actions', type=str, help='Which actions to perform')
    return parser.parse_args()

def setup_models(config):
    """Setup and register models with best parameters from config."""
    print("Registering models...")
    manager = ModelManager()
    
    # Create adapters with best params from config
    tree_params = config.get_best_params('RandomForest')
    rf_adapter = RandomForestAdapter(**tree_params)
    logreg_params = config.get_best_params('LogisticRegression')
    logreg_adapter = LogisticRegressionAdapter(**logreg_params)
    mlp_params = config.get_best_params('MLPClassifier')
    mlp_adapter = MLPClassifierAdapter(**mlp_params)
    svm_params = config.get_best_params('SVM')
    svm_adapter = SVMAdapter(**svm_params)
    adaboost_params = config.get_best_params('AdaBoost')
    adaboost_adapter = AdaBoostAdapter(**adaboost_params)
    gb_params = config.get_best_params('GradientBoosting')
    gb_adapter = GradientBoostingAdapter(**gb_params)

    # Register models
    print("Registering models...")
    manager = ModelManager()
    manager.register('rf', rf_adapter)
    manager.register('logreg', logreg_adapter)
    manager.register('mlp', mlp_adapter)
    manager.register('svm', svm_adapter)
    manager.register('adaboost', adaboost_adapter)
    manager.register('gb', gb_adapter)

    
    return manager

def process_with_model(manager, model_name, X_train, X_test, y_train, y_test, preprocessed_ds, args, 
                      save_model=False, do_predict=False, do_gridsearch=False, do_shap=False, do_lime=False):
    """Process model with specified actions."""
    if model_name not in manager.models:
        print(f"Model '{model_name}' is not registered. Available: {list(manager.models.keys())}")
        return
    
    adapter = manager.get(model_name)
    
    if do_predict:
        print(f"Performing cross-validation for {model_name}...")

        start = time.time()
        confident_intervals = scoring_with_bootstrap(adapter, X_train, y_train, X_train, y_train, n_bootstrap = 50)
        end = time.time()
        execution_time = end - start
        
        # Save and print results
        save_results_to_file(
            SCORING_FILE_PATH.format(model_name=model_name, dataset=args.dataset),
            confident_intervals, execution_time
        )

        print(confident_intervals)
        print(f"Time took for do_predict: {execution_time:.4f} seconds")

    if save_model:
        adapter.fit(X_train, y_train)
        adapter.save(MODEL_SAVE_PATH.format(model_name=model_name, dataset=args.dataset))
        adapter.load(MODEL_SAVE_PATH.format(model_name=model_name, dataset=args.dataset))

    if do_gridsearch:
        param_grid = config.get_param_grid({
            'rf': 'RandomForest',
            'logreg': 'LogisticRegression',
            'mlp': 'MLPClassifier',
            'svm': 'SVM',
            'adaboost': 'AdaBoost',
            'gb': 'GradientBoosting'
        }[model_name])
        best_params, best_score = adapter.grid_search(X_train, y_train, param_grid)
        with open(GRID_SEARCH_FILE_PATH.format(model_name=model_name, dataset=args.dataset), "w") as f:
            f.write(str(best_params) + '\n')
            f.write(str(best_score) + '\n')
        print(f'Best {model_name} params:', best_params)
        print(f'Best {model_name} score:', best_score)

    if do_shap:
        print(f"Calculating SHAP values for {model_name}...")
        try:
            shap_type = 'logistic' if model_name in ['logreg', 'svm'] else 'tree' if model_name in ['rf', 'adaboost', 'gb'] else 'mlp'
            shap_values = adapter.shap_values(X_train, X_test, shap_type)
            with open(SHAP_FILE_PATH.format(model_name=model_name, dataset=args.dataset), "wb") as f:
                pickle.dump(shap_values, f)
            print(f'SHAP values shape for {model_name}:', shap_values.values.shape)
        except Exception as e:
            print(f'SHAP calculation failed for {model_name}:', e)

    if do_lime:
        print(f"Calculating LIME values for {model_name}...")
        try:
            lime_explanations = adapter.lime_values(X_train, X_test, num_features=10, num_samples=5)
            with open(LIME_FILE_PATH.format(model_name=model_name), "wb") as f:
                pickle.dump(lime_explanations, f)
            print(f'LIME explanations calculated for {model_name} for {len(lime_explanations)} samples')
        except Exception as e:
            print(f'LIME calculation failed for {model_name}:', e)

def get_user_inputs(args):
    """Get model name and actions from args or user input."""
    model_name = args.model or input("Enter model name to process (rf, logreg, mlp, svm, adaboost, gb): ").strip()
    
    print("Select actions to perform:")
    print("1. Save model")
    print("2. Prediction (fit, predict, score, save)")
    print("3. Grid Search")
    print("4. SHAP values")
    print("5. LIME values")
    print("6. All")
    
    actions = args.actions or input("Enter numbers separated by comma (e.g. 1,2,3,4,5,6): ").strip()
    
    return model_name, actions

def main():
    """Main execution function."""
    args = parse_args()
    
    # Load and preprocess data
    X_train, X_test, y_train, y_test, preprocessed_ds = load_and_preprocess_data(args.dataset)
    
    # Setup configuration and models
    config = ConfigLoader('model_config.yaml')
    manager = setup_models(config)
    
    # Get user inputs
    model_name, actions = get_user_inputs(args)
    
    # Parse actions
    save_model = '1' in actions or '6' in actions
    do_predict = '2' in actions or '6' in actions
    do_gridsearch = '3' in actions or '6' in actions
    do_shap = '4' in actions or '6' in actions
    do_lime = '5' in actions or '6' in actions
    
    # Process model
    process_with_model(
        manager, model_name, X_train, X_test, y_train, y_test, preprocessed_ds, args,
        save_model=save_model, do_predict=do_predict, 
        do_gridsearch=do_gridsearch, do_shap=do_shap, do_lime=do_lime
    )

if __name__ == "__main__":
    main()
    