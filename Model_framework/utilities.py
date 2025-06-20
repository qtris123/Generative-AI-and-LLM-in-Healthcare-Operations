import pandas as pd
import numpy as np
from sklearn.utils import resample
from tqdm import tqdm

def preprocess_data(continuous_after: pd.DataFrame) -> pd.DataFrame:
    unknown_employment_index = continuous_after[ (continuous_after['employmentStatus_Full_Time'] == 0) & (continuous_after['employmentStatus_Part_Time'] == 0) & (continuous_after['employmentStatus_Unemployed'] == 0)].index
    unknown_employment_age = continuous_after[ (continuous_after['employmentStatus_Part_Time'] == 0) & (continuous_after['employmentStatus_Unemployed'] == 0) & (continuous_after['employmentStatus_Full_Time'] == 0) ]['age']
    unemployment_index = continuous_after[ continuous_after['employmentStatus_Unemployed'] == 1].index
    unknown_employment_index

    type1_index = list(unemployment_index) + list(unknown_employment_index)
    type2_index = list(continuous_after[ continuous_after['employmentStatus_Full_Time'] == 1].index) + list(continuous_after[ continuous_after['employmentStatus_Part_Time'] == 1].index)
    new_row = [0] * continuous_after.shape[0]
    for index in type1_index:
        new_row[index] = 1

    continuous_after['Employed'] = new_row
    continuous_after.drop(['employmentStatus_Part_Time', 'employmentStatus_Full_Time'], axis =1, inplace = True )

    new_row = [0] * continuous_after.shape[0]
    for index in type2_index:
        new_row[index] = 1
    
    continuous_after['Unemployed'] = new_row
    continuous_after.drop(['employmentStatus_Unemployed'], axis =1, inplace = True )
    continuous_after.drop(['Unemployed','ocp_wr_low_los', 'ocp_jl_low_los', 'ocp_hd_low_los','licenseStatus_Not_Suspended','licenseStatus_Suspended'], axis=1, inplace=True) 

    #Cost multiplication
    continuous_after['weighted_jl_total'] = continuous_after['c_occu_jail']*continuous_after['ocp_jl_total']
    continuous_after['weighted_wr_total'] = continuous_after['c_occu_wr']*continuous_after['ocp_wr_total']
    continuous_after['weighted_hd_total'] = continuous_after['c_occu_hd']*continuous_after['ocp_hd_total']
    continuous_after.drop(['c_occu_jail', 'c_occu_wr', 'c_occu_hd', 'ocp_jl_total', 'ocp_wr_total', 'ocp_hd_total'], axis = 1, inplace=True)

    # one hot encoding  -> the target is "placement", so we don't need to encode this column, also bcz RandomForest can handle categorical target data
    # work release = 0 | home detention = 1 | jail = 2
    for ind, item in continuous_after.iterrows():
        if item['placement'] == 'work release':
            continuous_after['placement'].iat[ind] = '0'
        elif item['placement'] == 'home detention':
            continuous_after['placement'].iat[ind] = '1'
        elif item['placement'] == 'jail':
            continuous_after['placement'].iat[ind] = '2'

    #object to int
    continuous_after['placement'] = pd.to_numeric(continuous_after['placement'])

    # drop imbalanced data #JUNE 5th: TUNR OFF THESE DROPPINGS
    #continuous_after.drop(['race_Other', 'registeredSexOffender_TRUE', 'violentOffender_TRUE', 'gangMember_x_TRUE', 'homeless_TRUE'], axis = 1, inplace = True)

    return continuous_after 


def preprocess_data_naive(continuous_after: pd.DataFrame) -> pd.DataFrame:
    #Cost multiplication
    continuous_after['weighted_jl_total'] = continuous_after['c_occu_jail']*continuous_after['ocp_jl_total']
    continuous_after['weighted_wr_total'] = continuous_after['c_occu_wr']*continuous_after['ocp_wr_total']
    continuous_after['weighted_hd_total'] = continuous_after['c_occu_hd']*continuous_after['ocp_hd_total']
    continuous_after.drop(['c_occu_jail', 'c_occu_wr', 'c_occu_hd', 'ocp_jl_total', 'ocp_wr_total', 'ocp_hd_total'], axis = 1, inplace=True)

    # one hot encoding  -> the target is "placement", so we don't need to encode this column, also bcz RandomForest can handle categorical target data
    # work release = 0 | home detention = 1 | jail = 2
    for ind, item in continuous_after.iterrows():
        if item['placement'] == 'work release':
            continuous_after['placement'].iat[ind] = '0'
        elif item['placement'] == 'home detention':
            continuous_after['placement'].iat[ind] = '1'
        elif item['placement'] == 'jail':
            continuous_after['placement'].iat[ind] = '2'

    #object to int
    continuous_after['placement'] = pd.to_numeric(continuous_after['placement'])

    return continuous_after

from sklearn.model_selection import train_test_split
def split_data(my_csv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = my_csv["placement"]
    X = my_csv.drop(["placement"], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split( X, y, train_size = 0.7, random_state = 50) 

    return X_train, X_test, y_train, y_test

def split_data_clientTrajectory(my_csv: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    y = my_csv["Recidivism"]
    X = my_csv.drop(["Recidivism"], axis = 1)

    X_train, X_test, y_train, y_test = train_test_split( X, y, train_size = 0.7, random_state = 50) 

    return X_train, X_test, y_train, y_test

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, roc_auc_score

def calculate_metrics(y_true, y_pred, y_proba):
    """Helper function to calculate metrics for a given prediction set."""
    metrics = {}
    
    # Calculate basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred)
    metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
    metrics['precision'] = precision_score(y_true, y_pred, average='weighted')
    metrics['recall'] = recall_score(y_true, y_pred, average='weighted')
    
    # Calculate ROC AUC if probabilities are available
    if y_proba is not None:
        try:
            if y_proba.ndim == 2:
                if y_proba.shape[1] == 2:  # Binary classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
                else:  # Multiclass classification
                    metrics['roc_auc'] = roc_auc_score(y_true, y_proba, multi_class='ovr')
            else:
                metrics['roc_auc'] = roc_auc_score(y_true, y_proba)
        except:
            metrics['roc_auc'] = None
    else:
        metrics['roc_auc'] = None
        
    return metrics


def scoring_with_bootstrap(adapter, X_train, y_train, X_test, y_test, n_bootstrap, confidence_level=0.95):
    """
    Calculate scoring metrics with bootstrap confidence intervals using different model predictions.
    
    Args:
        adapter: Model adapter instance
        X_train: Training features
        y_train: Training labels
        X_test: Test features
        y_test: Test labels
        n_bootstrap: Number of bootstrap samples
        confidence_level: Confidence level for intervals
    """
    # Bootstrap confidence intervals
    bootstrap_metrics = {'accuracy': [], 'f1': [], 'precision': [], 'recall': [], 'roc_auc': []}
    
    for i in tqdm(range(n_bootstrap), desc="Bootstrap iterations"):
        # Resample training data with replacement
        indices = resample(range(len(X_train)), n_samples=len(X_train), random_state=i)
        X_train_boot = X_train.iloc[indices] if hasattr(X_train, 'iloc') else X_train[indices]
        y_train_boot = y_train.iloc[indices] if hasattr(y_train, 'iloc') else y_train[indices]
        
        # Fit model on bootstrap sample
        adapter.fit(X_train_boot, y_train_boot)
        
        # Make predictions on test set
        y_pred_boot = adapter.predict(X_test)
        try:
            y_proba_boot = adapter.get_model().predict_proba(X_test)
        except AttributeError:
            y_proba_boot = None
        
        # Calculate metrics for this bootstrap sample
        boot_metrics = calculate_metrics(y_test, y_pred_boot, y_proba_boot)
        
        # Store results
        for metric, value in boot_metrics.items():
            if value is not None:
                bootstrap_metrics[metric].append(value)
    
    # Calculate confidence intervals and use average as main value
    alpha = 1 - confidence_level
    lower_percentile = (alpha / 2) * 100
    upper_percentile = (1 - alpha / 2) * 100
    
    confidence_intervals = {}
    for metric in bootstrap_metrics.keys():
        if bootstrap_metrics[metric]:  # Only if we have valid bootstrap samples
            # Use average of bootstrap samples as the main value
            avg_value = np.mean(bootstrap_metrics[metric])
            ci_lower = np.percentile(bootstrap_metrics[metric], lower_percentile)
            ci_upper = np.percentile(bootstrap_metrics[metric], upper_percentile)
            confidence_intervals[metric] = {
                'value': avg_value,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'std': np.std(bootstrap_metrics[metric])
            }
        else:
            confidence_intervals[metric] = {
                'value': None,
                'ci_lower': None,
                'ci_upper': None,
                'std': None
            }
    
    return confidence_intervals

if __name__ == "__main__":
    my_ds_raw = pd.read_csv("../Generative-AI-and-LLM-in-Healthcare-Operations/dataset/demographics_to_placement_simulation_WR_no_cap_continuous_2.csv")
    print(my_ds_raw.columns)
    my_ds = preprocess_data(my_ds_raw)
    X_train, X_test, y_train, y_test = split_data(my_ds)
    print(X_train.head())
    print(X_test.head())

