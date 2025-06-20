"""""
Preprocessing steps:
- Train a model for step 1 
- (Don't need to model for step 2 because the decision process is made using Xianquan's algorithm, 
just need to input m_risk and occupancy status)
- Prepare the data used in the main program, including a smaller Recidivism dataset (ouputs an m_risk) and an occupancy 
status (ocp_jl, ocp_wr, ocp_hd)
"""

import pandas as pd
import pickle

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.metrics import roc_auc_score


def halvingGridSearch(rf, X_train, y_train):
    param_grid = {'max_depth': [ 5, 10, 15, 20, 22], 'min_samples_split': [ 10, 15, 20, 25, 30], 
              'bootstrap': [True,False], 
              'min_samples_leaf': [4,6,8,10] }
    sh = HalvingGridSearchCV(rf, param_grid , cv=5, factor=2, min_resources=100, scoring='roc_auc').fit(X_train,y_train)
    return sh.best_params_

def recidivism(myDF, search):
    y = myDF['Recidivism']
    X = myDF.drop(['Recidivism'], axis=1) 
    X_train, X_test, y_train, y_test = train_test_split( X, y, train_size = 0.7, random_state = 51)
    ## HalvingGridSearch
    if search:
        dummy_rf = RandomForestClassifier(random_state=50)
        params = halvingGridSearch(dummy_rf, X_train, y_train)
        rf = RandomForestClassifier(max_depth = params['max_depth'], min_samples_split = params['min_samples_split'],
                                    bootstrap = params['boostrap'], min_samples_leaf = params['min_samples_leaf'], random = 50)
        rf.fit(X_train, y_train)
    else: 
        rf = RandomForestClassifier(max_depth=10, min_samples_split=30, n_estimators=1500, bootstrap=False, criterion = 'gini',
                            min_samples_leaf=6, max_features='sqrt', random_state = 50)
        rf.fit(X_train, y_train)
    return (rf, roc_auc_score(y_test, rf.predict_proba(X_test)[:,1]) )

def split_data(myDF): #works with Recidivism data only
    y = myDF['Recidivism']
    X = myDF.drop(['Recidivism'], axis=1) 
    X_train, X_holdout, y_train, y_holdout = train_test_split( X, y, train_size = 0.8, random_state = 50)
    X_train['Recidivism'] = y_train
    X_holdout['Recidivism'] = y_holdout
    return (X_train, X_holdout) #working data, holdout data

def train_and_save_model_stage1():
    ...    

def train_and_save_model_stage2():
    ...


if __name__ == "__main__": 
    recidivism_data = pd.read_csv("dataset/clientTrajectory_JailUpdate.csv")
    recidivism_data.drop(['WorkReleaseArrival', 'WorkReleaseDeparture','MonitoringArrival',
                    'MonitoringDeparture','HomeArrival', 'HomeDeparture', 'ArrivalTime',
                    'JailDeparture','JailArrival', 'ProfileID','DNACollected_TRUE'], axis = 1, inplace = True)
    working_data, holdout_data = split_data(recidivism_data)
    holdout_data.to_csv("recidivism_holdout_data.csv")

    rf_model, performance_score = recidivism(working_data, False)
    print(performance_score)
    file_name = 'finalized_recidivism_rf.sav'
    pickle.dump(rf_model, open(file_name, 'wb'))

    #placement1_data = pd.read_csv("dataset/placementDecisions_rl_los_cost_1.csv")
    #placement_data = pd.read_csv("placement-decions-different-costs/combined_placementDecisions_rl_los_cost.csv")

