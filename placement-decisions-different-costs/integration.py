""""
General notes: This code includes two seperate parts.
     At the first stage, it predicts the Recidivism rate (m_risk) based on a demographic datset of people
     Then, it uses the rate and occupancy status to place the person to a suitable program

     - INPUT: 
     - OUTPUT:
"""

# Import packages
import numpy as np
import pandas as pd
import random
import pickle
from copy import deepcopy
import itertools
import glob

import sys
import os
from cc_cost_parameters import *
from cc_flow_parameters import *
from func_placement_decision import *

from sklearn.metrics import roc_auc_score

import warnings
warnings.filterwarnings("ignore")


if __name__ == '__main__':
    #Stage1
    holdout_data = pd.read_csv("recidivism_holdout_data.csv")
    loaded_model = pickle.load(open('finalized_recidivism_rf.sav', 'rb'))

    X_holdout = holdout_data.drop(['Recidivism'], axis = 1)
    y_holdout = holdout_data['Recidivism']

    person_1_m_risk = loaded_model.predict(X_holdout.head(1).drop(['Unnamed: 0'], axis = 1))[0]

    #Stage2
    m_risk = person_1_m_risk
    cur_occupancy = [400, 160, 350]
    print(f"the placement is {placement_onetype_based(m_risk, cur_occupancy)}")
    

