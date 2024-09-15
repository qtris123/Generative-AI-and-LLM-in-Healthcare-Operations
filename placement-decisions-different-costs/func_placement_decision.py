""""
Community Corrections Project - CC Simulation Functions

Author: Xiaoquan Gao
Last Updated: 9/14/2024
"""

"""
General notes: This code is built for the process flow simulation based on the RL-based heuristic

- INPUT: 
- OUTPUT: 
"""

# Import packages
import math
import numpy as np
import pandas as pd
import random
import pickle
from copy import deepcopy
import itertools
import glob

from cc_flow_parameters import *
from cc_cost_parameters import *

import warnings
warnings.filterwarnings("ignore")

N_LosTypes=2

# placement decision for each new arrival
def placement_onetype_based(m_risk, occupancy):
    conj_rcdvm,  conj_vio = congestion_based_risks(occupancy)
    cost = {}
    for j_prog in range(N_Programs):
        '''One-step Cost'''
        ocp_new = [occupancy[j] + (j_prog == j) for j in range(N_Programs)]
        Cost_ocp = ocp_cost_lin(ocp_new) - ocp_cost_lin(occupancy)
        # Recidivism Cost
        Cost_rcd = conj_rcdvm[j_prog] * P_rcdvm[j_prog][m_risk][m_need]
        # Violation Cost
        Cost_vol = conj_vio[j_prog] * P_vio[j_prog][m_risk]
        # C = Cost_ocp + Cost_rcd + Cost_vol + 0.2 * random.random()
        C = Cost_ocp + Cost_rcd + Cost_vol

        '''Cost-to-go'''
        # Approximated the value function from lasso
        V_togo = 0.0008 * ocp_new[0] * ocp_new[0] + 0.0002 * ocp_new[1] ** 2 + 0.0005 * ocp_new[2] * ocp_new[2]
        cost[j_prog] = C + gamma * V_togo

    return min(cost, key=cost.get)


if __name__ == '__main__':

    # 0 for high-risk, 1 for low-risk
    m_risk = 0
    # occupancy for [jail, work release, home detention]
    occupancy = [400, 160, 350]
    placement = placement_onetype_based(m_risk, occupancy)
    print('Placement decision:', placement)
