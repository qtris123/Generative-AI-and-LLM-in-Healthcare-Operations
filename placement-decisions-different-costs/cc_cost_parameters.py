""""
Community Corrections Project - Cost Functions
 - Parameterized based on TCCC data for the model with 4 programs and 2 risk types

Author: Xiaoquan Gao
Last Updated: 12/18/2022
"""

import math
import pandas as pd
import numpy as np
import random

from cc_flow_parameters import *

random.seed(2022)
np.random.seed(2022)


def cost_wr(ocp_wr):
    if ocp_wr < 150:
        return 0
    else:
        return 9999 * (ocp_wr-150) ** 2


def ocp_cost(occupancy):
    cost = sum([c_unit_occu[j] * occupancy[j]**2 for j in range(N_Programs)])
    if 0 <= occupancy[1] <= 100:
        cost += 3 * occupancy[1] ** 2
    elif occupancy[1] > 100:
        cost += float('inf')
    return cost


def ocp_cost_lin(occupancy):
    cost = sum([c_unit_occu[j] * occupancy[j] * SentenceLength[0][j] for j in range(N_Programs)])
    if 150 <= occupancy[1] <= 200:
        cost += 3 * occupancy[1]
    elif occupancy[1] > 200:
        cost += float('inf')
    return cost


def congestion_based_risks(occupancy):
    # Threshold for congestion for each station
    '''Adjust conj_th'''
    # conj_th = [320, 130, 300]
    conj_th = [320, 130, 300]

    # Compute the congestion-adjusted risks
    p_rcdvm_adj = [1, 1, 1.5]
    p_vio_adj = [1, 1, 1.5]
    # p_rcdvm_adj = [0, 0, 0]
    # p_vio_adj = [0, 0, 0]

    p_rcdvm_conj = [1 + p_rcdvm_adj[j] * max(occupancy[j] - conj_th[j], 0) / conj_th[j] for j in range(N_Programs)]
    p_vio_conj = [1 + p_vio_adj[j] * max(occupancy[j] - conj_th[j], 0) / conj_th[j] for j in range(N_Programs)]

    # print(occupancy, [[max(occupancy[j] - conj_th[j], 0) / 3 * conj_th[j] for j in range(N_Programs)]])

    return p_rcdvm_conj, p_vio_conj


# gamma = 0.98  # discounting factor
gamma = 0.5

warm_start = 60    # only count costs after ### days

'''Unit Cost'''
# unit cost of recidivism
c_unit_rcdvm = 1

# unit cost of occupancy
temp_occu_coef = 0.00036 * c_unit_rcdvm
temp_occu_ratio = [1.0, 0.8, 0.3]
c_unit_occu = [temp_occu_coef * x for x in temp_occu_ratio]

# Violation cost
c_unit_vio = 0.3 * c_unit_rcdvm  # unit cost of violation

'''Violation and Recidivism Probability'''
# probability of violation each epoch -- only depend on risk class
p_vio = [[0, 0.002, 0.003], [0, 0.0012, 0.0018]]

# Cummulative violation probability
P_vio = [[0 for _ in range(N_RiskTypes)] for _ in range(N_Programs)]
for j in range(N_Programs):
    for m in range(N_RiskTypes):
        P_vio[j][m] = p_vio[m][j] * SentenceLength[m][j]

# compute the recidivism risk based on the TCCC data
T = 1080  # Recidivism Window
# Recidivism -- depend both on risk class and need class
p_base = [[[] for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)]
# recidivism risk for high-risk classes (high-need, low-need) in different stations
p_base[0][0].extend([0.0014, 0.0006, 0.0008])
p_base[0][1].extend([0.0014, 0.0010, 0.0012])
# recidivism risk for low-risk classes (high-need, low-need) in different stations
p_base[1][0].extend([0.0010, 0.0002, 0.0006])
p_base[1][1].extend([0.0010, 0.0006, 0.0008])
# print(p_base)

p_rate = 0.997
def p_rcdvm_time(p_base, p_rate):
    p_rcdvm = {}
    for j_prog in range(N_Programs):
        p_rcdvm[j_prog] = [[[p_base[m1][m2][j_prog]] for m2 in range(N_NeedTypes)] for m1 in range(N_RiskTypes)]
    for t in range(Total_Horizon):
        for j in range(N_Programs):
            for m1 in range(N_RiskTypes):
                for m2 in range(N_NeedTypes):
                    p_rcdvm[j][m1][m2].append(p_rcdvm[j][m1][m2][t] * p_rate)
    # print(len(p_rcdvm), len(p_rcdvm[0]), len(p_rcdvm[0][0]))
    for j in range(N_Programs):
        for m1 in range(N_RiskTypes):
            for m2 in range(N_NeedTypes):
                p_rcdvm[j][m1][m2] = tuple(p_rcdvm[j][m1][m2])
    return p_rcdvm

p_rcdvm  = p_rcdvm_time(p_base, p_rate)

# Cummulative recidivism probability for simulation (during total horizon)
P_rcdvm = [[[0 for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)] for _ in range(N_Programs)]
for j in range(N_Programs):
    for m1 in range(N_RiskTypes):
        for m2 in range(N_NeedTypes):
            if j == 0:
                P_rcdvm[j][m1][m2] = sum(p_rcdvm[j][m1][m2][0:int(T-SentenceLength[m1][j])])
            else:
                P_rcdvm[j][m1][m2] = sum(p_rcdvm[j][m1][m2][0:int(T)])


