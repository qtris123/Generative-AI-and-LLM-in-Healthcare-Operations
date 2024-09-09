""""
Community Corrections Project - Flow Parameters
- Realized setting, estimated based on TCCC data
- All the parameterize is on the fast-timescale model, i.e., arrival and departure counted per day

Author: Xiaoquan Gao
Last Updated: 9/30/2022
"""

import math

# Flow parameters
# Station: 0: jail. 1: work release. 2: home detention. 3: day reporting
# Risk type: 0: severe. 1: mild.

'''Modeling Settings'''
N_Programs = 3  # Number of programs
N_RiskTypes = 2  # Number of risk types
N_NeedTypes = 2  # Number of need types
# N_Types = N_RiskTypes * N_NeedTypes  # Total number of customer types
Total_Horizon = 1080    # Length of simulation horizon
# Total_Horizon = 90     # Length of simulation horizon

'''Arrival Parameters'''
# ArrivalRate[m1][m2]: the arrival rate \lambda for risk-type m1 and need-type m2 clients
ArrivalRate = [[0 for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)]

# Parameterize the arrival rate based on IDOC and TCCC data
# A_base = (20.09 + 26.2 + 0.8 + 37.2) / 7

if N_RiskTypes == 2 and N_NeedTypes == 2:
    A_base = 12.04 * 0.8 * 1.0
    for m_risk in range(N_RiskTypes):
        for m_need in range(N_NeedTypes):
            # Adjust the arrival according to the risk and need type
            ArrivalRate[m_risk][m_need] = A_base * (0.3 + 0.4 * m_risk) * 0.5
    ClassNames = [['severe-need', 'severe-no-need'], ['mild-need', 'mild-no-need']]
    # print(ArrivalRate)
else:
    raise Exception('Need to re-parameterize for the model other than 2 risk types.')

'''Departure Parameters'''
# SentenceLength[m][j]: the sentenced length \mu for type m clients in station j
# Sentence length only depends on the risk type, not need type
SentenceLength = [[0 for _ in range(N_Programs)] for _ in range(N_RiskTypes)]

# Parameterize the sentenced length based on IDOC and TCCC data
if N_Programs == 4:
    # Use the realized setting as in Tippecanoe County
    D_base = [12 * 7, 5 * 7, 14 * 7, 15 * 7]  # Baseline LOS for each program
    for m1 in range(N_RiskTypes):
        # Adjust the LOS according to the risk type
        for j in range(N_Programs):
            SentenceLength[m1][j] = D_base[j] + m1 * 2 * 7 - int(N_Programs / 2.0)
elif N_Programs == 3:
    # Use the realized setting as in Tippecanoe County
    D_base = [12 * 7, 5 * 7, 14 * 7]  # Baseline LOS for each program
    for m1 in range(N_RiskTypes):
        # Adjust the LOS according to the risk type
        for j in range(N_Programs):
            SentenceLength[m1][j] = D_base[j] + m1 * 2 * 7 - int(N_Programs / 2.0)
    ProgramNames = ['jail', 'work release', 'home detention']
else:
    raise Exception('Need to re-parameterize for the model other than 4 programs.')


# Risk Type Transition Probabilities -- only risk types
if N_Programs == 3 and N_RiskTypes == 2:
    risk_tran_dict = {}
    # risk_tran_dict[(program, type 1,  type 2)]: prob. transit from type 1->2 in program
    # risk transition in jail, work release, home detention
    risk_tran_dict[(0, 0, 1)], risk_tran_dict[(0, 1, 0)] = 0.005, 0.01
    risk_tran_dict[(1, 0, 1)], risk_tran_dict[(1, 1, 0)] = 0.01, 0.005
    risk_tran_dict[(2, 0, 1)], risk_tran_dict[(2, 1, 0)] = 0.015, 0.005
    for j in range(N_Programs):
        for m in range(N_RiskTypes):
            risk_tran_dict[(j, m, m)] = 1 - risk_tran_dict[(j, m, N_RiskTypes-1-m)]

    # probability of transit from type 1 to type 2 in this program during each epoch
else:
    raise Exception('Need to re-parameterize for the model other than (3 programs, 2 risk types).')

