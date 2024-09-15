""""
Community Corrections Project - CC Simulation Functions

Author: Xiaoquan Gao
Last Updated: 9/14/2024
"""

"""
General notes: This code is built for the process flow simulation based on the RL-based heuristic

- INPUT: (1) Flow parameters (2) One-type RL value functions
- OUTPUT: A pickle file -- [[current # (t), # released(t)] for each program, each class, each time t]
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

N_LosTypes = 2

# Generate number of new arrivals following the Poisson process and flow parameters
def get_number_arrivals():
    return np.random.poisson(ArrivalRate)


# compute the sum of nested list
def nested_sum(lst):
    if isinstance(lst, (float, int)):
        return lst
    else:
        return sum(nested_sum(item) for item in lst)


# function to sum over the first two index for a given 3rd index
def sum_by_3rd_index(my_list, index_3):
    # initialize sum to 0
    total = 0
    # iterate over the elements of the list and accumulate the sum of the first two indices for the given 3rd index
    for i in range(len(my_list)):
        for j in range(len(my_list[i])):
            total += my_list[i][j][index_3]
    return total


# function to take the average over the first index of a dictionary whose values are 3-layer lists
def average_over_key(dict):
    # convert the list to a numpy array
    avg = [[[0 for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)] for _ in range(N_Programs)]
    # compute the sum over the first axis using the numpy.sum() function
    for t in dict.keys():
        for j in range(N_Programs):
            for m_risk in range(N_RiskTypes):
                for m_need in range(N_NeedTypes):
                    if isinstance(dict[t][j][m_risk][m_need], list):
                        avg[j][m_risk][m_need] += dict[t][j][m_risk][m_need][N_LosTypes - 1]
                    else:
                        avg[j][m_risk][m_need] += dict[t][j][m_risk][m_need]

    return np.array(avg) / len(dict)


def sum_over_key(dict):
    # convert the list to a numpy array
    avg = [[[0 for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)] for _ in range(N_Programs)]
    # compute the sum over the first axis using the numpy.sum() function
    for t in dict.keys():
        for j in range(N_Programs):
            for m_risk in range(N_RiskTypes):
                for m_need in range(N_NeedTypes):
                    avg[j][m_risk][m_need] += dict[t][j][m_risk][m_need]

    return avg


def convert_int(x):
    if isinstance(x, (np.ndarray, list)):
        return list(map(convert_int, x))
    else:
        return int(x)


# placement decision for each new arrival
def placement_onetype_based(m_risk, m_need, occupancy, conj_rcdvm, conj_vio):
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


# determine the placement decisions for the new arrivals based on arr_numbers and current occupancy
def rl_based_routing(num_arrivals, occupancy, conj_rcdvm, conj_vio):
    route_num = [[[0 for _ in range(N_Programs)] for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)]
    for m_risk in range(N_RiskTypes):
        for m_need in range(N_NeedTypes):
            for idx in range(num_arrivals[m_risk][m_need]):
                program = placement_onetype_based(m_risk, m_need, occupancy, conj_rcdvm, conj_vio)
                route_num[m_risk][m_need][program] += 1
    return route_num


def sum_over_risk_need(current_ocp, prog, los):
  sum_ocp = 0
  # Iterate through all possible values of risk and need
  for risk in range(len(current_ocp[prog])):
    for need in range(N_NeedTypes):
      sum_ocp += current_ocp[prog][risk][need][los]
  return max(0, sum_ocp)


def run_rl_based_heuristic_simulation(end_time, placement_df, c_unit_occu, c_unit_vio):
    random.seed(2024)
    np.random.seed(2024)
    placement_df = pd.DataFrame()

    # Running variable to keep track of what ID values have already been used (start at 0, increment as clients arrive)
    id_running = 0

    # Running variable to keep track of time
    t = 0
    cost = 0
    c_unit_rcdvm = 1

    # Initialize client and performance dataframe/dictionary
    # The client dataframe only keeps track of clients still in the system
    client_df = pd.DataFrame(columns=['ID', 'RiskType', 'NeedType', 'Program', 'ArrivalTime'])

    # Keep track of performance measure
    occup_dict = {}   # Key: time. Value: [current # (t) for each program, each class]
    dep_dict = {}  # Key: time. Value: [departure # (t) for each program, each class].
    rcdvm_dict = {}
    vio_dict = {}
    rcd_arr_lst = [0 for _ in range(Total_Horizon)]

    # Initialize the system state
    occup_dict[-1] = [[[[0 for _ in range(N_LosTypes)] for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)]
                      for _ in range(N_Programs)]
    num_arrivals = [[[0 for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)] for _ in range(end_time)]

    """
    Begin time loop
    """
    while t < end_time:
        # Initiate the occupancy and departure measure
        occup_dict[t] = [[[[0 for _ in range(N_LosTypes)] for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)]
                         for _ in range(N_Programs)]
        dep_dict[t] = [[[0 for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)] for _ in range(N_Programs)]
        rcdvm_dict[t] = [[[0 for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)] for _ in range(N_Programs)]
        vio_dict[t] = [[[0 for _ in range(N_NeedTypes)] for _ in range(N_RiskTypes)] for _ in range(N_Programs)]

        # Get the number of new arrivals for each types of clients
        rcd_arr_lst[t] = nested_sum(num_arrivals[t])
        temp = get_number_arrivals()
        for m_risk in range(N_RiskTypes):
            for m_need in range(N_NeedTypes):
                num_arrivals[t][m_risk][m_need] += temp[m_risk][m_need]

        # Get number of new arrivals based on the input routing policy
        occupancy_los = [[sum_over_risk_need(occup_dict[t-1], j_prog, los) for los in range(N_LosTypes)]
                         for j_prog in range(N_Programs)]
        # print(occupancy_los)
        occupancy = [occupancy_los[j][N_LosTypes-1] for j in range(N_Programs)]
        cost += ocp_cost_lin(occupancy)
        # Compute the congestion-adjusted risks
        p_rcdvm_conj, p_vio_conj = congestion_based_risks(occupancy)

        route = rl_based_routing(num_arrivals[t], occupancy, p_rcdvm_conj, p_vio_conj)
        # print('Time:', t, 'Arrivals:', num_arrivals[t], 'Placement:', route[0], route[1], 'Occupancy:', occupancy)

        # Add new arrivals into the client list
        for m_risk in range(N_RiskTypes):
            for m_need in range(N_NeedTypes):
                for j_prog in range(N_Programs):
                    for k in range(route[m_risk][m_need][j_prog]):
                        new_row_client = [{'ID': id_running, 'RiskType': m_risk, 'NeedType': m_need,
                                    'Program': j_prog, 'ArrivalTime': t}]
                        client_df = pd.concat([client_df, pd.DataFrame.from_records(new_row_client)], ignore_index=True)
                        id_running += 1

                        new_row_placement = pd.DataFrame({'c_occu_jail': c_unit_occu[0],
                                                          'c_occu_wr': c_unit_occu[1],
                                                          'c_occu_hd': c_unit_occu[2],
                                                          'c_vio': c_unit_vio,
                                                          'ocp_jl_low_los': occupancy_los[0][0],
                                                          'ocp_jl_total': occupancy_los[0][1],
                                                          'ocp_wr_low_los': occupancy_los[1][0],
                                                          'ocp_wr_total': occupancy_los[1][1],
                                                          'ocp_hd_low_los': occupancy_los[2][0],
                                                          'ocp_hd_total': occupancy_los[2][1],
                                                          'class_mild': m_risk,
                                                          'need': 1-m_need,
                                                          'placement': ProgramNames[j_prog]}, index=[0])
                        if t > warm_start:
                            placement_df = pd.concat([placement_df, new_row_placement], axis=0, ignore_index=True)

        # Compute number of departures in each program
        for m_risk in range(N_RiskTypes):
            for m_need in range(N_NeedTypes):
                for j_prog in range(N_Programs):
                    dep_client_arr_time = t - SentenceLength[m_risk][j_prog]
                    # print('dep_client_arr_time:', dep_client_arr_time, 't:', t,
                    #       'SentenceLength:', SentenceLength[m_type][j_prog])
                    client_depature_idx = client_df[(client_df.RiskType == m_risk) & (client_df.NeedType == m_need)
                                                    & (client_df.Program == j_prog)
                                                    & (client_df.ArrivalTime < dep_client_arr_time + 0.1)].index
                    dep_dict[t][j_prog][m_risk][m_need] = len(client_depature_idx)

                    # Simulate the recidivism of released clients
                    for k in range(dep_dict[t][j_prog][m_risk][m_need]):
                        if j_prog == 0:
                            t_start = 0
                        else:
                            t_start = SentenceLength[m_risk][j_prog]
                        p_rcdvm_temp = deepcopy(p_rcdvm[j_prog][m_risk][m_need][t_start:Total_Horizon])
                        p_rcdvm_temp = [p * p_rcdvm_conj[j_prog] for p in p_rcdvm_temp]
                        sum_temp = sum(p_rcdvm_temp)
                        if Total_Horizon - t_start > 0 and random.random() < sum_temp:
                            p_rcdvm_temp = [p/sum_temp for p in p_rcdvm_temp]
                            rcdvm_time = random.choices(np.arange(t_start, Total_Horizon), weights=p_rcdvm_temp)[0]
                            if t + rcdvm_time < Total_Horizon:
                                num_arrivals[int(t + rcdvm_time)][m_risk][m_need] += 1
                                rcdvm_dict[t][j_prog][m_risk][m_need] += 1
                                cost += c_unit_rcdvm
                                # print('Released clients recidivism')

                    # Drop the departed clients
                    if len(client_depature_idx):
                        client_df = client_df.drop(client_depature_idx)

        '''Compute occupancy in each LOS class in each program'''
        # Two LOS classes for each progrsm: < 1/2 base sentence length; > 1/2 base sentence length
        for m_risk in range(N_RiskTypes):
            for m_need in range(N_NeedTypes):
                for j_prog in range(N_Programs):
                    client_occup_df = client_df[(client_df.RiskType == m_risk) & (client_df.NeedType == m_need)
                                                & (client_df.Program == j_prog)
                                                & (client_df.ArrivalTime < t + 0.1)]
                    client_occup_df_los_low = client_occup_df[(client_df.ArrivalTime > t - 0.5*SentenceLength[m_risk][j_prog])]
                    # occupancy from customers whose LOS < 1/2 base sentence length
                    occup_dict[t][j_prog][m_risk][m_need][0] = len(client_occup_df_los_low)
                    # total occupancy
                    occup_dict[t][j_prog][m_risk][m_need][N_LosTypes-1] = len(client_occup_df)

        # Simulate recidivism of CC participants
        '''To optimize the code: avoid enumerating all the CC participants'''
        client_df.reindex
        for j_prog in range(1, N_Programs):
            client_cc_df = client_df[(client_df.Program == j_prog)
                                     & (client_df.ArrivalTime < t + 0.1)]
            for index, row in client_cc_df.iterrows():
                m_risk = row['RiskType']
                m_need = row['NeedType']
                los = int(t - row['ArrivalTime'])
                if random.random() < p_rcdvm_conj[j_prog] * p_rcdvm[j_prog][m_risk][m_need][los]:
                    # Modify the occupancy level, number of departure and number of recidivism
                    occup_dict[t][j_prog][m_risk][m_need][N_LosTypes-1] -= 1
                    if los < 0.5*SentenceLength[m_risk][j_prog]:
                        occup_dict[t][j_prog][m_risk][m_need][0] -= 1
                    rcdvm_dict[t][j_prog][m_risk][m_need] += 1
                    cost += c_unit_rcdvm
                    # print('CC participants recidivism')

                    # Add recidivism arrival
                    # new_row = [{'ID': id_running, 'Type': m_type, 'Program': j_prog, 'ArrivalTime': t+rcdvm_time}]
                    # client_df = pd.concat([client_df, pd.DataFrame.from_records(new_row)], ignore_index=True)
                    # id_running += 1
                    if int(t + 1) < end_time:
                        num_arrivals[int(t + 1)][m_risk][m_need] += 1
                elif random.random() < p_vio_conj[j_prog] * p_vio[m_risk][j_prog]:
                    # Violation: participants being sent to jail
                    # Modify the occupancy levels of CC program and jail, and number of violation
                    occup_dict[t][j_prog][m_risk][m_need][N_LosTypes-1] -= 1
                    occup_dict[t][0][m_risk][m_need][N_LosTypes-1] += 1
                    if los < 0.5*SentenceLength[m_risk][j_prog]:
                        occup_dict[t][j_prog][m_risk][m_need][0] -= 1
                        occup_dict[t][0][m_risk][m_need][0] += 1
                    vio_dict[t][j_prog][m_risk][m_need] += 1
                    cost += c_unit_vio

                    # Move the CC participant with technical violation to jail
                    # row_idx = client_df.index[client_df['ID'] == row['ID']][0]
                    row_idx = np.where(client_df['ID'] == row['ID'])[0]
                    client_df.iloc[row_idx]['Program'] = 0

        # Update time
        t += 1

    # csv_filename = f'placementDecisions_rl_los_cost_{str(setting_index)}.csv'
    # placement_df.to_csv(csv_filename, index=False)

    Avg_ocp = average_over_key(occup_dict)
    cost_sum = 0
    for j_prog in range(N_Programs):
        cost_sum += c_unit_occu[j_prog] * nested_sum(Avg_ocp[j_prog]) * Total_Horizon

    Sum_rcd = sum_over_key(rcdvm_dict)
    cost_sum += c_unit_rcdvm * nested_sum(Sum_rcd)

    Sum_vio = sum_over_key(vio_dict)
    cost_sum += c_unit_vio * nested_sum(Sum_vio)

    return Avg_ocp, Sum_rcd, Sum_vio, cost_sum, placement_df


if __name__ == '__main__':

    # Define ranges for each parameter
    temp_occu_coef_range = [0.00009, 0.00036, 0.00144]  # Range for temp_occu_coef (baseline cost of occupancy)
    c_unit_vio_factor_range = [0.1, 0.3, 0.5]  # Range for c_unit_vio factor (unit cost of violation)

    # Define variations for temp_occu_ratio
    # Order of stations: [jail, wr, hd]
    temp_occu_ratio_variations = [
        [1.0, 0.8, 0.3],  # Original ratio
        [1.0, 0.8, 0.1],  # Alternative ratio -- cheap HD
        [1.0, 0.3, 0.3],  # Alternative ratio -- expensive HD
    ]

    # Create all combinations of parameters
    parameter_combinations = list(itertools.product(
        temp_occu_coef_range,
        c_unit_vio_factor_range,
        temp_occu_ratio_variations
    ))

    # Open a file to write all cost parameters and results
    placement_df = pd.DataFrame(columns=['c_occu_jail', 'c_occu_wr', 'c_occu_hd', 'c_vio',
                                         'ocp_jl_low_los', 'ocp_jl_total',
                                         'ocp_wr_low_los', 'ocp_wr_total',
                                         'ocp_hd_low_los', 'ocp_hd_total',
                                         'class_mild', 'need', 'placement'])
    with open('simulation_results_2.txt', 'w') as f:
        client_df = pd.DataFrame(columns=['ID', 'RiskType', 'NeedType', 'Program', 'ArrivalTime'])
        for index, (temp_occu_coef, c_unit_vio_factor, temp_occu_ratio) in enumerate(parameter_combinations):
            print('Simulation for Setting', index)
            # Calculate c_unit_occu and c_unit_vio based on the current combination
            c_unit_occu = [temp_occu_coef * x for x in temp_occu_ratio]
            c_unit_vio = c_unit_vio_factor * c_unit_rcdvm

            # Write parameters to file
            f.write(f"Simulation {index}:\n")
            f.write(f"c_unit_occu: {c_unit_occu}\n")
            f.write(f"c_unit_vio: {c_unit_vio}\n")
            f.write(f"c_unit_rcdvm: {c_unit_rcdvm}\n")

            # Run simulation
            print('index:', index)
            Avg_ocp, Sum_rcd, Sum_vio, total_cost, placement_df = run_rl_based_heuristic_simulation(Total_Horizon, placement_df, c_unit_occu, c_unit_vio)

            # Write results to file
            f.write("Results:\n")
            f.write(f"Average occupancy: {Avg_ocp}\n")
            f.write(f"Total recidivism: {nested_sum(Sum_rcd)}\n")
            f.write(f"Total violations: {nested_sum(Sum_vio)}\n")
            f.write(f"Total cost: {total_cost}\n")
            f.write("\n" + "=" * 50 + "\n\n")

            placement_df.to_csv('combined_placementDecisions_rl_los_cost_2.csv', index=False)

        print("Simulation completed. Results written to simulation_results.txt")

    # # Assuming all CSVs are in the current directory and follow the same naming pattern
    # csv_files = glob.glob('placementDecisions_rl_los_cost_*.csv')
    #
    # # Initialize an empty list to hold the DataFrames
    # dfs = []
    #
    # # Loop through the list of CSV files and read each one into a DataFrame
    # for csv_file in csv_files:
    #     df = pd.read_csv(csv_file)
    #     dfs.append(df)
    #
    # # Concatenate all the DataFrames into a single DataFrame
    # combined_df = pd.concat(dfs, ignore_index=True)
    # combined_df.to_csv('combined_placementDecisions_rl_los_cost.csv', index=False)
    # print("All CSV files have been integrated into 'combined_placementDecisions_rl_los_cost.csv'")
