def print_test():
    print("MG_env was read successfully")


import numpy as np
import sys
import pandas as pd
import matplotlib.pyplot as plt
from random import seed
from random import randint

sys.path.append('/MG_env')
from MG_env.PSCAD_env import *

PSCAD_env_read()

real_day_timestep = 15 * 60  # this is the time step should be used for parsing the load profile
# therefore we have (24 hours * 60 minute)/ real_day_timestep(15) = 96 windows
real_to_sim_ratio = 60
bus_total = 2
nominal_V_LL = 12.47
nominal_values = [1, 1, 1, 50, 0, 0, 0, 0, 0, 0, 1, 1, 1, 50, 0, 0, 0, 0, 0, 0]

# if the timestamp is string you can convert it to datetime object
# https://stackoverflow.com/questions/35595710/splitting-timestamp-column-into-separate-date-and-time-columns

"""
load_pd = pd.DataFrame(load_data.iloc[:, 0:3])
load_pd["day"] = [d.day for d in load_pd['date']]
load_pd["hour"] = [d.hour for d in load_pd['date']]
load_pd["minute"] = [d.minute for d in load_pd['date']]
load_pd["second"] = [d.second for d in load_pd['date']]
load_pd.head()


def plot_df(df, x, y, title="", xlabel="Time", ylabel="Load-MW", dpi=100):
    plt.figure(figsize=(16, 5), dpi=dpi)
    plt.plot(x, y, color='tab:red')
    plt.gca().set(title=title, xlabel=xlabel, ylabel=ylabel)
    plt.show()

plot_df(load_pd, x=load_pd.date, y=load_pd.load, title="Daily load profile- 5 minutes interval")
"""

# Define the model topology
model_topo = {
    'BUS_1': {'Voltage': ['V_BUS1_A', 'V_BUS1_B', 'V_BUS1_C'],
              'Resource': ['BESS_1_SOC', 'BESS_1_P', 'BESS_1_Q', 'PV1_P', 'PV1_Q'],
              'Load': ['Load_1_P', 'Load_1_Q']
              },

    'BUS_2': {'Voltage': ['V_BUS2_A', 'V_BUS2_B', 'V_BUS2_C'],
              'Resource': ['BESS_2_SOC', 'BESS_2_P', 'BESS_2_Q', 'PV2_P', 'PV2_Q'],
              'Load': ['Load_2_P', 'Load_2_Q']
              }
}

historical_load_bus_1 = pd.read_csv(r'C:\Users\AD-MYazdani\Desktop\SDSU Research-RL-Microgrid\MG_env\dataSet_15min.csv')



# Organize Files
def get_result(sourceFolder, outputName):
    files = os.listdir(sourceFolder)

    # get files with outputs
    outputFiles = [os.path.join(sourceFolder, out) for out in files if '.out' in out and outputName in out]

    resultList = []
    for fileName in outputFiles:
        res = pd.read_fwf(fileName, skiprows=1, header=None, delim_whitespace=True, index_col=0)
        resultList.append(res)
    names = []
    with open(os.path.join(sourceFolder, outputName + '.inf'), 'r') as file:
        for name in file:
            name = name.split('Desc="')
            name = name[1].split('Group')
            name = name[0].split('"')
            names.append(name[0])

    results = pd.concat(resultList, axis=1, ignore_index=True)
    results.columns = names

    return results


# global variable: # of BESS in the System and corresponding PV
def bess_reward(input_df, col, soc_up_lim=80, soc_low_lim=20, excess_lim1=10, excess_lim2=15):
    # overcharge scenario:
    up_threshold1 = soc_up_lim + excess_lim1
    df_filter = input_df.where(input_df.loc[:, col] > soc_up_lim).dropna()  # add kW +/-
    normalization_factor = 95  # consider changing this based on the function inputs
    df_filter['reward'] = (df_filter.loc[:, col]) / normalization_factor  # between 0,1
    # df_filter = input_df.where(input_df.BESS_1_SOC> up_threshold1 )

    # Undervoltage scenario
    low_threshold1 = soc_low_lim - excess_lim1
    df_filter = input_df.where(input_df.loc[:, col] < soc_low_lim).dropna()  # add kW +/-
    normalization_factor = 20  # consider changing this based on the function inputs
    df_filter['reward'] = (soc_low_lim - df_filter.loc[:, col]) / normalization_factor  # between 0,1

    reward = df_filter.reward.sum()

    return -1 * (reward/input_df.shape[0])*10  # making a reward a number between 0 and 10


def dual_resource(input_df, BESS_kw_cap, timeStep):
    # if between 6AM and 6PM or PV kw >10 , then BESS should not discharge unless load > PV size.
    # sign convention: kW<0 -> power drawn from the grid, batteries charged.
    #                  kW>0 -> power into the grid, batteries discharged.

    # we want to count number of occurrence where PV is available AND load < PV_output  AND P_BESS > 0
    # In this case we need to punish the agent, the reward should be calculated as follows:
    # ( number of occurrence in each 15 minutes(real time) -> 60 seconds(simulation)/number of samples )*100

    # This is to address the condition load < PV_output  AND P_BESS > 0
    df_filter = input_df.where(input_df.load < input_df.PV1_P).dropna()
    df_filter = input_df.where(input_df.BESS_1_P < 0).dropna()
    max_discharge = -1 * df_filter.BESS_1_P.min()
    # This will make the reward between 0 and 1
    df_filter['reward'] = -1 * (df_filter.BESS_1_P / BESS_kw_cap)
    # reward need to get updated
    return 1


def next_day_reserve(load_predict_df, pv_predict_df, i_episod):
    # the expectation is to ensure battery is charged enough (based on it's limitations) before goes to discharge 
    # from evening to the morning of next day
    # 1- Obtain the the PV prediction(this should be at least hourly for upto 24 hours ahead)
    # 2- Find the current date
    # 3- Go to the 1 day ahead PV generation model(for now this will be fake but there should be a separate 
    #    ML model to do this prediction, maybe pytorch NN)
    # 4- Compare total predicted PV generation for the next day and compare with the total predicted load(or maybe
    #    load of the day) between times that solar is available. 

    hours = []
    hr_gen = []

    #     total_pv_kw_next_day= pv_predict_df.pv_kw.sum()
    #     total_load_next_day = load_predict_df.load.sum()

    for i in range(len(pv_predict_df)):
        if pv_predict_df.iloc[i, 1] > 10:
            hours.append(pv_predict_df.iloc[i, 0])

    for j in hours:
        hour = j.hour
        hr_gen.append(pv_predict_df.iloc[hour, 1] - load_predict_df.iloc[hour, 1])

    total_excess_gen_next_day = sum(hr_gen)
    return total_excess_gen_next_day


def voltage_reward(input_df, three_ph_v, avg_V_col_name):
    v_nominal = 1
    v_limit_1 = 0.03
    low_threshold1 = v_nominal - v_limit_1

    input_df[avg_V_col_name] = input_df[three_ph_v].mean(axis=1)

    min_val = input_df[three_ph_v[0]].min()
    v_diff = v_limit_1 - min_val

    # Under voltage
    input_df = input_df[(input_df[avg_V_col_name] < low_threshold1)]
    input_df['Reward'] = ((input_df[avg_V_col_name] - min_val) / v_diff) * 10
    reward = (input_df['Reward'].sum() / real_to_sim_ratio) * 10

    return reward


# remember to reset the actions
def reset(model_topology, nominal_values):
    name = []
    x = list(model_topo.keys())[0]
    y = list(model_topo[x].keys())[0]
    z = model_topo[x][y]

    for x in list(model_topo.keys()):
        for y in list(model_topo[x]):
            for z in list(model_topo[x][y]):
                name.append(z)

    reset_pd = pd.DataFrame(nominal_values, index=name)
    reset_pd = reset_pd.T
    reset_pd.insert(0, "Time", 0)
    return_arr = np.array(reset_pd.iloc[0, :])
    return return_arr


def find_hour(num):
    hour = int(num / 12)
    segment = int(num % 12)
    return hour, segment


def avg_state_sample(input_df, transient=10, randomness=10):
    output_stack = pd.DataFrame()
    rand_index = []
    data_len = input_df.shape[0]
    start_index = 0 + transient
    end_index = data_len - transient

    avg_data = input_df.iloc[start_index:end_index, :]

    for i in range(randomness):
        rand_index.append(randint(start_index, 50))

    for row in (rand_index):
        string = pd.Series(avg_data.iloc[row, :])
        output_stack = output_stack.append(string, ignore_index=True)

    output_stack = output_stack[avg_data.columns]
    output_sample = output_stack.mean()

    return output_sample


def per_unit(input_df, nominal_V_LL):
    for col in input_df.columns:
        if 'V_BUS' in col:
            input_df.loc[:, col] = input_df.loc[:, col] / nominal_V_LL
        else:
            continue

    return input_df


def sys_reward(input_df, model_topo, timeStep):
    Bus_voltage_reward = []
    Bus_bess_reward = []
    Bus_load_reward = []

    Buses = list(model_topo.keys())
    connected_to_bus = list(model_topo[Buses[0]].keys())
    for m in range(len(Buses)):

        for n in range(len(connected_to_bus)):

            if connected_to_bus[n] == 'Voltage':
                bus_n_voltage = list(model_topo[Buses[m]][connected_to_bus[n]])
                # print(bus_n_voltage)
                Avg_V_col_name = 'V_' + str(Buses[m]) + '_avg'
                # print(Avg_V_col_name)
                v_reward = voltage_reward(input_df, bus_n_voltage, Avg_V_col_name)
                Bus_voltage_reward.append(v_reward)

            if connected_to_bus[n] == 'Resource':

                assets_connected_to_bus_n = list(model_topo[Buses[m]][connected_to_bus[n]])
                loads_connected_to_bus_n = list(model_topo[Buses[m]]['Load'])
                for k in range(len(assets_connected_to_bus_n)):
                    if "BESS" and "SOC" in assets_connected_to_bus_n[k]:
                        BESS_col_name = str(assets_connected_to_bus_n[k])  # +'_SOC'
                        Bus_bess_reward.append(
                            bess_reward(input_df, BESS_col_name))  # data_pd should be the actual dataframe

            #                   if "PV" in assets_connected_to_bus_n[k]:
            #                       print(5454)

            if connected_to_bus[n] == 'Load':
                print('Load Reward- This function still requires work')

    combined_reward_per_bus = [i + j for i, j in zip(Bus_voltage_reward, Bus_bess_reward)]

    return combined_reward_per_bus


def step_env(actions, current_episod, timeStep):
    #temp vars for testing(chenge to the correct ones)
    loads = [historical_load_bus_1.iloc[timeStep].loadMW/3,historical_load_bus_1.iloc[timeStep].loadMW/3,historical_load_bus_1.iloc[timeStep].loadMW/3,
             historical_load_bus_1.iloc[timeStep].loadKVAR/3, historical_load_bus_1.iloc[timeStep].loadKVAR/3, historical_load_bus_1.iloc[timeStep].loadKVAR/3]
    # File to read from
    path = r"C:\Users\AD-MYazdani\Desktop\SDSU Research-RL-Microgrid\MG_env\Output_Dest_Folder"
    MG_env_path = os.getcwd() + '\\' + 'MG_env' + '\\'
    runTimeFilePath = MG_env_path
    destFolder = "Output_Dest_Folder"
    resultPath = os.path.join(MG_env_path, destFolder)
    projectName = "feeder4_BESS_CSI"
    fortranExt = '.gf46'
    sourceFolder = MG_env_path + projectName + fortranExt


    next_eps = current_episod + 1
    current_file_name = 'state' + '_' + str(current_episod) + '.csv'
    next_file_name = 'state' + '_' + str(next_eps) + '.csv'



    current_df = pd.read_csv(path + '\\' + current_file_name)
    current_df = per_unit(current_df, 12)
    current_state = avg_state_sample(current_df)
    BESS_1_SOC_prev_state = current_df.loc[current_df.index[-1], 'BESS_1_SOC']

    system_reward = sys_reward(current_df, model_topo, timeStep)
    # actions_from_agent = action  #[p1+,p1-,q1+,q1-, p2+,p2-,q2+,q2-, ... , pn+,pn-,qn+,qn-]

    # call PSCAD: first you have to figure out how to pass the snapshot values to PSCAD


    # Apply load to the dynamic load
    # Apply the actions

    run_PSCAD(runTimeFilePath, resultPath, projectName, 'output_nextState.out', 'snap_nextState.snp', True, actions, loads , BESS_1_SOC_prev_state)

    #Swap "snap_nextState.snp" with "snap_currentState.snp"
    currentStateToBeRemoved = os.path.join(sourceFolder, 'snap_currentState.snp')
    nextStateToBeRenamed = os.path.join(sourceFolder, 'snap_nextState.snp')
    print(currentStateToBeRemoved)
    print(nextStateToBeRenamed)

    if os.path.isfile(currentStateToBeRemoved):
        print("currentState Removed")
        os.remove(currentStateToBeRemoved)
    # else:
    #     print("Error: %s file not found" % currentStateToBeRemoved)

    os.rename(nextStateToBeRenamed, currentStateToBeRemoved )
    print("RENAME COMPLETED")


    outputFile = os.path.join(resultPath, next_file_name)
    next_df = get_result(sourceFolder, 'output_nextState')
    next_df.insert(loc=0, column='Time', value=next_df.index)
    next_df.to_csv(outputFile, index=False)

    next_df = per_unit(next_df, 12)
    next_state = avg_state_sample(next_df)

    if current_episod >= 288:
        done = 1
    else:
        done = 0
    return next_state, system_reward, done  # (next_state, reward, done)
