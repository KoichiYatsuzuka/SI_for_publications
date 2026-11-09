"""
電流値を逐次計算して、その波形をフーリエ変換してインピーダンスを求める。
大量のイテレーションのため、遅い。
計算の中心はここではない。
"""
#%%
import pandas as pd
import numpy as np
import os
import time

import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec

import common
import non_analytical_impedance as na_imp
import general_kinetics as kin

POINTS_PER_PERIOD = 3600
NUMBER_OF_PERIODS = 11

data_files_dir = "./sequential_numrical_simulation/"
candidates_df = pd.read_csv(data_files_dir+"/parameter_candidates.csv", index_col=0)

applied_potential = ec.PotentialArray(np.linspace(-0.3, 0.3, 13))
frequencies = ec.FrequencyArray(np.logspace(-2, 5, 29))
potential_amplitude = ec.Potential(0.01)

start = time.time()

for row_num in range(len(candidates_df)):
    row = candidates_df.iloc[row_num]

    simulation_name = "index{}_amp{}_{}ppp_{}periods".format(row_num, potential_amplitude, POINTS_PER_PERIOD, NUMBER_OF_PERIODS)

    if os.path.isfile(data_files_dir+simulation_name+".csv"):
        print("The file already exists. Skip the simulation. -> ", simulation_name)
        continue

    reaction1 = kin.ElementaryStep_ET(
        common.Alpha(row[common.KEY_ALPHA_1]),
        common.ReactionRateCoefficient(10**row[common.KEY_LOG_K0_1]),
        ec.Potential(row[common.KEY_E0_1]),
        forward_direction=kin.Direction.CATHODIC
    )

    reaction2 = kin.ElementaryStep_ET(
        common.Alpha(row[common.KEY_ALPHA_2]),
        common.ReactionRateCoefficient(10**row[common.KEY_LOG_K0_2]),
        ec.Potential(row[common.KEY_E0_2]),
        forward_direction=kin.Direction.CATHODIC
    )

    active_site_density = 10**row[common.KEY_LOG_ACTIVE_SITE_DENSITY]

    imp = na_imp.impedance_via_sequential_current_simulation(
        applied_potential,
        frequencies,
        reaction1,
        reaction2,
        potential_amplitude,
        active_site_density,
        POINTS_PER_PERIOD,
        NUMBER_OF_PERIODS
    )
    

    df_mut = pd.DataFrame()

    for imp_each_potential in imp:
        df_imp_mut = imp_each_potential.to_data_frame()         

        df_imp_mut["potential"] = imp_each_potential.applied_potential

        df_mut = pd.concat([df_mut, df_imp_mut])

    # header = \
    #     ["index: {}".format(row_num),
    #     "alpha 1: {}".format(row[common.KEY_ALPHA_1]),
    #     "log K0 1: {}".format(row[common.KEY_LOG_K0_1]),
    #     "E0 1: {}".format(row[common.KEY_E0_1]),
    #     "alpha 2: {}".format(row[common.KEY_ALPHA_2]),
    #     "log K0 2: {}".format(row[common.KEY_LOG_K0_2]),
    #     "E0 2: {}".format(row[common.KEY_E0_2]),
    #     "log active site density: {}".format(row[common.KEY_LOG_ACTIVE_SITE_DENSITY]),
    #     "-------------------------------------------------------------------------------"]

    df_mut.reindex(columns=[
        "potential", "frequency", "real Z", "imaginary Z"
    ]).to_csv(data_files_dir+simulation_name+".csv")

    intermediate = time.time()
    print(str(row_num)+": {} sec".format(intermediate-start))

    