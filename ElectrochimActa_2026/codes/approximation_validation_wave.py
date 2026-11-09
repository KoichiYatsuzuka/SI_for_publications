#%%
import numpy as np
#import pandas as pd

import matplotlib.pyplot as plt
#import matplotlib as matpl
#import originpro as op

from typing import Callable
from dataclasses import dataclass

import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec

import common
import non_analytical_impedance as na_imp
import impedance_simulator.intermediate_impedance as int_imp
#import impedance_simulator.kinetics_to_obserbable as kin_obs
#import tafel_simulater as tafel
import general_kinetics as kin

from joblib import Parallel, delayed

SHOWN_PERIODS = 3

@dataclass(frozen=True)
class ReactionNetwork:
    reaction1: kin.ElementaryStep_ET
    reaction2: kin.ElementaryStep_ET
    gamma: float
def get_non_fft_sequential_wave(
    applied_potential: ec.PotentialArray,
    frequencies: ec.FrequencyArray,
    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,
    potential_amplitude: ec.Potential,
    active_site_density: float,
    points_per_period: int = 7200,
    number_of_periods: int = 11
):

    current_list = []
    for frequency in frequencies:

        if isinstance(frequency, ec.Frequency) is False:
            raise TypeError("The type of frequency must be ec.Frequency.")

        current = na_imp.sequential_current_simulator(
            applied_potential,
            frequency,
            reaction1,
            reaction2,
            potential_amplitude,
            active_site_density,
            points_per_period,
            number_of_periods
        )

        extracted_current = current[:, -1*SHOWN_PERIODS*points_per_period:]

        current_list.append(extracted_current)
    
    i_list_each_e_f: list[list[ec.CurrentArray]]= []
    for i_array_each_e in current_list:
        i_list_each_e: list[ec.CurrentArray] = []
        if not cmn.is_voarray(i_array_each_e):
            raise TypeError            
        for i_array_each_f in i_array_each_e:
            if not cmn.is_voarray(i_array_each_f):
                raise TypeError
            i_list_each_e.append(
                i_array_each_f - (i_array_each_f.max() + i_array_each_f.min())/2
            )
        i_list_each_e_f.append(i_list_each_e)


    return ec.CurrentArray(i_list_each_e_f).transpose(1,0,2)


def get_fft_sequential_wave(
    non_fft_current: ec.CurrentArray, # shape: (印加周波数毎、印加電位毎、分解周波数毎)、印加電位部分は一次元
    frequencies: ec.FrequencyArray,
    number_of_periods: int,
):
    """電位が一次元なので、印加周波数と経過時間の二次元配列
    """
    
    fft_currents = na_imp.extraction_of_fft_current(
        non_fft_current.transpose(1,0,2), #元の関数の都合上、転置してから入れている。
        frequencies,
        number_of_periods
    )
    """返ってきている値: 上から
    印加周波数毎、
    印加電位毎、
    電流の振幅（tuple[分離に使った周波数, 周波数に対応する振幅（複素数）]; 印加周波数付近と定常部分の抽出）
    の多次元配列

    最後の部分だけ、再合成する
    """
    t = np.linspace(0, number_of_periods, 7200*number_of_periods)

    current_list_f_e: list[list[ec.CurrentArray]] = []
    
    for current_each_applied_freq in fft_currents:
        current_list_e: list[ec.CurrentArray] = []
        for current_each_applied_potential in current_each_applied_freq:
            current_amp = current_each_applied_potential[1]

            total_current_amp: complex = current_amp[1:].sum().value

            current_amp_real = total_current_amp.real #cos成分
            current_amp_imag = total_current_amp.imag #-sin成分

            current_amp_sin = current_amp_imag * -1
            current_amp_cos = current_amp_real 

            current = current_amp_sin*np.sin(2*np.pi*t) + current_amp_cos*np.cos(2*np.pi*t)# + current_amp[0].value.real/2

            current_list_e.append(ec.CurrentArray(current))
        current_list_f_e.append(current_list_e)
    return ec.CurrentArray(current_list_f_e).transpose(1,0,2) #最後に電位と周波数軸を転置

def get_simd_wave(
    applied_potential: ec.PotentialArray,
    frequencies: ec.FrequencyArray,
    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,
    potential_amplitude: ec.Potential,
    active_site_density: float,
    simulation_function: Callable
    [
        [
            np.ndarray, 
            ec.PotentialArray, 
            kin.ElementaryStep_ET, 
            kin.ElementaryStep_ET, 
            ec.Potential, 
            float
        ], 
        ec.ImpedanceArray],
):
    
    current_amp_complex:np.ndarray = potential_amplitude.value/simulation_function(
        frequencies,
        applied_potential,
        reaction1,
        reaction2,
        potential_amplitude,
        active_site_density,
    )

    # current_center = tafel.faraday_steady_current_array(
    #     ec.PotentialArray([applied_potential]),
    #     reaction1,
    #     reaction2,
    #     active_site_density
    # )

    current_amp_real:np.ndarray = current_amp_complex.real #sin成分
    current_amp_imag:np.ndarray = current_amp_complex.imag #cos成分
    #print(current_amp_complex)
    current_list_e_f:list[list[ec.CurrentArray]] = []
    for r_each_f, i_each_f in zip(current_amp_real, current_amp_imag):
        current_list_f: list[ec.CurrentArray] = []
        for r, i in zip(r_each_f, i_each_f):
            current = r*np.sin(2*np.pi*np.linspace(0, SHOWN_PERIODS, 7200*SHOWN_PERIODS)) \
                + i*np.cos(2*np.pi*np.linspace(0, SHOWN_PERIODS, 7200*SHOWN_PERIODS))#\
                #+ i_c.value #定常電流分
            current_list_f.append(current)
        current_list_e_f.append(current_list_f)

    return ec.CurrentArray(current_list_e_f)

def get_reaction_parameters(id: int)->ReactionNetwork:
    """
    id: RN_number
    Returns:
        reaction1, reaction2, gamma
    """

    #--------------put RN parameters here---------------------
    match id:
        case 1:
            RN = ReactionNetwork(
                reaction1=kin.ElementaryStep_ET(
                    common.Alpha(0.5),
                    common.ReactionRateCoefficient(8e2),
                    ec.Potential(0.15),
                    forward_direction=kin.Direction.CATHODIC
                ),
                reaction2=kin.ElementaryStep_ET(
                    common.Alpha(0.6),
                    common.ReactionRateCoefficient(1e1),
                    ec.Potential(-0.1),
                    forward_direction=kin.Direction.CATHODIC
                ),
                gamma=1e-10
            )
        case 2:
            RN = ReactionNetwork(
                reaction1=kin.ElementaryStep_ET(
                    common.Alpha(0.5),
                    common.ReactionRateCoefficient(8e3),
                    ec.Potential(0.15),
                    forward_direction=kin.Direction.CATHODIC
                ),
                reaction2=kin.ElementaryStep_ET(
                    common.Alpha(0.6),
                    common.ReactionRateCoefficient(1e2),
                    ec.Potential(-0.1),
                    forward_direction=kin.Direction.CATHODIC
                ),
                gamma=1e-10
            )
        case 3:
            RN = ReactionNetwork(
                reaction1=kin.ElementaryStep_ET(
                    common.Alpha(0.5),
                    common.ReactionRateCoefficient(8e3),
                    ec.Potential(0.15),
                    forward_direction=kin.Direction.CATHODIC
                ),
                reaction2=kin.ElementaryStep_ET(
                    common.Alpha(0.6),
                    common.ReactionRateCoefficient(1e2),
                    ec.Potential(-0.1),
                    forward_direction=kin.Direction.CATHODIC
                ),
                gamma=1e-11
            )
        case 4:
            RN = ReactionNetwork(
                reaction1=kin.ElementaryStep_ET(
                    common.Alpha(0.5),
                    common.ReactionRateCoefficient(8e2),
                    ec.Potential(0.15),
                    forward_direction=kin.Direction.CATHODIC
                ),
                reaction2=kin.ElementaryStep_ET(
                    common.Alpha(0.6),
                    common.ReactionRateCoefficient(1e1),
                    ec.Potential(-0.1),
                    forward_direction=kin.Direction.CATHODIC
                ),
                gamma=1e-11
            )
        # case 1:
        #     #RN1
        #     reaction1 = kin.ElementaryStep_ET(
        #         common.Alpha(0.5),
        #         common.ReactionRateCoefficient(1e2),
        #         ec.Potential(0.1),
        #         forward_direction=kin.Direction.CATHODIC
        #     )

        #     reaction2 = kin.ElementaryStep_ET(
        #         common.Alpha(0.6),
        #         common.ReactionRateCoefficient(1e1),
        #         ec.Potential(-0.1),
        #         forward_direction=kin.Direction.CATHODIC
        #     )
        #     gamma = 1e-10

        # case 2:
        #     #RN2
        #     reaction1 = kin.ElementaryStep_ET(
        #         common.Alpha(0.5),
        #         common.ReactionRateCoefficient(1e3),
        #         ec.Potential(0.1),
        #         forward_direction=kin.Direction.CATHODIC
        #     )

        #     reaction2 = kin.ElementaryStep_ET(
        #         common.Alpha(0.5),
        #         common.ReactionRateCoefficient(1e0),
        #         ec.Potential(-0.1),
        #         forward_direction=kin.Direction.CATHODIC
        #     )
        #     gamma = 1e-10

        # case 3:
        #     #RN3
        #     reaction1 = kin.ElementaryStep_ET(
        #         common.Alpha(0.5),
        #         common.ReactionRateCoefficient(1e0),
        #         ec.Potential(0.1),
        #         forward_direction=kin.Direction.CATHODIC
        #     )

        #     reaction2 = kin.ElementaryStep_ET(
        #         common.Alpha(0.6),
        #         common.ReactionRateCoefficient(1e1),
        #         ec.Potential(-0.1),
        #         forward_direction=kin.Direction.CATHODIC
        #     )
        #     gamma = 1e-10
        case _:
            raise ValueError("RN_number is strange.")
    #--------------put RN parameters here---------------------

    return RN


c_list = [
    "#555555",
    "#F02020",
    "#5050FF",
    "#A020A0",
    "#5050FF",
]

LINE_STYLE_FINE_DASH = (0, (1, 1))

TEXT_POS_X1 = 0.05
# TEXT_POS_X2 = 0.15
# TEXT_POS_X3 = 0.25
# TEXT_POS_X4 = 0.45

TEXT_POS_Y1 = 0.975
TEXT_POS_Y2 = 0.95

TEXT_POS_Y3 = 0.92
TEXT_POS_Y4 = 0.895
TEXT_POS_Y5 = 0.87
TEXT_POS_Y6 = 0.845

TEXT_POS_Y7 = 0.815
TEXT_POS_Y8 = 0.79
TEXT_POS_Y9 = 0.765


def plot_wave_and_EIS(
    potentials: ec.PotentialArray,
    potential_amplitude: ec.Potential,
    frequencies: ec.FrequencyArray,
    frequencies_fine: ec.FrequencyArray,
    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,
    gamma: float,
    output_filename_suffix: str = "",
):
    e = potentials
    f = frequencies
    f_fine = frequencies_fine

    #print(potential)
    raw_current = get_non_fft_sequential_wave(
        e,
        f,
        reaction1,
        reaction2,
        potential_amplitude,
        active_site_density=gamma,
        points_per_period=7200,
        number_of_periods=11
    )*1000 #A -> mA

    fft_current = get_fft_sequential_wave(
        raw_current,
        f,
        number_of_periods=SHOWN_PERIODS
    ) # mAのまま

    # simd_current_via_Q = get_simd_wave(
    #     e,
    #     f,
    #     reaction1,
    #     reaction2,
    #     potential_amplitude,
    #     active_site_density=gamma,
    #     simulation_function=int_imp.faraday_catalytic_impedance_MkIII
    # )*1000 #A -> mA

    # simd_current_via_tau = get_simd_wave(
    #     e,
    #     f,
    #     reaction1,
    #     reaction2,
    #     potential_amplitude,
    #     active_site_density=gamma,
    #     simulation_function=int_imp.faraday_catalytic_impedance_MkV
    # )*1000 #A -> mA

    simd_current_via_differential = get_simd_wave(
        e,
        f,
        reaction1,
        reaction2,
        potential_amplitude,
        active_site_density=gamma,
        simulation_function=int_imp.faraday_catalytic_impedance_diff_eq
    )*1000


    for i, potential in enumerate(e):

        fig = plt.figure(figsize=(8, 10))
        
        wave_fig_area = fig.add_axes((0.05, 0.4, 0.95, 0.35))
        wave_fig_area.axis("off")

        ax_Nyquist = fig.add_axes((0.1, 0.05, 0.375, 0.3))
        ax_Bode_R = fig.add_axes((0.6, 0.2, 0.375, 0.15))
        ax_Bode_phase = fig.add_axes((0.6, 0.05, 0.375, 0.15))


        #----------------パラメータ計算-------------------------------
        higher_potential = potential + potential_amplitude
        lower_potential = potential - potential_amplitude

        def calc_k_relax(potential):
            return reaction1.rate_fw(potential)+reaction1.rate_bw(potential)+reaction2.rate_fw(potential)+reaction2.rate_bw(potential) 

        k_relax = calc_k_relax(potential)
        k_relax_higher = calc_k_relax(higher_potential)
        k_relax_lower = calc_k_relax(lower_potential)

        def calc_theta_H(potential):
            return (reaction1.rate_fw(potential)+reaction2.rate_bw(potential))/calc_k_relax(potential)

        theta_H_cent = calc_theta_H(potential)
        theta_H_higher = calc_theta_H(higher_potential)
        theta_H_lower = calc_theta_H(lower_potential)

        #----------------パラメータ記入-------------------------------
        
        ## 説明変数
        fig.text(TEXT_POS_X1, TEXT_POS_Y1, r"$E_{\mathrm{cent.}}$"+": {:#.3} V".format(potential.value))
        fig.text(TEXT_POS_X1, TEXT_POS_Y2, r"$E_{\mathrm{amp.}}$"+": {:#.2} V".format(potential_amplitude.value))
        
        ## 
        fig.text(TEXT_POS_X1+0.15, TEXT_POS_Y1, r"$\alpha_1$"+ ": {:#.2}".format(reaction1.alpha))
        fig.text(TEXT_POS_X1+0.15, TEXT_POS_Y2, r"$\alpha_2$"+ ": {:#.2}".format(reaction2.alpha))
        
        fig.text(TEXT_POS_X1+0.25, TEXT_POS_Y1, r"$k_{0, 1}$"+": {:#.2e}".format(reaction1.k0)+" sec$^{-1}$")
        fig.text(TEXT_POS_X1+0.25, TEXT_POS_Y2, r"$k_{0, 2}$"+": {:#.2e}".format(reaction2.k0)+" sec$^{-1}$")
        
        fig.text(TEXT_POS_X1+0.45, TEXT_POS_Y1, r"$E_{0, 1}$"+ ": {:#.2}".format(reaction1.E0.value)+" V")
        fig.text(TEXT_POS_X1+0.45, TEXT_POS_Y2, r"$E_{0, 2}$"+ ": {:#.2}".format(reaction2.E0.value)+" V")

        fig.text(TEXT_POS_X1, TEXT_POS_Y3,
                r"$k_1$"+ ": {:#.2e}".format(reaction1.rate_fw(potential)) + r" sec$^{-1}$"+ \
                r" (seq. amp.: " + "{:#.2e} and {:#.2e}, ".format(
                    reaction1.rate_fw(higher_potential)- reaction1.rate_fw(potential), 
                    reaction1.rate_fw(lower_potential)- reaction1.rate_fw(potential)) + \
                r" app. amp.: $\pm$" + " {:#.2e})".format(
                    np.abs(reaction1.rate_fw(higher_potential)-reaction1.rate_fw(lower_potential))/2)
            )
        fig.text(TEXT_POS_X1, TEXT_POS_Y4,
                r"$k_{\mathrm{1r}}$"+ ": {:#.2e}".format(reaction1.rate_bw(potential)) + r" sec$^{-1}$"+ \
                r" (seq. amp.: " + "{:#.2e} and {:#.2e}, ".format(
                    reaction1.rate_bw(higher_potential)- reaction1.rate_bw(potential), 
                    reaction1.rate_bw(lower_potential)- reaction1.rate_bw(potential)) + \
                r" app. amp.: $\pm$" + " {:#.2e})".format(
                    np.abs(reaction1.rate_bw(higher_potential)-reaction1.rate_bw(lower_potential))/2)
            )
        fig.text(TEXT_POS_X1, TEXT_POS_Y5,
                r"$k_2$"+ ": {:#.2e}".format(reaction2.rate_fw(potential)) + r" sec$^{-1}$"+ \
                r" (seq. amp.: " + "{:#.2e} and {:#.2e}, ".format(
                    reaction2.rate_fw(higher_potential)- reaction2.rate_fw(potential), 
                    reaction2.rate_fw(lower_potential)- reaction2.rate_fw(potential)) + \
                r" app. amp.: $\pm$" + " {:#.2e})".format(
                    np.abs(reaction2.rate_fw(higher_potential)-reaction2.rate_fw(lower_potential))/2)
            )
        fig.text(TEXT_POS_X1, TEXT_POS_Y6,
                r"$k_{\mathrm{2r}}$"+ ": {:#.2e}".format(reaction2.rate_bw(potential)) + r" sec$^{-1}$"+ \
                r" (seq. amp.: " + "{:#.2e} and {:#.2e}, ".format(
                    reaction2.rate_bw(higher_potential)- reaction2.rate_bw(potential), 
                    reaction2.rate_bw(lower_potential)- reaction2.rate_bw(potential)) + \
                r" app. amp.: $\pm$" + " {:#.2e})".format(
                    np.abs(reaction2.rate_bw(higher_potential)-reaction2.rate_bw(lower_potential))/2)
            ) 
        ## 中間変数
        fig.text(TEXT_POS_X1, TEXT_POS_Y7,
                r"$\theta_{\mathrm{*, SS}}$"+ ": {:#.2e}".format(1-theta_H_cent) + \
                r" (seq. amp.: " + "{:#.2e} and {:#.2e}, ".format(-theta_H_lower+theta_H_cent, -theta_H_higher+theta_H_cent) + \
                r" app. amp.: $\pm$" + " {:#.2e})".format(np.abs(theta_H_higher-theta_H_lower)/2)
            )
        fig.text(TEXT_POS_X1, TEXT_POS_Y8,
                r"$\theta_{\mathrm{H, SS}}$"+ ": {:#.2e}".format(theta_H_cent) + \
                r" (seq. amp.: " + "{:#.2e} and {:#.2e}, ".format(theta_H_lower-theta_H_cent, theta_H_higher-theta_H_cent) + \
                r" app. amp.: $\pm$" + " {:#.2e})".format(np.abs(theta_H_higher-theta_H_lower)/2)
            )
        fig.text(TEXT_POS_X1, TEXT_POS_Y9,
                r"$k_\mathrm{relax}$"+ ": {:#.2e}".format(k_relax) + \
                r" (seq. amp.: " + "{:#.2e} and {:#.2e}, ".format(k_relax_lower-k_relax, k_relax_higher-k_relax) + \
                r" app. amp.: $\pm 0$"+" (const))"
            )
        

        


        #fig.text(0.05, TEXT_POS_Y1, r"$E_{\mathrm{cent.}}$"+": {:#.2} V".format(potential.value))
        # fig.text(0.05, TEXT_POS_Y2, r"$k_\mathrm{relax}$"+": {:#.2e}".format(
        #         k_relax
        #         )+" sec$^{-1}$")
        # fig.text(0.05, TEXT_POS_Y3, r"$\theta_{\mathrm{H, SS}}$"+ ": {:#.2e}".format(
        #     (reaction1.rate_fw(potential)+reaction2.rate_bw(potential))/k_relax
        # ))
        # fig.text(0.05, TEXT_POS_Y4, r"$\theta_{\mathrm{*, SS}}$"+ ": {:#.2e}".format(
        #     (reaction1.rate_bw(potential)+reaction2.rate_fw(potential))/k_relax
        # ))

        
        # fig.text(0.25, TEXT_POS_Y1, r"$k_1$"+ ": {:#.2e}".format(reaction1.rate_fw(potential)) + r" sec$^{-1}$")
        # fig.text(0.25, TEXT_POS_Y2, r"$k_{1\mathrm{r}}$"+ ": {:#.2e}".format(reaction1.rate_bw(potential)) + r" sec$^{-1}$")
        # fig.text(0.25, TEXT_POS_Y3, r"$k_2$"+ ": {:#.2e}".format(reaction2.rate_fw(potential)) + r" sec$^{-1}$")
        # fig.text(0.25, TEXT_POS_Y4, r"$k_{2\mathrm{r}}$"+ ": {:#.2e}".format(reaction2.rate_bw(potential)) + r" sec$^{-1}$")


        # fig.text(0.45, TEXT_POS_Y1, r"$\alpha_1$"+ ": {:#.2}".format(reaction1.alpha))
        # fig.text(0.45, TEXT_POS_Y2, r"$k_{0, 1}$"+": {:#.2e}".format(reaction1.k0)+" sec$^{-1}$")
        # fig.text(0.45, TEXT_POS_Y3, r"$E_{0, 1}$"+ ": {:#.2}".format(reaction1.E0.value)+" V")

        # fig.text(0.65, TEXT_POS_Y1, r"$\alpha_2$"+ ": {:#.2}".format(reaction2.alpha))
        # fig.text(0.65, TEXT_POS_Y2, r"$k_{0, 2}$"+": {:#.2e}".format(reaction2.k0)+" sec$^{-1}$")
        # fig.text(0.65, TEXT_POS_Y3, r"$E_{0, 2}$"+ ": {:#.2}".format(reaction2.E0.value)+" V")

        #----------------波形プロット-------------------------------
        wave_axes: list[plt.Axes] =\
            [
                wave_fig_area.inset_axes((0.00, 0.5, 0.33, 0.45)),
                wave_fig_area.inset_axes((0.33, 0.5, 0.33, 0.45)),
                wave_fig_area.inset_axes((0.66, 0.5, 0.33, 0.45)),
                wave_fig_area.inset_axes((0.00, 0.0, 0.33, 0.45)),
                wave_fig_area.inset_axes((0.33, 0.0, 0.33, 0.45)),
                wave_fig_area.inset_axes((0.66, 0.0, 0.33, 0.45))
            ]
        for j, frequency in enumerate(f):
            #fig, ax = cmn.create_standard_matplt_canvas()

            wave_axes[j].axis("off")

            ax_main = wave_axes[j].inset_axes((0.2, 0.4, 0.75, 0.5))
            ax_main.tick_params(top=False, labelbottom=False)

            ax_main.plot(
                np.linspace(0, SHOWN_PERIODS, len(raw_current[i][j])), 
                raw_current[i][j], 
                c = c_list[0],
                label = "Seq"
                )
            ax_main.plot(
                np.linspace(0, SHOWN_PERIODS, len(fft_current[i][j])), 
                fft_current[i][j],
                c = c_list[1],
                ls = "--", 
                label = "FFT"
                )
            # ax_main.plot(
            #     np.linspace(0, SHOWN_PERIODS, len(simd_current_via_Q[i][j])), 
            #     simd_current_via_Q[i][j],
            #     c = c_list[2],
            #     ls = ":",
            #     label = "an_via_Q"
            #     )
            # ax_main.plot(
            #     np.linspace(0, SHOWN_PERIODS, len(simd_current_via_tau[i][j])), 
            #     simd_current_via_tau[i][j],
            #     c = c_list[3],
            #     ls = "-.",
            #     label = "an_via_tau"
            #     )
            ax_main.plot(
                np.linspace(0, SHOWN_PERIODS, len(simd_current_via_differential[i][j])), 
                simd_current_via_differential[i][j],
                c = c_list[4],
                ls = LINE_STYLE_FINE_DASH,
                label = "an_via_diff"
                )
            

            y_min: float = raw_current[i][j].min().value
            y_max: float = raw_current[i][j].max().value
            

            ax_main.set_ylim(y_min*1.2, y_max*1.8)
            #ax_main.set_xlabel("Time/period [-]")
            if j == 0 or j== 3:
                ax_main.set_ylabel(r"${\Delta}j$ [mA]")

            ax_main.text(0, y_max*1.3, "Frequency: {:.2e} Hz".format(frequency.value))
            #ax_main.text(0, y_max*1.37, "Potential: {:#.2} V".format(potential.value))
            # ax_main.text(0, y_max*1.19, r"$k_\mathrm{relax}$"+": {:#.2e}".format(
            #     reaction1.rate_fw(potential)+reaction1.rate_bw(potential)+\
            #     reaction2.rate_fw(potential)+reaction2.rate_bw(potential)
            #     ))
            #ax_main.legend()
            if j==2:
                ax_main.legend(loc="upper right", bbox_to_anchor=(1, 2.2))

            ax_dev = wave_axes[j].inset_axes((0.2, 0.0, 0.75, 0.35))

            # ax_dev.plot(
            #     np.linspace(0, SHOWN_PERIODS, len(simd_current_via_Q[i][j])), 
            #     (simd_current_via_Q[i][j]-fft_current[i][j])/fft_current[i][j].max(),
            #     ls = ":", 
            #     c = c_list[2],
            #     label = "an_via_Q"
            # )

            # ax_dev.plot(
            #     np.linspace(0, SHOWN_PERIODS, len(simd_current_via_tau[i][j])), 
            #     (simd_current_via_tau[i][j]-fft_current[i][j])/fft_current[i][j].max(),
            #     ls = "-.", 
            #     c = c_list[3],
            #     label = "an_via_tau"
            # )

            ax_dev.plot(
                np.linspace(0, SHOWN_PERIODS, len(simd_current_via_differential[i][j])), 
                (simd_current_via_differential[i][j]-fft_current[i][j])/fft_current[i][j].max(),
                ls = LINE_STYLE_FINE_DASH,
                c = c_list[4],
                label = "an_via_diff"
            )

            if j ==0 or j==3:
                ax_dev.set_ylabel("Dev.")
            if j>2:
                ax_dev.set_xlabel("Time/period [-]")
            #ax_dev.set_xlabel("Time/period [-]")
            #ax_dev.set_ylabel("Normalized (Numerical $-$ FFT)[-]")
            ax_dev.set_ylim(-1, 1)
            #ax_dev.legend()

            #fig.savefig("./fig/ApproximationValidation/compare_RN{}_{:#.2}_{:.2e}.png".format(RN_number, potential.value, frequency.value), dpi = 600)

            #plt.close()

        #-----------------------Nyquistプロット------------------------
        
        imp_via_Q = int_imp.faraday_catalytic_impedance_MkIII(
            frequency=f_fine,
            potential=ec.PotentialArray([potential]),
            reaction1=reaction1,
            reaction2=reaction2,
            potential_amplitude=potential_amplitude,
            activesite_density=gamma
        )[0]

        imp_via_tau = int_imp.faraday_catalytic_impedance_MkV(
            frequency=f_fine,
            potential=ec.PotentialArray([potential]),
            reaction1=reaction1,
            reaction2=reaction2,
            potential_amplitude=potential_amplitude,
            activesite_density=gamma
        )[0]

        imp_via_differential = int_imp.faraday_catalytic_impedance_diff_eq(
            frequency=f_fine,
            potential=ec.PotentialArray([potential]),
            reaction1=reaction1,
            reaction2=reaction2,
            potential_amplitude=potential_amplitude,
            activesite_density=gamma
        )[0]

        EIS_seq = na_imp.impedance_via_sequential_current_simulation(
            potentials=ec.PotentialArray([potential]),
            frequencies=f_fine,
            reaction1=reaction1,
            reaction2=reaction2,
            potential_amplitude=potential_amplitude,
            activesite_density=gamma,
            simulation_period_number=11,
        )[0]

        max_res = max([
            EIS_seq.real_Z.float_array().max(),
            imp_via_Q.real.float_array().max(),
            imp_via_tau.real.float_array().max(),
            imp_via_differential.real.float_array().max(),
        ])

        min_res = min([
            EIS_seq.real_Z.float_array().min(),
            imp_via_Q.real.float_array().min(),
            imp_via_tau.real.float_array().min(),
            imp_via_differential.real.float_array().min(),
        ])

        x_width = max_res - min_res

        max_imag = max([
            np.abs(EIS_seq.imaginary_Z.float_array()).max(),
            np.abs(imp_via_Q.imag.float_array()).max(),
            np.abs(imp_via_tau.imag.float_array()).max(),
            np.abs(imp_via_differential.imag.float_array()).max(),
        ])

        ax_Nyquist.set_xlabel(r"Real $Z$ / Ohm")
        ax_Nyquist.set_ylabel(r"$-$Imaginary $Z$ / Ohm")
        ax_Nyquist.set_xlim(min_res - x_width*0.05, max_res+ x_width*0.05)
        ax_Nyquist.set_ylim(-1*max_imag*1.05, max_imag*1.05)


        ax_Nyquist.scatter(
            EIS_seq.real_Z.float_array(),
            -1*EIS_seq.imaginary_Z.float_array(),
            c = c_list[1],
            label = "Seq"
        )
        ax_Nyquist.plot(
            EIS_seq.real_Z.float_array(),
            -1*EIS_seq.imaginary_Z.float_array(),
            c = c_list[1],
            ls = "--"
        )

        # ax_Nyquist.scatter(
        #     imp_via_Q.real.float_array(),
        #     -1*imp_via_Q.imag.float_array(),
        #     c = c_list[2],
        #     label = "an_via_Q"
        # )
        # ax_Nyquist.plot(
        #     imp_via_Q.real.float_array(),
        #     -1*imp_via_Q.imag.float_array(),
        #     c = c_list[2],
        #     ls = ":"
        # )

        # ax_Nyquist.scatter(
        #     imp_via_tau.real.float_array(),
        #     -1*imp_via_tau.imag.float_array(),
        #     c = c_list[3],
        #     label = "an_via_tau"
        # )
        # ax_Nyquist.plot(
        #     imp_via_tau.real.float_array(),
        #     -1*imp_via_tau.imag.float_array(),
        #     c = c_list[3],
        #     ls = "-."
        # )

        ax_Nyquist.scatter(
            imp_via_differential.real.float_array(),
            -1*imp_via_differential.imag.float_array(),
            c = c_list[4],
            label = "an_via_diff"
        )
        ax_Nyquist.plot(
            imp_via_differential.real.float_array(),
            -1*imp_via_differential.imag.float_array(),
            c = c_list[4],
            ls = LINE_STYLE_FINE_DASH
        )

        #ax_Bode_phase: plt.Axes = ax_Bode_R.twinx()

        ax_Bode_R.set_xscale("log")
        ax_Bode_R.tick_params(labelbottom = False)
        ax_Bode_R.set_ylabel(r"$\mathrm{Abs}(Z)$"+" / Ohm")

        ax_Bode_phase.set_xscale("log")
        ax_Bode_phase.set_xlabel("Frequency / Hz")    
        ax_Bode_phase.set_ylabel("Phase / degree")

        ax_Bode_R.scatter(
            EIS_seq.frequency.float_array(),
            EIS_seq.abs_Z.float_array(),
            c = c_list[1],
        )
        ax_Bode_R.plot(
            EIS_seq.frequency.float_array(),
            EIS_seq.abs_Z.float_array(),
            c = c_list[1],
            ls = "--"
        )

        # ax_Bode_R.scatter(
        #     f_fine.float_array(),
        #     abs(imp_via_Q),
        #     c = c_list[2],
        # )
        # ax_Bode_R.plot(
        #     f_fine.float_array(),
        #     abs(imp_via_Q),
        #     c = c_list[2],
        #     ls = ":"
        # )

        # ax_Bode_R.scatter(
        #     f_fine.float_array(),
        #     abs(imp_via_tau),
        #     c = c_list[3],
        # )
        # ax_Bode_R.plot(
        #     f_fine.float_array(),
        #     abs(imp_via_tau),
        #     c = c_list[3],
        #     ls = "-."
        # )

        ax_Bode_R.scatter(
            f_fine.float_array(),
            abs(imp_via_differential),
            c = c_list[4],
        )
        ax_Bode_R.plot(
            f_fine.float_array(),
            abs(imp_via_differential),
            c = c_list[4],
            ls = LINE_STYLE_FINE_DASH
        )

        ax_Bode_phase.scatter(
            EIS_seq.frequency.float_array(),
            np.rad2deg(EIS_seq.phase),
            c = c_list[1],
            #marker="s"
        )
        ax_Bode_phase.plot(
            EIS_seq.frequency.float_array(),
            np.rad2deg(EIS_seq.phase),
            c = c_list[1],
            ls = "--"
        )

        # ax_Bode_phase.scatter(
        #     f_fine.float_array(),
        #     np.angle(imp_via_Q.complex_array(), deg=True),
        #     c = c_list[2],
        #     #marker="s"
        # )
        # ax_Bode_phase.plot(
        #     f_fine.float_array(),
        #     np.angle(imp_via_Q.complex_array(), deg=True),
        #     c = c_list[2],
        #     ls = ":"
        # )

        # ax_Bode_phase.scatter(
        #     f_fine.float_array(),
        #     np.angle(imp_via_tau.complex_array(), deg=True),
        #     c = c_list[3],
        #     #marker="s",
        # )
        # ax_Bode_phase.plot(
        #     f_fine.float_array(),
        #     np.angle(imp_via_tau.complex_array(), deg=True),
        #     c = c_list[3],
        #     ls = "-."
        # )

        ax_Bode_phase.scatter(
            f_fine.float_array(),
            np.angle(imp_via_differential.complex_array(), deg=True),
            c = c_list[4],
            #marker="s",
        )

        ax_Bode_phase.plot(
            f_fine.float_array(),
            np.angle(imp_via_differential.complex_array(), deg=True),
            c = c_list[4],
            ls = LINE_STYLE_FINE_DASH
        )




        fig.savefig(
            "./fig/ApproximationValidation/impedance_plot_RN{}_cnt{:.3}V_amp{:.2f}mV".format(
                RN_number, potential.value, potential_amplitude.value*1000)+output_filename_suffix+".png",
            dpi = 600
        )
        plt.close()
    del raw_current
    del fft_current
#%% RN 1-3を広い電位範囲で計算
f = ec.FrequencyArray(np.logspace(-1, 4, 6))
e = ec.PotentialArray(np.linspace(-0.2, 0.2, 9))
#e = ec.PotentialArray([0.05])
f_fine= ec.FrequencyArray(np.logspace(0, 7, 36))


potential_amplitudes = ec.PotentialArray([1E-5, 0.01])

for RN_number in [2, 1, 3, 4]:
    for potential_amplitude in potential_amplitudes:
        rn = get_reaction_parameters(RN_number)

        plot_wave_and_EIS(
            potentials=e,
            potential_amplitude=potential_amplitude,
            frequencies=f,
            frequencies_fine=f_fine,
            reaction1=rn.reaction1,
            reaction2=rn.reaction2,
            gamma=rn.gamma
        )
#%% RN 1を狭い電位範囲で計算して、capasitanceからindunctanceへの遷移を確認
f = ec.FrequencyArray(np.logspace(-1, 4, 6))
e = ec.PotentialArray(np.linspace(-0.12, -0.11, 11))
#e = ec.PotentialArray([0.05])
f_fine= ec.FrequencyArray(np.logspace(-2, 5, 36))


potential_amplitudes = ec.PotentialArray([0.01])


for RN_number in [1]:
    for potential_amplitude in potential_amplitudes:
        rn = get_reaction_parameters(RN_number)

        plot_wave_and_EIS(
            potentials=e,
            potential_amplitude=potential_amplitude,
            frequencies=f,
            frequencies_fine=f_fine,
            reaction1=rn.reaction1,
            reaction2=rn.reaction2,
            gamma=rn.gamma,
            output_filename_suffix="_capa_to_induct"
        )

#%%
#RN1
reaction1 = kin.ElementaryStep_ET(
    common.Alpha(0.5),
    common.ReactionRateCoefficient(1e3),
    ec.Potential(0.2),
    forward_direction=kin.Direction.CATHODIC
)

reaction2 = kin.ElementaryStep_ET(
    common.Alpha(0.6),
    common.ReactionRateCoefficient(1e2),
    ec.Potential(-0.2),
    forward_direction=kin.Direction.CATHODIC
)
gamma = 1e-10


#RN2
reaction1 = kin.ElementaryStep_ET(
    common.Alpha(0.5),
    common.ReactionRateCoefficient(1e3),
    ec.Potential(0.2),
    forward_direction=kin.Direction.CATHODIC
)

reaction2 = kin.ElementaryStep_ET(
    common.Alpha(0.5),
    common.ReactionRateCoefficient(1e2),
    ec.Potential(-0.2),
    forward_direction=kin.Direction.CATHODIC
)
gamma = 1e-10


#RN3
reaction1 = kin.ElementaryStep_ET(
    common.Alpha(0.5),
    common.ReactionRateCoefficient(1e0),
    ec.Potential(0.2),
    forward_direction=kin.Direction.CATHODIC
)

reaction2 = kin.ElementaryStep_ET(
    common.Alpha(0.6),
    common.ReactionRateCoefficient(1e1),
    ec.Potential(-0.2),
    forward_direction=kin.Direction.CATHODIC
)
gamma = 1e-10