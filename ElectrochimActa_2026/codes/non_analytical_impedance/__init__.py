import numpy as np
import pandas as pd
import numba

import BFC_libs.common as cmn
import BFC_libs.data_object as data
import BFC_libs.quantity_array as qa
import BFC_libs.electrochemistry as ec
from BFC_libs.physical_constants import F
import common
import general_kinetics as kin
import impedance_simulator.kinetics_to_obserbable as kin_obs

def impedance_via_sequential_current_simulation(
    potentials: ec.PotentialArray,
    frequencies: ec.FrequencyArray,

    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    potential_amplitude: ec.Potential,
    activesite_density: float,

    simulation_points_per_period: int = 3600,
    simulation_period_number: int = 100

):
    """
    時間発展計算を行い、そのフーリエ変換/逆フーリエ変換からファラデーインピーダンスを計算する関数。
    返り値は第一次元が電位ごと、第二次元が周波数ごとのインピーダンス（ec.Impedance）となるnumpy（ec.ImpedanceArray）。
    """


    if simulation_period_number < 10:
        raise ValueError("simulation_period_number must be greater than 10 to ensure convergence of the simulation.")
    if simulation_points_per_period < 360:
        raise ValueError("simulation_points_per_period must be greater than 360 to ensure convergence of the simulation.")
    
    current_list: list[qa.QArray[ec.CurrentArray]] = []
    for freq in frequencies:
        if not isinstance(freq, ec.Frequency):
            raise TypeError("The variable frequencies must be 1-dim array.")

        current_tmp = sequential_current_simulator(
            potential=potentials,
            frequency=freq,
            reaction1=reaction1,
            reaction2=reaction2,
            potential_amplitude=potential_amplitude,
            activesite_density=activesite_density,
            points_per_period=simulation_points_per_period,
            period_number=simulation_period_number
        )

        current = current_tmp
        #print(current)
        current_list.append(current)

    current_array = qa.QArray[qa.QArray[ec.CurrentArray]](np.array(current_list))

    extracted_current_fft = extraction_of_fft_current(
        current_array,
        frequencies,
        simulation_period_number 
    )

    impedances = calculate_impedance_from_fft_res(
        extracted_current_fft,
        potential_amplitude
    ).T #1次元目が電位、2次元目が周波数に変換

    tmp_list_EIS:list[ec.EIS] = []

    for impedance_at_each_potential, potential in zip(impedances, potentials):
        if not isinstance(impedance_at_each_potential, ec.ImpedanceArray):
            raise TypeError()

        tmp_EIS = ec.EIS(
            _comment = ["numericaly simulated EIS"],
            _condition = ["applied center potential = {} V".format(str(potential)),
                          "potential amplitude is {} V".format(potential_amplitude),
                          "activesite number = {}".format(activesite_density),
                          "step1: alpha = {}, E = {}, k0 = {}".format(reaction1.alpha, reaction1.E0, reaction1.k0),
                          "step2: alpha = {}, E = {}, k0 = {}".format(reaction2.alpha, reaction2.E0, reaction2.k0)],
            _original_file_path = "simd_data",
            _data_name = "simd vis sequentaill calculation at {} V".format(potential),
            _real_Z = impedance_at_each_potential.real,
            _imaginary_Z = impedance_at_each_potential.imag,
            _frequency = frequencies,
            _other_data = pd.DataFrame(),
           _applied_potential = potential, 
        )

        tmp_list_EIS.append(tmp_EIS)
    
    return data.DataArray[ec.EIS](tmp_list_EIS, ec.EIS)


def extraction_of_fft_current(
    current_array: qa.QArray[qa.QArray[ec.CurrentArray]],
    frequencies: ec.FrequencyArray,
    period_number: int
) -> list[list[tuple[ec.FrequencyArray, ec.CurrentCArray]]]:



    if current_array.ndim != 3:
        raise ValueError("current_array must be 3-dimensional array with t-axis, E-axia, and f-axis.")


    total_points = len(current_array[0][0])
    points_per_period = total_points//period_number

    if len(current_array) != len(frequencies):
            raise ValueError("current_array must have the same length as frequencies.")

    fft_res_list_each_freq:list[list[tuple[ec.FrequencyArray, ec.CurrentCArray]]] = []
    for current_each_freq, freq in zip(current_array, frequencies):
        dt = 1/points_per_period/freq.value

        fft_res_list_tmp_each_potential: list[tuple[ec.FrequencyArray, ec.CurrentCArray]] = []
        for current_each_potential in current_each_freq:
            raw_fft = np.fft.fft(current_each_potential)

            deconvoluted_freq = np.fft.fftfreq(total_points, dt)

            tmp = deconvoluted_freq < freq.value*1.01
            tmp2 = deconvoluted_freq > freq.value*0.99
            tmp3 = deconvoluted_freq == 0
            freq_mask = (tmp&tmp2)|tmp3

            extracted_fft = raw_fft[freq_mask]
            amp = extracted_fft/(total_points/2)
            extracted_freq = deconvoluted_freq[freq_mask]

            fft_res_list_tmp_each_potential.append(
                (ec.FrequencyArray(extracted_freq),
                 ec.CurrentCArray(amp)))

        fft_res_list_each_freq.append(fft_res_list_tmp_each_potential)

    return fft_res_list_each_freq

def calculate_impedance_from_fft_res(
    fft_res: list[list[tuple[ec.FrequencyArray, ec.CurrentCArray]]], #1次元目:各印加周波数ごと, 2次元目: 各電位ごと
    potential_amplitude: ec.Potential
)->qa.QArray[ec.ImpedanceArray]:
    
    impedance_list: list[list[ec.Impedance]] = []
    for fft_res_each_freq in fft_res:
        tmp: list[ec.Impedance] = []
        for fft_res_each_potential in fft_res_each_freq:
            freq = fft_res_each_potential[0][1:] #0 Hzを除去
            amp = fft_res_each_potential[1][1:]
            
            amp_total:complex = amp.sum().value
            
            amp_real = amp_total.real #cos成分
            amp_imag = amp_total.imag #-sin成分
            #print(amp_real, amp_imag)

            # この前後でrealとimagの定義が異なる
            # ここ以降、real = sin成分, imag = cos成分
            current_real = -1*amp_imag
            current_imag = amp_real
            #print(current_real, current_imag)
            total_current_amp = np.sqrt(current_real**2+current_imag**2)
            current_phase = np.arcsin(current_imag/total_current_amp)

            abs_impedance = potential_amplitude.value/total_current_amp

            tmp.append(ec.Impedance(
                abs_impedance*np.cos(current_phase)+ 1J*abs_impedance*np.sin(current_phase)
            ))
        impedance_list.append(tmp)

    return qa.QArray[ec.ImpedanceArray](impedance_list)



def sequential_current_simulator_(
    potential: ec.PotentialArray,
    frequency: ec.Frequency,

    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    potential_amplitude: ec.Potential,

    activesite_density: float,

    points_per_period: int = 3600,
    period_number: int = 10
    

)->ec.CurrentArray:
    
    #points_per_period = 3600 #仮
    
    steady_theta_center = kin_obs.two_steps_one_intermediate(
        reaction1,
        reaction2,
        potential,
        potential_amplitude
    )[common.KEY_THETA_CENTER]

    relative_time = cmn.TimeArray(np.linspace(0,period_number,points_per_period*period_number))

    period_length = 1/points_per_period/frequency

    actual_time:cmn.TimeArray = relative_time / frequency.value

    potential_meshed_tmp, actual_time_meshed_tmp = np.meshgrid(
        potential.float_array(),
        actual_time.float_array(),
        indexing='ij'
    )

    potential_meshed = qa.QArray[ec.PotentialArray](potential_meshed_tmp)
    actual_time_meshed = qa.QArray[cmn.TimeArray](actual_time_meshed_tmp)

    applied_potential:ec.PotentialArray = potential_meshed + ec.PotentialArray(potential_amplitude.value * np.sin(2 * np.pi * actual_time_meshed.float_array() * frequency.value))

    k1_meshed = reaction1.rate_fw(applied_potential)
    k2_meshed = reaction2.rate_fw(applied_potential)
    k1r_meshed = reaction1.rate_bw(applied_potential)
    k2r_meshed = reaction2.rate_bw(applied_potential)

    steady_theta:np.ndarray = (k1_meshed + k2r_meshed) / (k1_meshed + k1r_meshed + k2_meshed + k2r_meshed)
    actual_theta_mutable = np.zeros(steady_theta.shape, dtype=np.float64)
    #catalytic_current = np.zeros(steady_theta.shape, dtype=np.float64)
    #chemsorption_current = np.zeros(steady_theta.shape, dtype=np.float64)
    total_current = np.zeros(steady_theta.shape, dtype=np.float64)

    #print(total_current)

    for j, tmp_tuple in enumerate(zip(
        applied_potential,
        k1_meshed,
        k2_meshed,
        k1r_meshed,
        k2r_meshed,
        steady_theta
    )):
        e, k1, k2, k1r, k2r, steady_theta_at_e = tmp_tuple
        #print(steady_theta_at_e)
        actual_theta_mutable[j][0] = steady_theta_center[j]
        total_current[j][0] = \
            (-k1[0] * (1-actual_theta_mutable[j][0])\
             -k2[0] * actual_theta_mutable[j][0]\
            +k1r[0] * actual_theta_mutable[j][0]\
            +k2r[0] * (1-actual_theta_mutable[j][0]))\
            * activesite_density*F

        #chemsorption_current[0] = 0

        #total_current[0] = catalytic_current[0] + chemsorption_current[0] 

        for i in range(1, len(e)):
            k_relax = k1r[i] + k2[i] + k1[i] + k2r[i]
            #print(k_relax)

            theta_amp = steady_theta_at_e[i] - actual_theta_mutable[j][i-1]
            

            actual_theta_mutable[j][i] = steady_theta_at_e[i] - theta_amp* np.exp(-k_relax * period_length.value)
            #print(theta_amp)
            if actual_theta_mutable[j][i] < 0:
                actual_theta_mutable[j][i] = 0
            elif actual_theta_mutable[j][i] > 1:
                actual_theta_mutable[j][i] = 1

            total_current[j][i] = \
                (-k1[i] * (1- actual_theta_mutable[j][i])\
                -k2[i] * actual_theta_mutable[j][i]\
                + k1r[i] * actual_theta_mutable[j][i]
                +k2r[i]* (1- actual_theta_mutable[j][i]))\
                * activesite_density * F

            #chemsorption_current[j][i] = (actual_theta_mutable[j][i]-actual_theta_mutable[j][i-1])*F*activesite_density/period_length.value

            #total_current[j][i] = (catalytic_current[j][i] + chemsorption_current[j][i])
        #print(actual_theta_mutable[j][1:]-actual_theta_mutable[j][:-1])
        #print(chemsorption_current[j])
    return ec.CurrentArray(total_current)

def sequential_current_simulator(
    potential: ec.PotentialArray,
    frequency: ec.Frequency,

    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    potential_amplitude: ec.Potential,

    activesite_density: float,

    points_per_period: int = 3600,
    period_number: int = 10
    

)->qa.QArray[ec.CurrentArray]:
    """
    最初の数周期は安定していない。
    切り落として使うこと
    """
    
    period_length = 1/points_per_period/frequency.value
    steady_theta_center = kin_obs.two_steps_one_intermediate(
        reaction1,
        reaction2,
        potential,
        potential_amplitude
    )[common.KEY_THETA_CENTER]

    relative_time = cmn.TimeArray(np.linspace(0,period_number,points_per_period*period_number))

    actual_time:cmn.TimeArray = relative_time / frequency.value

    potential_meshed_tmp, actual_time_meshed_tmp = np.meshgrid(
        potential.float_array(),
        actual_time.float_array(),
        indexing='ij'
    )

    potential_meshed = qa.QArray[ec.PotentialArray](potential_meshed_tmp)
    actual_time_meshed = qa.QArray[cmn.TimeArray](actual_time_meshed_tmp)

    applied_potential:qa.QArray[ec.PotentialArray] = potential_meshed + qa.QArray[ec.PotentialArray](potential_amplitude.value * np.sin(2 * np.pi * actual_time_meshed.float_array() * frequency.value))

    k1_meshed = reaction1.rate_fw(applied_potential)
    k2_meshed = reaction2.rate_fw(applied_potential)
    k1r_meshed = reaction1.rate_bw(applied_potential)
    k2r_meshed = reaction2.rate_bw(applied_potential)


    total_current = sequential_current_simulator_core(
        
        potential_meshed.float_array(),
        actual_time_meshed.float_array(),
        frequency.value,
        k1_meshed,
        k2_meshed,
        k1r_meshed,
        k2r_meshed,
        steady_theta_center,
        potential_amplitude.value,
        activesite_density,
        period_length
    )
    return qa.QArray[ec.CurrentArray](total_current)

@numba.njit(parallel = True, cache=True)
def sequential_current_simulator_core(
    potential_meshed: np.ndarray,
    actual_time_meshed: np.ndarray,
    frequency: float,

    k1_meshed: np.ndarray,
    k2_meshed: np.ndarray,
    k1r_meshed: np.ndarray,
    k2r_meshed: np.ndarray,

    steady_theta_center: np.ndarray,

    potential_amplitude: float,

    activesite_density: float,

    period_length: float

):

    applied_potential = potential_meshed + potential_amplitude * np.sin(2 * np.pi * actual_time_meshed * frequency)


    steady_theta:np.ndarray = (k1_meshed + k2r_meshed) / (k1_meshed + k1r_meshed + k2_meshed + k2r_meshed)
    actual_theta_mutable = np.zeros(steady_theta.shape, dtype=np.float64)
    total_current = np.zeros(steady_theta.shape, dtype=np.float64)

    #print(total_current)

    for j, tmp_tuple in enumerate(zip(
        applied_potential,
        k1_meshed,
        k2_meshed,
        k1r_meshed,
        k2r_meshed,
        steady_theta
    )):
        e, k1, k2, k1r, k2r, steady_theta_at_e = tmp_tuple
        #print(steady_theta_at_e)
        actual_theta_mutable[j][0] = steady_theta_center[j]
        total_current[j][0] = \
            (-k1[0] * (1-actual_theta_mutable[j][0])\
             -k2[0] * actual_theta_mutable[j][0]\
            +k1r[0] * actual_theta_mutable[j][0]\
            +k2r[0] * (1-actual_theta_mutable[j][0]))\
            * activesite_density*F

        #chemsorption_current[0] = 0

        #total_current[0] = catalytic_current[0] + chemsorption_current[0] 

        for i in range(1, len(e)):
            k_relax = k1r[i] + k2[i] + k1[i] + k2r[i]
            #print(k_relax)

            theta_amp = steady_theta_at_e[i] - actual_theta_mutable[j][i-1]
            

            actual_theta_mutable[j][i] = steady_theta_at_e[i] - theta_amp* np.exp(-k_relax * period_length)
            #print(theta_amp)
            if actual_theta_mutable[j][i] < 0:
                actual_theta_mutable[j][i] = 0
            elif actual_theta_mutable[j][i] > 1:
                actual_theta_mutable[j][i] = 1

            total_current[j][i] = \
                (-k1[i] * (1- actual_theta_mutable[j][i])\
                -k2[i] * actual_theta_mutable[j][i]\
                + k1r[i] * actual_theta_mutable[j][i]
                +k2r[i]* (1- actual_theta_mutable[j][i]))\
                * activesite_density * F

    return total_current