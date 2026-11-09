
import numpy as np
import cmath
import scipy.integrate as integrate
from copy import deepcopy as copy
import math
#import quadpy 

from typing import Union, Final, Any
from typing import overload

from BFC_libs.physical_constants import F
import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec

import impedance_simulator as sim
import general_kinetics as kin
try:
    import kinetics_to_obserbable as kin_obs
except ImportError:
    import impedance_simulator.kinetics_to_obserbable as kin_obs
import common

def __faraday_catalytic_impedance_one_intermediate_(
    frequency: Union[float, np.ndarray],
    theta_amp: float,
    theta_center: float,
    k_decay_amp:float,
    k_decay_center:float,
    k_rds_amp: float,
    potential_amplitude: float,
    plot: bool = False 
)->Union[sim.impedance_element, np.ndarray[sim.impedance_element]]:
    """
    Cannot be used if f<1
    """
    """
    assumptions: 
    all physical parameters linearly follow to applied potential
    -> all parameters are periodic with the 2pi period

    potential input is E_amp*sin(wt) (not cosin)

    Wofram alpha output
    D[a,t]=-(k*sin(o*t) +k0)* (A*sin(o*t) - a)
    ->
    a(t) 
    = e^(k0 t - (k cos(o t))/o) -> term A
    * integral_1^t (
        1/2 e^((k cos(o ξ))/o - k0 ξ)   -> term B
        *(-A k + A cos(2 o ξ) k - 2 A k0 sin(o ξ))   ->term C 
    )dξ 
    + c_1 e^(k0 t - (k cos(o t))/o)
    where 
    a: actual theta deviation profile at a potential (f(t, f))
    k: k_decay_amp
    k0: k_decay_center
    A: theta_amp = virtual ampletude of theta at the steady state
    
    The last term can be ignored.
    theta_center + theta_amp *sin(wt) = theta at steady state at the potential (not actual)
    theta_center + a = actual theta profile

    Impedance will be provided by 
    (theta * k_rds)/E_amp
    """
    print(locals())
    TIME_POINT_NUMBER:Final = 1800

    th_ss_a = theta_amp
    th_ss_0 = theta_center
    k0 = k_decay_center
    ka = k_decay_amp

    angular_frequency = 2*np.pi*frequency
    time = np.linspace(0, 1/frequency, TIME_POINT_NUMBER) #unit: seconds
    t = time

    termA = np.exp(k0 * t - (ka * np.cos(angular_frequency*time))/angular_frequency)

    def termB(gsai):
        p = ka*np.cos(angular_frequency * gsai)/angular_frequency - k0*gsai
        return np.exp(p)/2
    
    def termC(gsai):
        a = - th_ss_a+ka
        b =   th_ss_a*np.cos(2*angular_frequency*gsai)
        c = - 2 * th_ss_a * k0 * np.sin(angular_frequency*gsai)
        return a+b+c
    
    def termBC(gsai):
        return termB(gsai)*termC(gsai)
    
    termD = th_ss_0 * np.exp(k0*time - ka*np.cos(angular_frequency*time)/angular_frequency)
    
    integrated_termBC_list = []
    if type(frequency) is np.ndarray:
        for i in range(len(time[0])):
            print(i)
            tmp = []
            def termB(gsai):
                p = ka*np.cos(angular_frequency[i] * gsai)/angular_frequency[i] - k0*gsai
                return np.exp(p)/2
            
            def termC(gsai):
                a = - th_ss_a+ka
                b =   th_ss_a*np.cos(2*angular_frequency[i]*gsai)
                c = - 2 * th_ss_a * k0 * np.sin(angular_frequency[i]*gsai)
                return a+b+c
            
            def termBC(gsai):
                return termB(gsai)*termC(gsai)
            for j in range(len(time)):
                res, _= integrate.quad(termBC, 1.0, time[j,i])
                tmp.append(res)
            integrated_termBC_list.append(np.array(tmp))
        integrated_termBC = np.array(integrated_termBC_list).T

        actual_theta = termA*integrated_termBC + th_ss_0*termD
        theta_SS = th_ss_a*np.sin(angular_frequency*time)+th_ss_0
        delta_theta = theta_SS - actual_theta
        print(delta_theta)
        rel_current = F*(actual_theta * k_rds_amp*np.sin(angular_frequency*time)\
            +(k_decay_amp*np.sin(angular_frequency*time)+k_decay_center)*delta_theta)

        impedance_list = []
        for i, _ in enumerate(rel_current[0]):
            rel_current_profile = rel_current[:, i]

            total_index = len(rel_current_profile) # 位相の計算に使う
            theoritical_max_of_potential = total_index//4 #pi/2の時にE = sin(wt)の最大値

            max = rel_current_profile.max()
            index_max = np.where(rel_current_profile == max)[0][0]

            expected_minimum_index = (index_max+total_index//2)%total_index
            #print(expected_minimum_index)
            min = rel_current_profile[expected_minimum_index]

            R = potential_amplitude/(2*max)

            dev_index = index_max - theoritical_max_of_potential
            dev_period = dev_index/total_index #最大値0.75, 最低値-0.25
            delta_phase = dev_period*2*np.pi #最大値3/2pi, 最低値-1/2pi
            #----------------急にphaseが跳んでいくことがあるので、ひとまずの対応策。将来的には不要にする---------------
            #if delta_phase > np.pi:
            #    delta_phase = -1*(delta_phase-np.pi)
            #-----------------------------------------------------------------------------------------
            #print(delta_phase)
            ret= sim.impedance_element(abs_Resistance=R, phase=delta_phase).to_complex()
            if ret is np.NaN:
                raise(ValueError("NaN was detected. R = {}, ph = {}".format(R, delta_phase)))
            impedance_list.append(ret)
        print(np.array(impedance_list))
        if plot == True:
            
            for i, cur in enumerate(rel_current.T):
                fig, ax = cmn.create_standard_matplt_canvas()
                ax.plot(time.T[i], cur)
                print(cur.max())
                print(np.where(cur == cur.max())[0][0])
        return np.array(impedance_list)
    else:
        #未完成
        raise(TypeError)


@overload
def faraday_catalytic_impedance_one_intermediate(
    frequency: Union[float, np.ndarray],
    theta_amp: float,
    theta_center: float,
    k_decay:float,
    k_rds_amp: float,
    potential_amplitude: ec.Potential,
    plot: bool = False,
)->Union[sim.impedance_element, np.ndarray[Any, complex]]:
    pass

@overload
def faraday_catalytic_impedance_one_intermediate(
    frequency: Union[float, np.ndarray],
    theta_amp: np.ndarray,
    theta_center: np.ndarray,
    k_decay:np.ndarray,
    k_rds_amp: np.ndarray,
    potential_amplitude: ec.Potential,
    plot: bool = False,
)->Union[sim.impedance_element, np.ndarray[Any, complex]]:
    pass


def faraday_catalytic_impedance_one_intermediate(
    frequency: Union[float, np.ndarray],
    theta_amp: float|np.ndarray,
    theta_center: float|np.ndarray,
    k_decay:float|np.ndarray,
    k_rds_amp: float|np.ndarray,
    potential_amplitude: ec.Potential,
    activesite_density: float,
    plot: bool = False,
)->Union[sim.impedance_element, np.ndarray[Any, complex]]:
    """
    k_relaxを用いた解析解（mathematica製）をつかってFaraday impedanceを計算する
    sequential計算の結果に全然合わない（半円ではなく、jカーブを描く）
    """
    
    TIME_POINT_NUMBER = 36000
    
    rel_time = np.linspace(0, 1, TIME_POINT_NUMBER) #unit: seconds

    observable_val_len = len(theta_amp) if isinstance(theta_amp, np.ndarray) else 1

    freq_raw = copy(frequency) #一字避難しないと、frquencyが際限なく大きくなってメモリが無くなる
    #del(frequency) #ややこしいので一回消す

    if isinstance(theta_amp, np.ndarray):
        if isinstance(freq_raw, np.ndarray):
            """
            周波数もthetaなどもarrayの時
            →3次元mesh化
            """
            time, frequency_mesh, theta_amp = np.meshgrid(rel_time, freq_raw, theta_amp, indexing='ij')
            _, _, theta_center = np.meshgrid(rel_time, freq_raw, theta_center, indexing='ij')
            _, _, k_decay= np.meshgrid(rel_time, freq_raw, k_decay, indexing='ij')
            _, _, k_rds_amp= np.meshgrid(rel_time, freq_raw, k_rds_amp, indexing='ij')

            meshed_rel_time = np.array([[rel_time for _ in range(len(freq_raw))] for _ in range(observable_val_len)]).T

        else:
            """
            周波数はスカラー、thetaなどはarrayの時
            →thetaなどだけ2次元mesh化
            """
            frequency_mesh = freq_raw
            rel_time_raw = rel_time
            theta_amp, rel_time = np.meshgrid(rel_time_raw, theta_amp, indexing='ij')
            _, theta_center = np.meshgrid(rel_time_raw, theta_center, indexing='ij')
            _, k_decay = np.meshgrid(rel_time_raw, k_decay, indexing='ij')
            _, k_rds_amp = np.meshgrid(rel_time_raw, k_rds_amp, indexing='ij')

            meshed_rel_time = np.array([rel_time for _ in range(observable_val_len)]).T

    else:
        if isinstance(freq_raw, np.ndarray):
            """
            周波数はarray、thetaなどはスカラーの時
            →周波数だけ2次元mesh化
            """
            rel_time_raw = rel_time
            rel_time, frequency_mesh = np.meshgrid(rel_time, freq_raw, indexing='ij')

            meshed_rel_time = np.array([rel_time_raw for _ in range(len(freq_raw))]).T 

        else:
            frequency_mesh = freq_raw
            """
            全部スカラー
            →mesh化不要
            """
            meshed_rel_time = rel_time
    
    
    time = meshed_rel_time*(1/frequency_mesh)
    angular_frequency = 2*np.pi*frequency_mesh

    termA = theta_amp*k_decay**2 * np.sin(angular_frequency*time)/(k_decay**2 + angular_frequency**2)
    termB = theta_amp*k_decay * np.cos(angular_frequency*time)/(k_decay**2 + angular_frequency**2)

    relative_actual_theta = termA - termB
    relative_steady_theta = theta_amp*np.sin(angular_frequency*time)
    actual_theta = relative_actual_theta + theta_center
    
    reaction_rate_theta = k_decay * (theta_amp*np.sin(angular_frequency*time) - relative_actual_theta)
    relative_reaction_rate_rds = actual_theta*k_rds_amp*np.sin(angular_frequency*time)

    rel_current = activesite_density*F*(reaction_rate_theta + relative_reaction_rate_rds)
    rel_current_T = rel_current.T
    #print(rel_current.shape)

    impedance_list:list[list[complex]]|list[complex] = []

    if rel_current_T.ndim == 3:
        for j, _ in enumerate(rel_current_T):
            tmp = []
            for i in range(len(rel_current_T[0])):
                rel_current_profile = rel_current_T[j, i]

                total_index = len(rel_current_profile) # 位相の計算に使う
                theoritical_max_of_potential = total_index//4 #pi/2の時にE = sin(wt)の最大値

                max = rel_current_profile.max()
                index_max = np.where(rel_current_profile == max)[0][0]

                expected_minimum_index = (index_max+total_index//2)%total_index
                #print(expected_minimum_index)
                min = rel_current_profile[expected_minimum_index]

                R = potential_amplitude.value/(2*max)

                dev_index = index_max - theoritical_max_of_potential
                dev_period = dev_index/total_index #最大値0.75, 最低値-0.25
                delta_phase = dev_period*2*np.pi #最大値3/2pi, 最低値-1/2pi
                #print(delta_phase)

                ret= sim.impedance_element(abs_Resistance=R, phase=delta_phase).to_complex()
                tmp.append(ret)
            impedance_list.append(tmp)
    elif rel_current_T.ndim == 2:
        for i in range(len(rel_current_T)):
            rel_current_profile = rel_current_T[i]

            total_index = len(rel_current_profile) # 位相の計算に使う
            theoritical_max_of_potential = total_index//4 #pi/2の時にE = sin(wt)の最大値

            max = rel_current_profile.max()
            index_max = np.where(rel_current_profile == max)[0][0]

            expected_minimum_index = (index_max+total_index//2)%total_index
            #print(expected_minimum_index)
            min = rel_current_profile[expected_minimum_index]

            R = potential_amplitude.value/(2*max)

            dev_index = index_max - theoritical_max_of_potential
            dev_period = dev_index/total_index #最大値0.75, 最低値-0.25
            delta_phase = dev_period*2*np.pi #最大値3/2pi, 最低値-1/2pi
            #print(delta_phase)

            ret= sim.impedance_element(abs_Resistance=R, phase=delta_phase).to_complex()
            
            impedance_list.append(ret)
    else:
        raise ValueError("shape of rel_current.T is {}".format(rel_current_T.shape))
    
    
    if plot == True:
        
        for i, cur in enumerate(relative_steady_theta.T):
            fig, ax = cmn.create_standard_matplt_canvas()
            ax.scatter(time.T[i], cur)
            #print(cur)
            #ax.plot(time.T[i], relative_actual_theta.T[i])
            #ax.plot(time.T[i], reaction_rate_theta.T[i])
            #ax.plot(time.T[i], relative_actual_theta.T[i] - cur)
            #print(cur.max())
            #print(np.where(cur == cur.max())[0][0])
    return np.array(impedance_list)



def faraday_catalytic_impedance_one_intermediate_single_poisoning(
    frequency: Union[float, np.ndarray],
    theta_amp: float|np.ndarray,
    theta_center: float|np.ndarray,
    k_decay:float|np.ndarray,
    k_rds_amp: float|np.ndarray,

    theta_poisoned: float|np.ndarray,
    theta_amp_poisoned: float|np.ndarray,
    k_decay_poisoning: float|np.ndarray,
    
    potential_amplitude: ec.Potential,
    activesite_density: float,
    plot: bool = False,
)->Union[sim.impedance_element, np.ndarray[Any, complex]]:
    
    TIME_POINT_NUMBER = 36000
    
    rel_time = np.linspace(0, 1, TIME_POINT_NUMBER) #unit: seconds

    observable_val_len = len(theta_amp) if isinstance(theta_amp, np.ndarray) else 1

    freq_raw = copy(frequency) #一字避難しないと、frquencyが際限なく大きくなってメモリが無くなる
    #del(frequency) #ややこしいので一回消す

    if isinstance(theta_amp, np.ndarray):
        if isinstance(freq_raw, np.ndarray):
            """
            周波数もthetaなどもarrayの時
            →3次元mesh化
            """
            time, frequency_mesh, theta_amp = np.meshgrid(rel_time, freq_raw, theta_amp, indexing='ij')
            _, _, theta_center = np.meshgrid(rel_time, freq_raw, theta_center, indexing='ij')
            _, _, k_decay= np.meshgrid(rel_time, freq_raw, k_decay, indexing='ij')
            _, _, k_rds_amp= np.meshgrid(rel_time, freq_raw, k_rds_amp, indexing='ij')
            _, _, theta_poisoned = np.meshgrid(rel_time, freq_raw, theta_poisoned, indexing='ij')
            _, _, theta_amp_poisoned = np.meshgrid(rel_time, freq_raw, theta_amp_poisoned, indexing='ij')
            _, _, k_decay_poisoning = np.meshgrid(rel_time, freq_raw, k_decay_poisoning, indexing='ij')

            meshed_rel_time = np.array([[rel_time for _ in range(len(freq_raw))] for _ in range(observable_val_len)]).T

        else:
            """
            周波数はスカラー、thetaなどはarrayの時
            →thetaなどだけ2次元mesh化
            """
            frequency_mesh = freq_raw
            rel_time_raw = rel_time
            theta_amp, rel_time = np.meshgrid(rel_time_raw, theta_amp, indexing='ij')
            _, theta_center = np.meshgrid(rel_time_raw, theta_center, indexing='ij')
            _, k_decay = np.meshgrid(rel_time_raw, k_decay, indexing='ij')
            _, k_rds_amp = np.meshgrid(rel_time_raw, k_rds_amp, indexing='ij')
            _, theta_poisoned = np.meshgrid(rel_time_raw, theta_poisoned, indexing='ij')
            _, theta_amp_poisoned = np.meshgrid(rel_time_raw, theta_amp_poisoned, indexing='ij')
            _, k_decay_poisoning = np.meshgrid(rel_time_raw, k_decay_poisoning, indexing='ij')

            meshed_rel_time = np.array([rel_time for _ in range(observable_val_len)]).T

    else:
        if isinstance(freq_raw, np.ndarray):
            """
            周波数はarray、thetaなどはスカラーの時
            →周波数だけ2次元mesh化
            """
            rel_time_raw = rel_time
            rel_time, frequency_mesh = np.meshgrid(rel_time, freq_raw, indexing='ij')

            meshed_rel_time = np.array([rel_time_raw for _ in range(len(freq_raw))]).T 

        else:
            frequency_mesh = freq_raw
            """
            全部スカラー
            →mesh化不要
            """
            meshed_rel_time = rel_time
    
    
    time = meshed_rel_time*(1/frequency_mesh)
    angular_frequency = 2*np.pi*frequency_mesh

    # H吸着体
    termA = theta_amp*k_decay**2 * np.sin(angular_frequency*time)/(k_decay**2 + angular_frequency**2)
    termB = theta_amp*k_decay * np.cos(angular_frequency*time)/(k_decay**2 + angular_frequency**2)

    relative_actual_theta = termA - termB
    relative_steady_theta = theta_amp*np.sin(angular_frequency*time)
    actual_theta = relative_actual_theta + theta_center
    
    reaction_rate_theta = k_decay * (theta_amp*np.sin(angular_frequency*time) - relative_actual_theta)
    relative_reaction_rate_rds = actual_theta*k_rds_amp*np.sin(angular_frequency*time)

    # 被毒
    termA_poison = theta_amp_poisoned*k_decay_poisoning**2 * np.sin(angular_frequency*time)/(k_decay_poisoning**2 + angular_frequency**2)
    termB_poison = theta_amp_poisoned*k_decay_poisoning * np.cos(angular_frequency*time)/(k_decay_poisoning**2 + angular_frequency**2)

    relative_actual_theta_poison = termA_poison - termB_poison
    relative_steady_theta_poison = theta_amp_poisoned*np.sin(angular_frequency*time)

    reaction_rate_theta_poison = k_decay_poisoning * (relative_steady_theta_poison - relative_actual_theta_poison)

    rel_current = activesite_density*F*(reaction_rate_theta + relative_reaction_rate_rds + reaction_rate_theta_poison)
    rel_current_T = rel_current.T
    #print(rel_current.shape)

    impedance_list:list[list[complex]]|list[complex] = []

    if rel_current_T.ndim == 3:
        for j, _ in enumerate(rel_current_T):
            tmp = []
            for i in range(len(rel_current_T[0])):
                rel_current_profile = rel_current_T[j, i]

                total_index = len(rel_current_profile) # 位相の計算に使う
                theoritical_max_of_potential = total_index//4 #pi/2の時にE = sin(wt)の最大値

                max = rel_current_profile.max()
                index_max = np.where(rel_current_profile == max)[0][0]

                expected_minimum_index = (index_max+total_index//2)%total_index
                #print(expected_minimum_index)
                min = rel_current_profile[expected_minimum_index]

                R = potential_amplitude.value/(2*max)

                dev_index = index_max - theoritical_max_of_potential
                dev_period = dev_index/total_index #最大値0.75, 最低値-0.25
                delta_phase = dev_period*2*np.pi #最大値3/2pi, 最低値-1/2pi
                #print(delta_phase)

                ret= sim.impedance_element(abs_Resistance=R, phase=delta_phase).to_complex()
                tmp.append(ret)
            impedance_list.append(tmp)
    elif rel_current_T.ndim == 2:
        for i in range(len(rel_current_T)):
            rel_current_profile = rel_current_T[i]

            total_index = len(rel_current_profile) # 位相の計算に使う
            theoritical_max_of_potential = total_index//4 #pi/2の時にE = sin(wt)の最大値

            max = rel_current_profile.max()
            index_max = np.where(rel_current_profile == max)[0][0]

            expected_minimum_index = (index_max+total_index//2)%total_index
            #print(expected_minimum_index)
            min = rel_current_profile[expected_minimum_index]

            R = potential_amplitude.value/(2*max)

            dev_index = index_max - theoritical_max_of_potential
            dev_period = dev_index/total_index #最大値0.75, 最低値-0.25
            delta_phase = dev_period*2*np.pi #最大値3/2pi, 最低値-1/2pi
            #print(delta_phase)

            ret= sim.impedance_element(abs_Resistance=R, phase=delta_phase).to_complex()
            
            impedance_list.append(ret)
    else:
        raise ValueError("shape of rel_current.T is {}".format(rel_current_T.shape))
    
    
    if plot == True:
        
        for i, cur in enumerate(relative_steady_theta.T):
            fig, ax = cmn.create_standard_matplt_canvas()
            ax.scatter(time.T[i], cur)
            #print(cur)
            #ax.plot(time.T[i], relative_actual_theta.T[i])
            #ax.plot(time.T[i], reaction_rate_theta.T[i])
            #ax.plot(time.T[i], relative_actual_theta.T[i] - cur)
            #print(cur.max())
            #print(np.where(cur == cur.max())[0][0])
    return np.array(impedance_list)

def faraday_catalytic_impedance_MkII(
    frequency: np.ndarray,
    potential: ec.PotentialArray,

    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    potential_amplitude: ec.Potential,
    activesite_density: float,
    plot: bool = False,
):
    """
    無印版と同じロジックだが、計算式はChatGPT製
    なぜ違う式が導出されたのかは結局分からずじまいだが、結局合わないので却って問題なかった
    """
    meshed_E, meshed_f = np.meshgrid(potential, frequency, indexing='ij')
    meshed_omega = 2*np.pi*meshed_f.float_array()

    k1="k1"
    k2="k2"
    k1r="k1r"
    k2r="k2r"

    KEYS=[k1, k2, k1r, k2r]

    get_k={
        k1: reaction1.rate_fw,
        k2: reaction2.rate_fw,
        k1r: reaction1.rate_bw,
        k2r: reaction2.rate_bw,
    }

    k_cent: dict[str, float|np.ndarray] = {}
    k_amp: dict[str, float|np.ndarray] = {}

    for key in KEYS:
        k_cent[key] = get_k[key](meshed_E)
        k_pos = get_k[key](meshed_E + potential_amplitude)
        k_neg = get_k[key](meshed_E - potential_amplitude)
        k_amp[key] = (k_pos - k_neg)/2

    observable_parameters = kin_obs.two_steps_one_intermediate(
        reaction1,
        reaction2,
        applied_potential = meshed_E,
        potential_amplitude = potential_amplitude
    )
    k_relax = observable_parameters[common.KEY_K_DECAY]
    theta_cent = observable_parameters[common.KEY_THETA_CENTER]
    theta_amp = observable_parameters[common.KEY_THETA_AMP]

    # print("k_relax: ", k_relax)
    # print("theta_cent: ", theta_cent)
    # print("theta_amp: ", theta_amp)
    # print("k_cent: ", k_cent)

    # print("k_amp: ", k_amp)
    sum_coefficient_cent = \
        -k_cent[k1] + k_cent[k2] - k_cent[k1r] + k_cent[k2r]
    
    sum_coefficient_amp = \
        -k_amp[k1] + k_amp[k2] - k_amp[k1r] + k_amp[k2r]
    
    
    
    termB=(theta_amp*k_relax)/(k_relax**2 + meshed_omega**2)

    tmp = termB**2 + (termB*k_relax)**2

    termC = np.sqrt(tmp)

    phi = np.arcsin(-(termB)/np.sqrt(termB**2+(termB*k_relax)**2))

    #print(termC)

    real_coefficeint_1 = termC*sum_coefficient_cent*np.cos(phi)
    real_coefficeint_2 = sum_coefficient_amp*theta_cent+(k_amp[k1]-k_amp[k2r])
    real_coefficeint = real_coefficeint_1 + real_coefficeint_2

    imag_coefficient = termC * sum_coefficient_cent * np.sin(phi)
    # print(real_coefficeint_1)
    # print(real_coefficeint_2)
    # print(real_coefficeint)
    # print(imag_coefficient)

    current_amp = np.sqrt(real_coefficeint**2 + imag_coefficient**2)*F*activesite_density
    phase_diff = np.arcsin(imag_coefficient/np.sqrt(real_coefficeint**2+imag_coefficient**2))
    
    R_abs = potential_amplitude.value / current_amp

    return R_abs*np.cos(phase_diff) + 1j*R_abs*np.sin(phase_diff)
    
def faraday_catalytic_impedance_MkIII(
    frequency: np.ndarray,
    potential: ec.PotentialArray,

    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    potential_amplitude: ec.Potential,
    activesite_density: float,
    plot: bool = False,
)->ec.ImpedanceArray:
    """
    k_rekaxを使った解析解をやめて、等価回路のキャパシタ容量QをΔθから求める方法
    悪くはないが、たまに大きくずれる。
    基本的にはMkVのほうがよく合うが、いずれかの被覆率がほぼ1（0.95以上？）かつk_relaxが一つの反応速度だけでは記述不可能なときは、こちらの方がよく合う。
    """
    
    meshed_E, meshed_f = np.meshgrid(potential, frequency, indexing='ij')
    meshed_omega = 2*np.pi*meshed_f.float_array()

    observable_parameters = kin_obs.two_steps_one_intermediate(
        reaction1,
        reaction2,
        applied_potential = meshed_E,
        potential_amplitude = potential_amplitude
    )
    k_relax = observable_parameters[common.KEY_K_DECAY]
    theta_cent = observable_parameters[common.KEY_THETA_CENTER]
    theta_amp = observable_parameters[common.KEY_THETA_AMP]

    ks_ano = np.array([
        reaction1.rate_fw(meshed_E+potential_amplitude),
        reaction1.rate_bw(meshed_E+potential_amplitude),
        reaction2.rate_fw(meshed_E+potential_amplitude),
        reaction2.rate_bw(meshed_E+potential_amplitude),
    ])

    ks_cath = np.array([
        reaction1.rate_fw(meshed_E-potential_amplitude),
        reaction1.rate_bw(meshed_E-potential_amplitude),
        reaction2.rate_fw(meshed_E-potential_amplitude),
        reaction2.rate_bw(meshed_E-potential_amplitude),
    ])

    signs_k, _, _ = np.meshgrid(
        np.array([1, -1, 1, -1]),
        potential, frequency, indexing='ij')
    
    signs_theta, _, _ = np.meshgrid(
        np.array([-1, 1, 1, -1]),
        potential, frequency, indexing='ij')
    
    ones_theta, _, _ = np.meshgrid(
        np.array([1, 0, 0, 1]),
        potential, frequency, indexing='ij')
    
    theta_ss_ano:np.ndarray = (ks_ano[0]+ks_ano[3])/ks_ano.sum(axis=0)
    theta_ss_cath:np.ndarray = (ks_cath[0] + ks_cath[3]) / ks_cath.sum(axis=0)
    
    ks_signed_ano:np.ndarray = signs_k * ks_ano
    ks_signed_cath:np.ndarray = signs_k * ks_cath

    coverages_ano:np.ndarray = \
        ones_theta + signs_theta*theta_ss_ano
    
    coverages_cath:np.ndarray = \
        ones_theta + signs_theta*theta_ss_cath
    
    delta_current_high_freq:np.ndarray = ((ks_signed_cath - ks_signed_ano)*(ones_theta + signs_theta*theta_cent)).sum(axis=0)*F*activesite_density
    delta_current_low_freq:np.ndarray = (coverages_cath*ks_signed_cath - coverages_ano*ks_signed_ano).sum(axis=0)*F*activesite_density

    Rct_high_freq = potential_amplitude.value*2/delta_current_high_freq
    Rct_low_freq = potential_amplitude.value*2/delta_current_low_freq - Rct_high_freq

    C_seudo = abs(F*activesite_density*theta_amp/potential_amplitude.value/2)
    #print(Rct_low_freq)
    return ec.ImpedanceArray(Rct_high_freq + Rct_low_freq/(1+1j*meshed_omega*Rct_low_freq*C_seudo))
    
def faraday_catalytic_impedance_diff_eq(
    frequency: np.ndarray,
    potential: ec.PotentialArray,

    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    potential_amplitude: ec.Potential,
    activesite_density: float,
    plot: bool = False,
)->ec.ImpedanceArray:
    """
    <s>無印版の改良のはずだが、具体的になにを改良したか思い出せないし、結局合わないので問題ない</s>
    今まで計算ミスをしていただけで、きちんと近似解をを計算できる関数。
    多くの例で誤差が少ない。
    """
    meshed_E, meshed_f = np.meshgrid(potential, frequency, indexing='ij')
    meshed_omega = 2*np.pi*meshed_f.float_array()

    k1="k1"
    k2="k2"
    k1r="k1r"
    k2r="k2r"

    KEYS=[k1, k2, k1r, k2r]

    get_k={
        k1: reaction1.rate_fw,
        k2: reaction2.rate_fw,
        k1r: reaction1.rate_bw,
        k2r: reaction2.rate_bw,
    }

    k_cent: dict[str, float|np.ndarray] = {}
    k_ano: dict[str, float|np.ndarray] = {}
    k_cath: dict[str, float|np.ndarray] = {}
    k_amp: dict[str, float|np.ndarray] = {}


    for key in KEYS:
        k_cent[key] = get_k[key](meshed_E)
        k_ano[key] = get_k[key](meshed_E + potential_amplitude)
        k_cath[key] = get_k[key](meshed_E - potential_amplitude)
        k_amp[key] = -(k_ano[key] - k_cath[key])/2
        #k_cent[key] = (k_ano[key] + k_cath[key])/2

    observable_parameters = kin_obs.two_steps_one_intermediate(
        reaction1,
        reaction2,
        applied_potential = meshed_E,
        potential_amplitude = potential_amplitude
    )
    k_relax = observable_parameters[common.KEY_K_DECAY]
    theta_cent = observable_parameters[common.KEY_THETA_CENTER]
    #theta_amp = observable_parameters[common.KEY_THETA_AMP]
    theta_cath = (k_cath[k1] + k_cath[k2r])/(k_cath[k1]+k_cath[k1r]+k_cath[k2]+k_cath[k2r])
    theta_ano = (k_ano[k1] + k_ano[k2r])/(k_ano[k1]+k_ano[k1r]+k_ano[k2]+k_ano[k2r])
    theta_amp = (theta_cath - theta_ano)/2

    current_cath \
        = (k_cath[k1]*(1-theta_cath)\
        +k_cath[k2]*theta_cath\
        -k_cath[k1r]*theta_cath\
        -k_cath[k2r]*(1-theta_cath))\
            *F*activesite_density
    
    current_ano \
        = (k_ano[k1]*(1-theta_ano)\
        +k_ano[k2]*theta_ano\
        -k_ano[k1r]*theta_ano\
        -k_ano[k2r]*(1-theta_ano))\
            *F*activesite_density

    delta_current_low_freq_lim = (current_cath - current_ano)/2

    # print("k_relax: ", k_relax)
    # print("theta_cent: ", theta_cent)
    # print("theta_amp: ", theta_amp)
    # print("k_cent: ", k_cent)

    # print("k_amp: ", k_amp)
    sum_coefficient_cent = \
        (-k_cent[k1] + k_cent[k2] - k_cent[k1r] + k_cent[k2r])
    
    sum_coefficient_amp = \
        (-k_amp[k1] + k_amp[k2] - k_amp[k1r] + k_amp[k2r])
    
    
    
    termB=(theta_amp*k_relax)/(k_relax**2 + meshed_omega**2)

    tmp = (meshed_omega)**2 + (k_relax)**2
    termC = termB*np.sqrt(tmp)

    termC_lfl = theta_amp
    
    phi = np.arctan(-(meshed_omega)/(k_relax))

    #print(termC)

    termG = termC*sum_coefficient_cent
    termH = sum_coefficient_amp*theta_cent+(k_amp[k1]-k_amp[k2r])

    termG_lfl = termC_lfl*sum_coefficient_cent

    # steady_rate_cath = ((1-theta_cath)*k_cath[k1] + theta_cath*k_cath[k2] - theta_cath*k_cath[k1r] - (1-theta_cath)*k_cath[k2r])
    # steady_rate_ano = ((1-theta_ano)*k_ano[k1] + theta_ano*k_ano[k2] - theta_ano*k_ano[k1r] - (1-theta_ano)*k_ano[k2r])
    # delta_steady_rate = (steady_rate_cath - steady_rate_ano)/2


    real_coefficient = termG*np.cos(phi) + termH
    imag_coefficient = termG*np.sin(phi)

    current_amp = np.sqrt(real_coefficient**2 + imag_coefficient**2)*F*activesite_density
    #current_amp = current_amp + delta_current_low_freq_lim - np.sqrt(termG_lfl**2 + termH**2 + 2*termG_lfl*termH)

    phase_diff = np.arctan(imag_coefficient/real_coefficient)
    R_abs = potential_amplitude.value / current_amp

    # print("potential: {}".format(potential))
    # print("termH_prime: {}".format(termH_prime))
    # print("termH: {}".format(termH))

    return ec.ImpedanceArray(R_abs*np.cos(phase_diff) - 1j*R_abs*np.sin(phase_diff)) #割るので回転方向は逆

def faraday_catalytic_impedance_diff_eq_prime_(
    frequency: ec.FrequencyArray,
    potential: ec.PotentialArray,

    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    potential_amplitude: ec.Potential,
    activesite_density: float,
    plot: bool = False,
)->ec.ImpedanceArray:
    """
    MkIVのk傾き計算の方法を変えたバージョン
    """
    meshed_E_tmp, meshed_f_tmp = np.meshgrid(potential, frequency, indexing='ij')
    meshed_E: ec.PotentialArray = meshed_E_tmp
    meshed_f: ec.FrequencyArray = meshed_f_tmp #そのままだと型ヒントが機能しないため

    meshed_omega = 2*np.pi*meshed_f.float_array()

    k1="k1"
    k2="k2"
    k1r="k1r"
    k2r="k2r"

    KEYS=[k1, k2, k1r, k2r]

    get_k: dict[str, function]={
        k1: reaction1.rate_fw,
        k2: reaction2.rate_fw,
        k1r: reaction1.rate_bw,
        k2r: reaction2.rate_bw,
    }

    k_amp: dict[str, float|np.ndarray] = {
        k1: reaction1.accelaration_fw(meshed_E)*potential_amplitude.value,
        k2: reaction2.accelaration_fw(meshed_E)*potential_amplitude.value,
        k1r: reaction1.accelaration_bw(meshed_E)*potential_amplitude.value,
        k2r: reaction2.accelaration_bw(meshed_E)*potential_amplitude.value,
    }

    k_cent: dict[str, float|np.ndarray] = {}
    k_ano: dict[str, float|np.ndarray] = {}
    k_cath: dict[str, float|np.ndarray] = {}


    for key in KEYS:
        k_cent[key] = get_k[key](meshed_E)
        k_ano[key] = get_k[key](meshed_E + potential_amplitude)
        k_cath[key] = get_k[key](meshed_E - potential_amplitude)

    observable_parameters = kin_obs.two_steps_one_intermediate(
        reaction1,
        reaction2,
        applied_potential = meshed_E,
        potential_amplitude = potential_amplitude
    )
    k_relax = observable_parameters[common.KEY_K_DECAY]
    theta_cent = observable_parameters[common.KEY_THETA_CENTER]
    theta_cath = (k_cath[k1] + k_cath[k2r])/(k_cath[k1]+k_cath[k1r]+k_cath[k2]+k_cath[k2r])
    theta_ano = (k_ano[k1] + k_ano[k2r])/(k_ano[k1]+k_ano[k1r]+k_ano[k2]+k_ano[k2r])
    theta_amp = (theta_cath - theta_ano)/2

    # print("k_relax: ", k_relax)
    # print("theta_cent: ", theta_cent)
    # print("theta_amp: ", theta_amp)
    print("k_cent: ", k_cent[k1])
    print("k_amp: ", k_amp)
    print("k_cath-ano: ", k_cath)
    sum_coefficient_cent = \
        -k_cent[k1] + k_cent[k2] - k_cent[k1r] + k_cent[k2r]
    
    sum_coefficient_amp = \
        -k_amp[k1] + k_amp[k2] - k_amp[k1r] + k_amp[k2r]
    
    
    
    termB=(theta_amp*k_relax)/(k_relax**2 + meshed_omega**2)

    tmp = (meshed_omega)**2 + (k_relax)**2
    termC = termB*np.sqrt(tmp)
    
    phi = np.arctan(-(meshed_omega)/(k_relax))

    termG = termC*sum_coefficient_cent
    termH = sum_coefficient_amp*theta_cent+(k_amp[k1]-k_amp[k2r])

    real_coefficient = termG*np.cos(phi) + termH
    imag_coefficient = termG*np.sin(phi)

    current_amp = np.sqrt(real_coefficient**2 + imag_coefficient**2)*F*activesite_density

    phase_diff = np.arctan(imag_coefficient/real_coefficient)
    R_abs = potential_amplitude.value / current_amp


    return ec.ImpedanceArray(R_abs*np.cos(phase_diff) - 1j*R_abs*np.sin(phase_diff)) #割るので回転方向は逆


def faraday_catalytic_impedance_MkV(
    frequency: np.ndarray,
    potential: ec.PotentialArray,

    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    potential_amplitude: ec.Potential,
    activesite_density: float,
    plot: bool = False,
)->ec.ImpedanceArray:
    """
    擬容量Qをk_relakxと等価回路から求める方法
    大体こいつで合うが、例外あり（MkIIIのコメント参照）
    """
    
    meshed_E, meshed_f = np.meshgrid(potential, frequency, indexing='ij')
    meshed_omega = 2*np.pi*meshed_f.float_array()

    observable_parameters = kin_obs.two_steps_one_intermediate(
        reaction1,
        reaction2,
        applied_potential = meshed_E,
        potential_amplitude = potential_amplitude
    )
    k_relax = observable_parameters[common.KEY_K_DECAY]
    theta_cent = observable_parameters[common.KEY_THETA_CENTER]
    theta_amp = observable_parameters[common.KEY_THETA_AMP]

    ks_ano = np.array([
        reaction1.rate_fw(meshed_E+potential_amplitude),
        reaction1.rate_bw(meshed_E+potential_amplitude),
        reaction2.rate_fw(meshed_E+potential_amplitude),
        reaction2.rate_bw(meshed_E+potential_amplitude),
    ])

    ks_cath = np.array([
        reaction1.rate_fw(meshed_E-potential_amplitude),
        reaction1.rate_bw(meshed_E-potential_amplitude),
        reaction2.rate_fw(meshed_E-potential_amplitude),
        reaction2.rate_bw(meshed_E-potential_amplitude),
    ])

    signs_k, _, _ = np.meshgrid(
        np.array([1, -1, 1, -1]),
        potential, frequency, indexing='ij')
    
    signs_theta, _, _ = np.meshgrid(
        np.array([-1, 1, 1, -1]),
        potential, frequency, indexing='ij')
    
    ones_theta, _, _ = np.meshgrid(
        np.array([1, 0, 0, 1]),
        potential, frequency, indexing='ij')
    
    theta_ss_ano:np.ndarray = (ks_ano[0]+ks_ano[3])/ks_ano.sum(axis=0)
    theta_ss_cath:np.ndarray = (ks_cath[0] + ks_cath[3]) / ks_cath.sum(axis=0)
    
    ks_signed_ano:np.ndarray = signs_k * ks_ano
    ks_signed_cath:np.ndarray = signs_k * ks_cath

    coverages_ano:np.ndarray = \
        ones_theta + signs_theta*theta_ss_ano
    
    coverages_cath:np.ndarray = \
        ones_theta + signs_theta*theta_ss_cath
    
    delta_current_high_freq:np.ndarray = ((ks_signed_cath - ks_signed_ano)*(ones_theta + signs_theta*theta_cent)).sum(axis=0)*F*activesite_density
    delta_current_low_freq:np.ndarray = (coverages_cath*ks_signed_cath - coverages_ano*ks_signed_ano).sum(axis=0)*F*activesite_density

    Rct_high_freq = potential_amplitude.value*2/delta_current_high_freq
    Rct_low_freq = potential_amplitude.value*2/delta_current_low_freq - Rct_high_freq

    C_seudo = abs(Rct_low_freq/(Rct_high_freq*(Rct_low_freq-Rct_high_freq)*k_relax))
    #print(Rct_low_freq)
    return ec.ImpedanceArray(Rct_high_freq + Rct_low_freq/(1+1j*meshed_omega*Rct_low_freq*C_seudo))

def simulate_total_impedance_single_intermediate_hybrid(
    frequency: ec.FrequencyArray,
    potential: ec.PotentialArray,
    
    # kinetic parameters
    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    
    # constant for Faraday impedance
    potential_amplitude: float|ec.Potential,

    # variables of circuit elements
    C_dl: float|ec.Impedance,
    R_sol: float|ec.Resistance,

    # variable: activesite density
    active_site_density: float,
    
    # variables for transission line model
    R_pore: float|ec.Resistance,
    lambda_fac: float,
):
    """
    改良されたfaraday_catalytic_impedance関数を使ったsimulator
    """
    faraday_impedance_via_tau = \
        faraday_catalytic_impedance_MkV(
            frequency,
            potential,
            reaction1,
            reaction2,
            potential_amplitude=potential_amplitude,
            activesite_density=active_site_density
        ) # type: ignore
    
    faraday_impedance_via_Q = \
        faraday_catalytic_impedance_MkIII(
            frequency,
            potential,
            reaction1,
            reaction2,
            potential_amplitude=potential_amplitude,
            activesite_density=active_site_density
        )
    
    meshed_E, _ = np.meshgrid(potential, frequency, indexing='ij')

    observable_parameters = kin_obs.two_steps_one_intermediate(
        reaction1,
        reaction2,
        applied_potential = meshed_E,
        potential_amplitude = potential_amplitude
    )
    
    k_relax = observable_parameters[common.KEY_K_DECAY]
    theta_cent = observable_parameters[common.KEY_THETA_CENTER]

    k1 = reaction1.rate_fw(ec.PotentialArray(meshed_E))
    k2 = reaction2.rate_fw(ec.PotentialArray(meshed_E))
    k1r = reaction1.rate_bw(ec.PotentialArray(meshed_E))
    k2r = reaction2.rate_bw(ec.PotentialArray(meshed_E))

    mask_high_theta = theta_cent > 0.95
    mask_low_theta = theta_cent < 0.05
    
    

    Z_cdl = 1/(1j * frequency.float_array() * 2*np.pi * C_dl)
    
    inv_faraday_impedance= 1/faraday_impedance
    inv_Z_cdl = 1/Z_cdl
    #print(type(inv_faraday_impedance), type(inv_Z_cdl))
    
    Z_B = 1/(inv_faraday_impedance + inv_Z_cdl)

    tmp = np.sqrt(float(R_pore) * Z_B.astype(complex))
    electrode_impedance = tmp*(1/np.tanh(np.sqrt(float(R_pore)/Z_B.astype(complex))*lambda_fac))
    total_impedance = electrode_impedance + R_sol
    #print(total_impedance)
    if total_impedance is np.NaN:
        raise ValueError("Z_faraday: {}, Z_B: {}, Z_cdl: {}, Z_electrode: {}"\
                         .format(faraday_impedance,Z_B, Z_cdl, electrode_impedance)
                                                                                     )
    return total_impedance

def simulate_total_impedance_single_intermediate_MkII(
    frequency: ec.FrequencyArray,
    potential: ec.PotentialArray,
    
    # kinetic parameters
    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,

    
    # constant for Faraday impedance
    potential_amplitude: float|ec.Potential,

    # variables of circuit elements
    C_dl: ec.Impedance,
    R_sol: float|ec.Resistance,

    # variable: activesite density
    active_site_density: float,
    
    # variables for transission line model
    R_pore: float|ec.Resistance,
    lambda_fac: float,
):
    """
    改良されたfaraday_catalytic_impedance関数を使ったsimulator
    """
    faraday_impedance = \
        faraday_catalytic_impedance_MkV(
            frequency,
            potential,
            reaction1,
            reaction2,
            potential_amplitude=potential_amplitude,
            activesite_density=active_site_density
        ) # type: ignore

    Z_cdl = 1/(1j * frequency.float_array() * 2*np.pi * C_dl.value)
    
    #Z_B= (1j*Z_F(frequency, pseudo_capacitance, R_HER, R_int))\
    #    /(1j-double_layer_capasitance*frequency*Z_F(frequency, pseudo_capacitance, R_HER, R_int))

    #print(type(Z_cdl), type(faraday_impedance))
    inv_faraday_impedance= 1/faraday_impedance
    inv_Z_cdl = 1/Z_cdl
    #print(type(inv_faraday_impedance), type(inv_Z_cdl))
    
    Z_B = 1/(inv_faraday_impedance + inv_Z_cdl)

    tmp = np.sqrt(float(R_pore) * Z_B.astype(complex))
    electrode_impedance = tmp*(1/np.tanh(np.sqrt(float(R_pore)/Z_B.astype(complex))*lambda_fac))
    total_impedance = electrode_impedance + R_sol
    #print(total_impedance)
    if total_impedance is np.NaN:
        raise ValueError("Z_faraday: {}, Z_B: {}, Z_cdl: {}, Z_electrode: {}"\
                         .format(faraday_impedance,Z_B, Z_cdl, electrode_impedance)
                                                                                     )
    return total_impedance

def simulate_total_impedance_single_intermediate(
    frequency: Union[float, np.ndarray],
    
    # variables for Faraday impedance 
    theta_amp: float|np.ndarray,
    theta_center: float|np.ndarray,
    k_decay:float|np.ndarray,
    k_rds_amp: float|np.ndarray,
    
    # constant for Faraday impedance
    potential_amplitude: float|ec.Potential,

    # variables of circuit elements
    C_dl: float|ec.Impedance,
    R_sol: float|ec.Resistance,

    # variable: activesite density
    active_site_density: float,
    
    # variables for transission line model
    R_pore: float|ec.Resistance,
    lambda_fac: float,

)->np.ndarray[Any,complex]: # type: ignore
    
    """
    古いシミュレータ
    合わない
    """
    
    faraday_impedance = \
        faraday_catalytic_impedance_one_intermediate(
            frequency=frequency,
            theta_amp=theta_amp,
            theta_center=theta_center,
            k_decay=k_decay,
            k_rds_amp=k_rds_amp,
            potential_amplitude=potential_amplitude,
            activesite_density=active_site_density
        ) # type: ignore

    Z_cdl = 1/(1j * frequency * 2*np.pi * C_dl)
    
    #Z_B= (1j*Z_F(frequency, pseudo_capacitance, R_HER, R_int))\
    #    /(1j-double_layer_capasitance*frequency*Z_F(frequency, pseudo_capacitance, R_HER, R_int))

    #print(type(Z_cdl), type(faraday_impedance))
    inv_faraday_impedance = 1/faraday_impedance
    inv_Z_cdl = 1/Z_cdl
    #print(type(inv_faraday_impedance), type(inv_Z_cdl))
    
    Z_B = 1/(inv_faraday_impedance + inv_Z_cdl)

    electrode_impedance = np.sqrt(float(R_pore) * Z_B)*(1/np.tanh(np.sqrt(float(R_pore)/Z_B)*lambda_fac))
    total_impedance = electrode_impedance + R_sol
    #print(total_impedance)
    if total_impedance is np.NaN:
        raise ValueError("Z_faraday: {}, Z_B: {}, Z_cdl: {}, Z_electrode: {}"\
                         .format(faraday_impedance,Z_B, Z_cdl, electrode_impedance)
                                                                                     )
    return total_impedance

    
def simulate_total_impedance_single_intermediate_single_poisoning(
    frequency: Union[float, np.ndarray],
    
    # variables for Faraday impedance 
    theta_amp: float|np.ndarray,
    theta_center: float|np.ndarray,
    k_decay:float|np.ndarray,
    k_rds_amp: float|np.ndarray,

    # variables for poisoning reaction
    theta_poisoned: float|np.ndarray,
    theta_poisoned_amp: float|np.ndarray,
    k_decay_poisoned: float|np.ndarray,

    # constant for Faraday impedance
    potential_amplitude: float|ec.Potential,

    # variables of circuit elements
    C_dl: float|ec.Impedance,
    R_sol: float|ec.Resistance,

    # variable: activesite density
    active_site_density: float,
    
    # variables for transission line model
    R_pore: float|ec.Resistance,
    lambda_fac: float,

)->np.ndarray[Any,complex]: # type: ignore
    
    
    faraday_impedance = \
        faraday_catalytic_impedance_one_intermediate_single_poisoning(
            frequency=frequency,

            theta_amp=theta_amp,
            theta_center=theta_center,
            k_decay=k_decay,
            k_rds_amp=k_rds_amp,
            
            theta_poisoned=theta_poisoned,
            theta_amp_poisoned=theta_poisoned_amp,
            k_decay_poisoning=k_decay_poisoned,
            
            potential_amplitude=potential_amplitude,
            activesite_density=active_site_density
        ) # type: ignore

    Z_cdl = 1/(1j * frequency * 2*np.pi * C_dl)
    
    #print(type(Z_cdl), type(faraday_impedance))
    inv_faraday_impedance = 1/faraday_impedance
    inv_Z_cdl = 1/Z_cdl
    #print(type(inv_faraday_impedance), type(inv_Z_cdl))
    
    Z_B = 1/(inv_faraday_impedance + inv_Z_cdl)

    electrode_impedance = np.sqrt(float(R_pore) * Z_B)*(1/np.tanh(np.sqrt(float(R_pore)/Z_B)*lambda_fac))
    total_impedance = electrode_impedance + R_sol
    #print(total_impedance)
    if total_impedance is np.NaN:
        raise ValueError("Z_faraday: {}, Z_B: {}, Z_cdl: {}, Z_electrode: {}"\
                         .format(faraday_impedance,Z_B, Z_cdl, electrode_impedance)
                                                                                     )
    return total_impedance


def simulate_total_impedance_single_intermediate_boundary_imp(
    frequency: Union[float, np.ndarray],
    
    # variables for Faraday impedance 
    theta_amp: float,
    theta_center: float,
    #k_decay_amp:float,
    #k_decay_center:float,
    k_decay:float,
    k_rds_amp: float,
    
    # constant for Faraday impedance
    potential_amplitude: float,

    # variables of circuit elements
    C_dl: float,
    R_sol: float,

    # variables of boundary impedances
    C_contact: float,
    R_contact: float,
    
    # variables for transission line model
    R_pore: float,
    lambda_fac: float,

)->np.ndarray[complex]:
    
    """
    触媒-担体界面のインピーダンスを入れたモデル。
    大して意味がなかった
    """
    
    faraday_impedance = \
        faraday_catalytic_impedance_one_intermediate(
            frequency=frequency,
            theta_amp=theta_amp,
            theta_center=theta_center,
            #k_decay_amp=k_decay_amp,
            #k_decay_center=k_decay_center,
            k_decay=k_decay,
            k_rds_amp=k_rds_amp,
            potential_amplitude=potential_amplitude
        )

    Z_cdl = 1/(1j * frequency * 2*np.pi * C_dl)

    Z_contact_inv = (1j * frequency * 2*np.pi * C_contact) + 1/R_contact
    Z_contact = 1/Z_contact_inv
    
    #Z_B= (1j*Z_F(frequency, pseudo_capacitance, R_HER, R_int))\
    #    /(1j-double_layer_capasitance*frequency*Z_F(frequency, pseudo_capacitance, R_HER, R_int))
    
    print(Z_cdl, Z_contact, faraday_impedance)
    
    Z_B = 1/(1/faraday_impedance + 1/Z_cdl)+Z_contact

    electrode_impedance = np.sqrt(R_pore * Z_B)*(1/np.tanh(np.sqrt(R_pore/Z_B)*lambda_fac))
    total_impedance = electrode_impedance + R_sol
    #print(total_impedance)
    if total_impedance is np.NaN:
        raise ValueError("Z_faraday: {}, Z_B: {}, Z_cdl: {}, Z_electrode: {}"\
                         .format(faraday_impedance,Z_B, Z_cdl, electrode_impedance)
                                                                                     )
    return total_impedance
