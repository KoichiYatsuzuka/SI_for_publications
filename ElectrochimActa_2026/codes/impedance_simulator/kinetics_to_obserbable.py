import numpy as np

import BFC_libs.electrochemistry as ec

from general_kinetics import ElementaryStep_ET
import common

def two_steps_one_intermediate(
    first_step: ElementaryStep_ET,
    second_step: ElementaryStep_ET,
    applied_potential: ec.Potential|ec.PotentialArray,
    potential_amplitude: ec.Potential = ec.Potential(0.01)
)->dict[str, float]|dict[str, np.ndarray]:
    """
    """
    #print(first_step)
    #print(applied_potential)

    if applied_potential is np.NaN:
        raise ValueError("np.Nan is sumstituted")
    
    if isinstance(applied_potential, ec.PotentialArray):
        cast_func = np.array
        #_applied_potential = applied_potential.float_array()
    elif isinstance(applied_potential, ec.Potential):
        cast_func = float
        #_applied_potential = applied_potential.value
    elif isinstance(applied_potential, np.ndarray):
        cast_func = np.array
        #_applied_potential = applied_potential
    elif isinstance(applied_potential, np.float64):
        cast_func = np.float64
        #_applied_potential = applied_potential
    else:
        raise TypeError("applied_potential must be ec.Potential or ec.PotentialArray")
    
    """if isinstance(potential_amplitude, ec.Potential):
        potential_amplitude = potential_amplitude.value"""
    


    k1 = cast_func(first_step.rate_fw(applied_potential=applied_potential))
    k2 = cast_func(second_step.rate_fw(applied_potential=applied_potential))
    k1r = cast_func(first_step.rate_bw(applied_potential=applied_potential))
    k2r = cast_func(second_step.rate_bw(applied_potential=applied_potential))

    #Eの変位がforward方向に大きいときの値
    k1_high= cast_func(first_step.rate_fw(applied_potential+potential_amplitude))
    k2_high = cast_func(second_step.rate_fw(applied_potential+potential_amplitude))
    k1r_high = cast_func(first_step.rate_bw(applied_potential+potential_amplitude))
    k2r_high = cast_func(second_step.rate_bw(applied_potential+potential_amplitude))

    #Eの変位がbackward方向に大きいときの値
    k1_low= cast_func(first_step.rate_fw(applied_potential-potential_amplitude))
    k2_low = cast_func(second_step.rate_fw(applied_potential-potential_amplitude))
    k1r_low = cast_func(first_step.rate_bw(applied_potential-potential_amplitude))
    k2r_low = cast_func(second_step.rate_bw(applied_potential-potential_amplitude))
    #print(k1, k2, k1r, k2r)
    sum_k = k1+k2+k1r+k2r
    sum_k_high = k1_high + k2_high + k1r_high + k2r_high
    sum_k_low = k1_low + k2_low + k1r_low + k2r_low
    
    ret = {}

    ret[common.KEY_THETA_CENTER] = (k1+k2r)/sum_k
    ret[common.KEY_THETA_AMP] = ((k1_high+k2r_high)/sum_k_high - (k1_low+k2r_low)/sum_k_low)/2

    ret[common.KEY_K_DECAY] = sum_k
    ret[common.KEY_K_RDS_AMP] = np.abs((k2_high-k1r_high) - (k2_low-k1r_low))

    #print(k2_high, k2_low, k1r_high, k1r_low)

    return ret


def two_steps_one_intermediate_one_anodic_poisoning(
    first_step: ElementaryStep_ET,
    second_step: ElementaryStep_ET,
    poisoning_step: ElementaryStep_ET,
    applied_potential: ec.Potential|ec.PotentialArray,
    potential_amplitude: ec.Potential = ec.Potential(0.01)
)->dict[str, float]|dict[str, np.ndarray]:
    """
    """
    #print(first_step)
    #print(applied_potential)

    if applied_potential is np.NaN:
        raise ValueError("np.Nan is sumstituted")
    
    if isinstance(applied_potential, ec.PotentialArray):
        cast_func = np.array
        #_applied_potential = applied_potential.float_array()
    elif isinstance(applied_potential, ec.Potential):
        cast_func = float
        #_applied_potential = applied_potential.value
    elif isinstance(applied_potential, np.ndarray):
        cast_func = np.array
        #_applied_potential = applied_potential
    elif isinstance(applied_potential, np.float64):
        cast_func = np.float64
        #_applied_potential = applied_potential
    else:
        raise TypeError("applied_potential must be ec.Potential or ec.PotentialArray")
    
    """if isinstance(potential_amplitude, ec.Potential):
        potential_amplitude = potential_amplitude.value"""
    


    k1 = cast_func(first_step.rate_fw(applied_potential=applied_potential))
    k2 = cast_func(second_step.rate_fw(applied_potential=applied_potential))
    k1r = cast_func(first_step.rate_bw(applied_potential=applied_potential))
    k2r = cast_func(second_step.rate_bw(applied_potential=applied_potential))

    k_poison = cast_func(poisoning_step.rate_bw(applied_potential=applied_potential))
    k_poison_r = cast_func(poisoning_step.rate_fw(applied_potential=applied_potential))

    #Eの変位がforward方向に大きいときの値
    k1_high= cast_func(first_step.rate_fw(applied_potential+potential_amplitude))
    k2_high = cast_func(second_step.rate_fw(applied_potential+potential_amplitude))
    k1r_high = cast_func(first_step.rate_bw(applied_potential+potential_amplitude))
    k2r_high = cast_func(second_step.rate_bw(applied_potential+potential_amplitude))
    k_poison_high = cast_func(poisoning_step.rate_bw(applied_potential+potential_amplitude))
    k_poison_r_high = cast_func(poisoning_step.rate_fw(applied_potential+potential_amplitude))

    #Eの変位がbackward方向に大きいときの値
    k1_low= cast_func(first_step.rate_fw(applied_potential-potential_amplitude))
    k2_low = cast_func(second_step.rate_fw(applied_potential-potential_amplitude))
    k1r_low = cast_func(first_step.rate_bw(applied_potential-potential_amplitude))
    k2r_low = cast_func(second_step.rate_bw(applied_potential-potential_amplitude))
    k_poison_low = cast_func(poisoning_step.rate_bw(applied_potential-potential_amplitude))
    k_poison_r_low = cast_func(poisoning_step.rate_fw(applied_potential-potential_amplitude))
    #print(k1, k2, k1r, k2r)
    sum_k = k1+k2+k1r+k2r
    sum_k_high = k1_high + k2_high + k1r_high + k2r_high
    sum_k_low = k1_low + k2_low + k1r_low + k2r_low
    
    ret = {}

    # print(k_poison)
    # print(k_poison_r)


    steady_theta_A = 1/(1 + (k1+k2r)/(k1r+k2) + k_poison/k_poison_r)
    steady_theta_B = (k1+k2r)/(k1r+k2) * steady_theta_A
    steady_theta_poisoned = k_poison/k_poison_r * steady_theta_A

    steady_theta_A_high = 1/(1 + (k1_high+k2r_high)/(k1r_high+k2_high) + k_poison_high/k_poison_r_high)
    steady_theta_B_high = (k1_high+k2r_high)/(k1r_high+k2_high) * steady_theta_A_high
    steady_theta_poisoned_high = k_poison_high/k_poison_r_high * steady_theta_A_high

    steady_theta_A_low = 1/(1 + (k1_low+k2r_low)/(k1r_low+k2_low) + k_poison_low/k_poison_r_low)
    steady_theta_B_low = (k1_low+k2r_low)/(k1r_low+k2_low) * steady_theta_A_low
    steady_theta_poisoned_low = k_poison_low/k_poison_r_low * steady_theta_A_low

    ret[common.KEY_THETA_CENTER] = steady_theta_A
    ret[common.KEY_THETA_AMP] = np.abs(steady_theta_A_high - steady_theta_A_low)

    ret[common.KEY_K_DECAY] = sum_k
    ret[common.KEY_K_RDS_AMP] = np.abs((k2_high-k1r_high) - (k2_low-k1r_low))

    ret[common.KEY_THETA_POISONED] = steady_theta_poisoned
    ret[common.KEY_THETA_POISONED_AMP] = np.abs(steady_theta_poisoned_high - steady_theta_poisoned_low)
    ret[common.KEY_K_DECAY_POISONED] = k_poison_r #近似あり（d theta_OH/dt = k*theta_blank -kr*theta_OH, thetaは定常状態の値）

    #print(k2_high, k2_low, k1r_high, k1r_low)

    return ret
