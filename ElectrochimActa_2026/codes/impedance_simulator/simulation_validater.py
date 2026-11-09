import numpy as np
import pandas as pd
from typing import Callable

import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec

import common
import general_kinetics as kin
import impedance_simulator.kinetics_to_obserbable as kin_to_obs
try:
    import intermediate_impedance as imp_1int
except ModuleNotFoundError:
    import impedance_simulator.intermediate_impedance as imp_1int


def sum_complex_distance(
    obsd_data_array: np.ndarray,
    simulated_data_array: np.ndarray
)->float:
    """
    Return the sum of distance in a complex plane
    """

    if len(obsd_data_array) != len(simulated_data_array):
        raise ValueError(
            "Lengthes of argeud arrays are different.\n"+
            "obs: {}, simd: {}".format(obsd_data_array.shape, simulated_data_array.shape)
        )
    
    re_obs = obsd_data_array.real
    re_sim = simulated_data_array.real

    im_obs = obsd_data_array.imag
    im_sim = simulated_data_array.imag

    sq_dev_re = (re_obs - re_sim)**2
    sq_dev_im = (im_obs - im_sim)**2

    sq_distances = sq_dev_re + sq_dev_im

    return sq_distances.sum()


def calc_distance_array(
    obsd_data_arrays: np.ndarray,
    simd_data_arrays: np.ndarray
)->np.ndarray:
    """
    Shapes of args must be identical
    """

    devs = []
    for obs_array, sim_array in zip(obsd_data_arrays, simd_data_arrays):
        devs.append(sum_complex_distance(obs_array, sim_array))
    
    return np.array(devs)

def make_normd_complex_impedance_and_current(
    impedance: np.ndarray,
    current: np.ndarray,
    weight_for_current: float,
    scaling_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.log10(np.abs(x))
)->np.ndarray:
    """
    Make 2D map of impeadane and i-V curve
    impedances will be given as a map with each potential and frequency.
    pontentail will be given as a vector with each potential.
    the map structure is \n
     -> frequency\n
    [Z11 Z12 ... Z1n I1], |\n
    [Z21 Z22 ... Z2n I2], ▽E\n
    ...\n
    [Zm1 Zm2 ... Zmn Im]\n

    The all values are scaled by scaling_function(),
    which allows only one np.ndarray arg and return np.ndarray,
    without changing the shape.
    Default is log_10(|x|)

    All values are normalized at each column
    """
    #print(current)
    #impedance_re = impedance.real
    #impedance_im = impedance.imag

    #scaled_impedance_re = scaling_function(impedance_re)
    #scaled_impedance_im = scaling_function(impedance_im)
    #scaled_impedance:np.ndarray = scaled_impedance_re+scaled_impedance_im*1j
    scaled_impedance:np.ndarray = scaling_function(impedance)

    scaled_current:np.ndarray = np.array([scaling_function(np.array(current, dtype=float))]).T
    scaled_weighted_current:np.ndarray = scaled_current*weight_for_current

    try:
        unified_aray = np.concatenate([scaled_impedance, scaled_weighted_current], axis=1)
        return unified_aray
    except(np.AxisError):
        raise np.AxisError("Failed to concatrate impedance array and current array.\n"+
                           "Shape of impedance array: {},\n"+
                           "Shape pf current array: {}".format(scaled_impedance.shape, scaled_weighted_current.shape))
    

def __impedance_simulation_passer(
    #independent paramters
    potential_array: ec.PotentialArray,
    frequency_array: ec.FrequencyArray,
    #kinetic parameters
    reaction1,
    reaction2,
    #constants
    potential_amplitude,
    
)->np.ndarray:

    impedance_list_each_potential = []
    for E in potential_array:
        para = kin_to_obs.two_steps_one_intermediate(
            reaction1,
            reaction2,
            E,
            potential_amplitude
        )

        
    impedance_array = np.array(impedance_list_each_potential)

    return impedance_array

def impedance_current_consequence_validator(
    #independent paramters
    potential_array: ec.PotentialArray,
    frequency_array: np.ndarray,
    #kinetic parameters
    E1: ec.Potential,
    alpha1: common.ElectronTransferCoefficient,
    k0_1: common.ReactionRateCoefficient,
    E2: ec.Potential,
    alpha2: common.ElectronTransferCoefficient,
    k0_2: common.ReactionRateCoefficient,
    #non-Faraday impedance elements
    R_sol: ec.Resistance,
    C_dl: ec.Impedance,
    C_dl_square_term: float,
    C_dl_linear_term: float,
    lambda_fac: float,
    R_pore: ec.Resistance,
    #active site density
    active_site_density: float,
    #constants
    potential_amplitude: ec.Potential,
    equilibrium_potential: ec.Potential,
    weight_for_current: float,
    #function_objects
    current_simulator: Callable[[ec.PotentialArray, kin.ElementaryStep_ET, kin.ElementaryStep_ET, float], np.ndarray],
    impedance_simulator: Callable[[ec.PotentialArray, ec.FrequencyArray, kin.ElementaryStep_ET, kin.ElementaryStep_ET,ec.Resistance, ec.ImpedanceArray, ec.Resistance, float,ec.Potential, float], np.ndarray],
    scaling_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.log10(np.abs(x))
)->np.ndarray:
    """
    to be called from fitting function
    Cdl is assumed to be a square function of potential
    """

    #E2 = equilibrium_potential*2 - E1

    #print(E1, E2)

    reaction1 = kin.ElementaryStep_ET(
        alpha=alpha1,
        k0=k0_1,
        E0=E1,
        forward_direction=kin.Direction.CATHODIC
    )

    reaction2 = kin.ElementaryStep_ET(
        alpha=alpha2,
        k0=k0_2,
        E0=E2,
        forward_direction=kin.Direction.CATHODIC
    )

    current_array = current_simulator(
        ec.PotentialArray(potential_array),
        reaction1,
        reaction2,
        active_site_density
    )

    C_dl_calcd = C_dl.value + C_dl_square_term * potential_array.float_array()**2 + C_dl_linear_term * potential_array.float_array()
    C_dl_meshed, _ = np.meshgrid(C_dl_calcd, frequency_array, indexing='ij')

    impedance_array = impedance_simulator(
        potential_array,
        ec.FrequencyArray(frequency_array),
        reaction1,
        reaction2,
        R_sol,
        ec.ImpedanceArray(C_dl_meshed),
        R_pore,
        lambda_fac,
        potential_amplitude,
        active_site_density
    )


    joined_array = make_normd_complex_impedance_and_current(
        impedance_array,
        current_array,
        weight_for_current,
        scaling_function
    )

    return joined_array

def impedance_current_consequence_validator_constant_Cdl(
    #independent paramters
    potential_array: np.ndarray,
    frequency_array: np.ndarray,
    #kinetic parameters
    E1: ec.Potential,
    alpha1: common.ElectronTransferCoefficient,
    k0_1: common.ReactionRateCoefficient,
    E2: ec.Potential,
    alpha2: common.ElectronTransferCoefficient,
    k0_2: common.ReactionRateCoefficient,
    #non-Faraday impedance elements
    R_sol: ec.Resistance,
    C_dl: ec.Impedance,
    lambda_fac: float,
    R_pore: ec.Resistance,
    #active site density
    active_site_density: float,
    #constants
    potential_amplitude: ec.Potential,
    equilibrium_potential: ec.Potential,
    weight_for_current: float,
    #function_objects
    current_simulator: Callable[[ec.PotentialArray, kin.ElementaryStep_ET, kin.ElementaryStep_ET, float], np.ndarray],
    impedance_simulator: Callable[[ec.PotentialArray, ec.FrequencyArray, kin.ElementaryStep_ET, kin.ElementaryStep_ET,ec.Resistance, ec.Impedance,ec.Resistance, float,ec.Potential, float], np.ndarray],
    scaling_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.log10(np.abs(x))
)->np.ndarray:
    """
    to be called from fitting function
    """

    #E2 = equilibrium_potential*2 - E1

    #print(E1, E2)

    reaction1 = kin.ElementaryStep_ET(
        alpha=alpha1,
        k0=k0_1,
        E0=E1,
        forward_direction=kin.Direction.CATHODIC
    )

    reaction2 = kin.ElementaryStep_ET(
        alpha=alpha2,
        k0=k0_2,
        E0=E2,
        forward_direction=kin.Direction.CATHODIC
    )

    current_array = current_simulator(
        ec.PotentialArray(potential_array),
        reaction1,
        reaction2,
        active_site_density
    )

    #impedance_list = []
    # for e in potential_array:
    #     impedance_list.append(
    #         impedance_simulator(
    #             e,
    #             ec.FrequencyArray(frequency_array),
    #             reaction1,
    #             reaction2,
    #             R_sol,
    #             C_dl,
    #             R_pore,
    #             lambda_fac,
    #             potential_amplitude,
    #             active_site_density
    #         )
    #     )
    #impedance_array = np.array(impedance_list)
    impedance_array = impedance_simulator(
        potential_array,
        ec.FrequencyArray(frequency_array),
        reaction1,
        reaction2,
        R_sol,
        C_dl,
        R_pore,
        lambda_fac,
        potential_amplitude,
        active_site_density
    )


    joined_array = make_normd_complex_impedance_and_current(
        impedance_array,
        current_array,
        weight_for_current,
        scaling_function
    )

    return joined_array


def impedance_current_consequence_validator_poisoning(
    #independent paramters
    potential_array: np.ndarray,
    frequency_array: np.ndarray,
    #kinetic parameters
    E1: ec.Potential,
    alpha1: common.ElectronTransferCoefficient,
    k0_1: common.ReactionRateCoefficient,
    E2: ec.Potential,
    alpha2: common.ElectronTransferCoefficient,
    k0_2: common.ReactionRateCoefficient,
    #poisoning reation
    E_poison: ec.Potential,
    alpha_poison: common.ElectronTransferCoefficient,
    k0_poison: common.ReactionRateCoefficient,
    #non-Faraday impedance elements
    R_sol: ec.Resistance,
    C_dl: ec.Impedance,
    lambda_fac: float,
    R_pore: ec.Resistance,
    #active site density
    active_site_density: float,
    #constants
    potential_amplitude: ec.Potential,
    equilibrium_potential: ec.Potential,
    weight_for_current: float,
    #function_objects
    current_simulator: Callable[[ec.PotentialArray, kin.ElementaryStep_ET, kin.ElementaryStep_ET, kin.ElementaryStep_ET, float], np.ndarray],
    impedance_simulator: Callable[[ec.Potential, ec.FrequencyArray, kin.ElementaryStep_ET, kin.ElementaryStep_ET, kin.ElementaryStep_ET,ec.Resistance, ec.Impedance,ec.Resistance, float,ec.Potential, float], np.ndarray],
    scaling_function: Callable[[np.ndarray], np.ndarray] = lambda x: np.log10(np.abs(x))
)->np.ndarray:
    """
    to be called from fitting function
    """

    #E2 = equilibrium_potential*2 - E1

    #print(E1, E2)

    reaction1 = kin.ElementaryStep_ET(
        alpha=alpha1,
        k0=k0_1,
        E0=E1,
        forward_direction=kin.Direction.CATHODIC
    )

    reaction2 = kin.ElementaryStep_ET(
        alpha=alpha2,
        k0=k0_2,
        E0=E2,
        forward_direction=kin.Direction.CATHODIC
    )

    poisoning_reaction = kin.ElementaryStep_ET(
        alpha=alpha_poison,
        k0=k0_poison,
        E0=E_poison,
        forward_direction=kin.Direction.ANODIC
    )

    current_array = current_simulator(
        ec.PotentialArray(potential_array),
        reaction1,
        reaction2,
        poisoning_reaction,
        active_site_density
    )

    impedance_list = []
    for e in potential_array:
        impedance_list.append(
            impedance_simulator(
                e,
                ec.FrequencyArray(frequency_array),
                reaction1,
                reaction2,
                poisoning_reaction,
                R_sol,
                C_dl,
                R_pore,
                lambda_fac,
                potential_amplitude,
                active_site_density
            )
        )
    impedance_array = np.array(impedance_list)


    joined_array = make_normd_complex_impedance_and_current(
        impedance_array,
        current_array,
        weight_for_current,
        scaling_function
    )

    return joined_array

def impedance_simulation_passer(
    center_potential: ec.Potential|ec.PotentialArray,
    frequency_array: ec.FrequencyArray,
    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,
    R_sol: ec.Resistance,
    C_dl: ec.Impedance,
    R_pore: ec.Resistance,
    lambda_fac: float,
    potential_amplitude: ec.Potential,
    active_site_density: float,
):
    
    obserbable_parameters = \
        kin_to_obs.two_steps_one_intermediate(
            reaction1,
            reaction2,
            center_potential,
            potential_amplitude
        )
    
    imp=imp_1int.simulate_total_impedance_single_intermediate_MkII(
        frequency_array,
        center_potential,
        reaction1,
        reaction2,
        R_sol=R_sol,
        R_pore=R_pore,
        C_dl=C_dl,
        lambda_fac=lambda_fac,
        potential_amplitude=potential_amplitude,
        active_site_density=active_site_density,
    )

    return imp

def impedance_simulation_passer_poisoning(
    center_potential: ec.Potential|ec.PotentialArray,
    frequency_array: ec.FrequencyArray,
    reaction1: kin.ElementaryStep_ET,
    reaction2: kin.ElementaryStep_ET,
    poisoning_step: kin.ElementaryStep_ET,
    R_sol: ec.Resistance,
    C_dl: ec.Impedance,
    R_pore: ec.Resistance,
    lambda_fac: float,
    potential_amplitude: ec.Potential,
    active_site_density: float,
):
    
    obserbable_parameters = \
        kin_to_obs.two_steps_one_intermediate_one_anodic_poisoning(
            reaction1,
            reaction2,
            poisoning_step,
            center_potential,
            potential_amplitude
        )
    
    imp=imp_1int.simulate_total_impedance_single_intermediate_single_poisoning(
        frequency_array.float_array(),
        R_sol=R_sol,
        R_pore=R_pore,
        C_dl=C_dl,
        lambda_fac=lambda_fac,
        potential_amplitude=potential_amplitude,
        active_site_density=active_site_density,
        **obserbable_parameters,
    )

    return imp