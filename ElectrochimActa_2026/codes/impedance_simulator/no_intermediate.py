import numpy as np
import cmath
import scipy.integrate as integrate
#import quadpy 

from typing import Union, Final

from BFC_libs.physical_constants import F
import BFC_libs.common as cmn

import impedance_simulator as sim

def simulate_total_impedance_no_intermediate(
    frequency: Union[float, np.ndarray],
    
    # variables for Faraday impedance 
    R_ct: float,
    
    # constant for Faraday impedance
    potential_amplitude: float,

    # variables of circuit elements
    C_dl: float,
    R_sol: float,
    
    # variables for transission line model
    R_pore: float,
    lambda_fac: float,

)->np.ndarray[complex]:
    
    Z_cdl = 1/(1j * frequency * 2*np.pi * C_dl)
    
    #Z_B= (1j*Z_F(frequency, pseudo_capacitance, R_HER, R_int))\
    #    /(1j-double_layer_capasitance*frequency*Z_F(frequency, pseudo_capacitance, R_HER, R_int))
    Z_B = 1/(1/R_ct + 1/Z_cdl)

    electrode_impedance = np.sqrt(R_pore * Z_B)*(1/np.tanh(np.sqrt(R_pore/Z_B)*lambda_fac))
    total_impedance = electrode_impedance + R_sol
    #print(total_impedance)
    if total_impedance is np.NaN:
        raise ValueError("Z_B: {}, Z_cdl: {}, Z_electrode: {}"\
                         .format(Z_B, Z_cdl, electrode_impedance)
                                                                                     )
    return total_impedance