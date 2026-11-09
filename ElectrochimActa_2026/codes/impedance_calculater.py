from dataclasses import dataclass
from typing import TypeVar, Generic
import numpy as np
import numpy.typing as nptype
import common
import general_kinetics as kin

import BFC_libs.quantity_array as qa
import BFC_libs.electrochemistry as ec
from BFC_libs.physical_constants import F, R, T

#float_or_ndarray = TypeVar("float_or_ndarray", float,np.ndarray)

class IntParams:
    kbar: np.ndarray
    A: np.ndarray
    B: np.ndarray
    Phi: np.ndarray
    theta_bar: np.ndarray
    delta_theta: np.ndarray
    # P: np.ndarray
    # R: np.ndarray
    # Q: np.ndarray

    def __init__(self,
                 step1: kin.ElementaryStep_ET, 
                 step2: kin.ElementaryStep_ET,
                 applied_potential: ec.PotentialArray,
                 angular_frequency: ec.FrequencyArray,
                 ):
        
        E_temp = applied_potential.float_array()
        omega_temp = angular_frequency.float_array()

        #二次元メッシュ化。行列の第一次元が電位、第二次元が各周波数
        omega_meshed, E_meshed = np.meshgrid(omega_temp, E_temp)

        E_meshed_qarray = qa.QArray[ec.PotentialArray](E_meshed)

        # 反応速度
        k1bar = step1.rate_fw(E_meshed_qarray)
        k1rbar = step1.rate_bw(E_meshed_qarray)

        k2bar = step2.rate_fw(E_meshed_qarray)
        k2rbar = step2.rate_bw(E_meshed_qarray)

        # 反応速度の微分（prime）
        k1p = step1.accelaration_fw(E_meshed_qarray)
        k1rp = step1.accelaration_bw(E_meshed_qarray)

        k2p = step2.accelaration_fw(E_meshed_qarray)
        k2rp = step2.accelaration_bw(E_meshed_qarray)


        self.kbar = k1bar + k1rbar + k2bar + k2rbar
        
        a = k1p + k2rp
        b = k1rp + k2p

        # 定義域: -pi/2 - pi/2
        phi = np.arctan(-omega_meshed/self.kbar)


        self.theta_bar = (k1bar+k2rbar)/self.kbar
        self.delta_theta = -((a+b)*self.theta_bar - a)/(np.sqrt(omega_meshed**2+self.kbar**2))

        kbar_neg = k1bar + k1rbar - k2bar - k2rbar
        a_neg = k1p - k2rp
        b_neg = k1rp - k2p

        self.A = a_neg - (a_neg + b_neg)*self.theta_bar - kbar_neg*self.delta_theta * np.cos(phi)

        self.B = -kbar_neg*self.delta_theta*np.sin(phi)

        self.Phi = np.arctan(self.B/self.A)

        # self.P = a_neg - (a_neg + b_neg)*delta_theta

        # self.Q = kbar_neg*((a+b)*self.theta_bar-a)

        # self.R = self.P * self.kbar - self.Q




def get_A(
    step1: kin.ElementaryStep_ET, 
    step2: kin.ElementaryStep_ET,
    applied_potential: ec.PotentialArray,
    angular_frequency: ec.FrequencyArray,
)->nptype.NDArray[np.floating]:

    intermediate_params = IntParams(
            step1,
            step2,
            applied_potential,
            angular_frequency
        )
    return intermediate_params.A

def get_B(
    step1: kin.ElementaryStep_ET, 
    step2: kin.ElementaryStep_ET,
    applied_potential: ec.PotentialArray,
    angular_frequency: ec.FrequencyArray,
)->nptype.NDArray[np.floating]:

    intermediate_params = IntParams(
            step1,
            step2,
            applied_potential,
            angular_frequency
        )
    return intermediate_params.B


def calculate_analytical_impedance(
    step1: kin.ElementaryStep_ET, 
    step2: kin.ElementaryStep_ET,
    applied_potential: ec.PotentialArray,
    angular_frequency: ec.FrequencyArray,
    activesites_num: float
)->nptype.NDArray[np.floating]:
    
    intermediate_params = IntParams(
        step1,
        step2,
        applied_potential,
        angular_frequency
    )

    A = intermediate_params.A
    B = intermediate_params.B

    impedance = (A - 1j*B)/(activesites_num*F*(A**2+B**2))

    return impedance




        



