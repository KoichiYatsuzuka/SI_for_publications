#%%
import general_kinetics as kin
import non_analytical_impedance as nai
import common

from typing import Final
#from functools import partial

import BFC_libs.electrochemistry as ec
#import BFC_libs.common as cmn
from BFC_libs.physical_constants import F, R, T

import numpy as np
#import numpy.fft as fft
#import scipy.integrate as integ


omega:ec.FrequencyArray = ec.FrequencyArray(np.logspace(0, 6, 40)*np.pi*2)
applied_potential = ec.PotentialArray([0.025, 0.05, 0.075, 0.1])
applied_potential_fine = ec.PotentialArray(np.linspace(-0.125, 0.125, 100))
activesites_num = 1E-9

POTENTIAL_AMP: Final[ec.Potential]= ec.Potential(0.005)
EQ_POTENTIAL: Final[ec.Potential] = ec.Potential(0.1)


def calc_steady_current(
    stepA: kin.ElementaryStep_ET,
    stepB: kin.ElementaryStep_ET,
    potential: ec.PotentialArray,
    active_site_num: float
)->np.ndarray:
    k1 = stepA.rate_fw(potential)
    k1r = stepA.rate_bw(potential)
    k2 = stepB.rate_fw(potential)
    k2r = stepB.rate_bw(potential)

    theta = (k1+k2r)/(k1+k1r+k2+k2r)
    vacuncy = 1-theta

    return F*active_site_num*(vacuncy*k1 + theta*k2 - theta*k1r - vacuncy*k2r)


step1 = kin.ElementaryStep_ET(
    common.ElectronTransferCoefficient(0.5),
    common.ReactionRateCoefficient(8E1),
    ec.Potential(EQ_POTENTIAL),
    kin.Direction.ANODIC
)

step2 = kin.ElementaryStep_ET(
    common.ElectronTransferCoefficient(0.6),
    common.ReactionRateCoefficient(1E0),
    ec.Potential(-1*EQ_POTENTIAL),
    kin.Direction.ANODIC
)


steady_current = calc_steady_current(step1, step2, applied_potential_fine, activesites_num)
steady_current_rough = calc_steady_current(step1, step2, applied_potential, activesites_num)


# 微分方程式の時間発展計算
time_evolved_impedance = nai.impedance_via_sequential_current_simulation(
    applied_potential,
    omega/2/np.pi,
    step1,
    step2,
    POTENTIAL_AMP,
    activesites_num,
    3600,
    10
)



for te_imp in time_evolved_impedance:
    
    te_imp.to_csv(
        common.TIME_EVOLUTION_OUTPUT_DIR+\
        common.TIME_EVOLUTION_FILE_FORMAT.format(E1 = EQ_POTENTIAL,potential_center = te_imp.applied_potential, potential_amp = POTENTIAL_AMP*1000))
