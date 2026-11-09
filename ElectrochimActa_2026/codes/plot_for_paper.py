#%%
import general_kinetics as kin
import common

from typing import Final

import BFC_libs.electrochemistry as ec
import BFC_libs.common as cmn
from BFC_libs.physical_constants import F

import matplotlib.cm as cm
import numpy as np

from plot_lib import MARKER_SIZE, LINE_WIDTH, step1, step2

applied_potential = ec.PotentialArray([0.025, 0.05, 0.075, 0.1])
applied_potential_fine = ec.PotentialArray(np.linspace(-0.125, 0.125, 100))
activesites_num = 1E-9

POTENTIAL_AMP: Final[ec.Potential]= ec.Potential(0.005)
EQ_POTENTIALS: Final[ec.PotentialArray] = ec.PotentialArray([-0.05, 0.0, 0.05])
ls_list = ["-", "--", ":"]

step1_ary:list[kin.ElementaryStep_ET] = [
    kin.ElementaryStep_ET(
        common.ElectronTransferCoefficient(0.5),
        common.ReactionRateCoefficient(8E1),
        ec.Potential(E1),
        kin.Direction.ANODIC
    ) for E1 in EQ_POTENTIALS
]

step2_ary:list[kin.ElementaryStep_ET] = [
    kin.ElementaryStep_ET(
        common.ElectronTransferCoefficient(0.6),
        common.ReactionRateCoefficient(1E0),
        ec.Potential(-1*E1),
        kin.Direction.ANODIC
    )for E1 in EQ_POTENTIALS
]

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

steady_current = calc_steady_current(step1, step2, applied_potential_fine, activesites_num)
steady_current_rough = calc_steady_current(step1, step2, applied_potential, activesites_num)
steady_current_E1_dependence = [
    calc_steady_current(
        _step1,
        _step2,
        applied_potential_fine,
        activesites_num
    ) for _step1, _step2 in zip(step1_ary, step2_ary)
]

fig, ax_tafel = cmn.create_standard_matplt_canvas()

ax_tafel.set_yscale("log")
ax_tafel.set_xlabel(r"$E$ [V vs RHE]")
ax_tafel.set_ylabel(r"$|~i~|$ [mA]")

#ax_tafel.plot(applied_potential_fine, np.abs(steady_current*1000), c = "#707070", zorder = 0)
for i, E in enumerate(applied_potential):
    color = cm.winter(0.2+i/len(applied_potential)*0.7)
    ax_tafel.scatter(
        E, 
        np.abs(steady_current_rough*1000)[i], 
        c = color, 
        zorder = 1,
    )

for i, i_ss in enumerate(steady_current_E1_dependence):
    #color = cm.spring(0.3+i/len(steady_current_E1_dependence)*0.7)
    ax_tafel.plot(
        applied_potential_fine,
        np.abs(i_ss*1000),
        zorder = 0,
        c = "#505050",
        ls = ls_list[i],
        label = r"$E_1$ = {} V".format(EQ_POTENTIALS[i])
    )

ax_tafel.legend()
fig.savefig("../fig/Tafel.png", dpi = 600)
