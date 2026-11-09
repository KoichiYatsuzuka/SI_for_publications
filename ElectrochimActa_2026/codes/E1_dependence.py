#%%
import impedance_calculater as ic
import BFC_libs.electrochemistry as ec
import BFC_libs.quantity_array as qa

import matplotlib.cm as cm
import numpy as np

from typing import Final
import numpy.typing as nptyping

import general_kinetics as kin
import common

from plot_lib import SOLUTION_RESISTANCE, DOUBLE_LAYER_CAPACITANCE, ACTIVESITES_NUM
from plot_lib import MARKER_SIZE, LINE_WIDTH
#from plot_lib import step1, step2
import plot_lib as plot_lib

APPLIED_POTENTIAL = ec.Potential(0.05)

POTENTIL_AMPLITUDE: Final[float] = 0.005
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


path_list: list[str] = []

for eq_potential in EQ_POTENTIALS:
    path_list.append(common.TIME_EVOLUTION_OUTPUT_DIR+common.TIME_EVOLUTION_FILE_FORMAT.format(E1= eq_potential, potential_center = APPLIED_POTENTIAL, potential_amp = POTENTIL_AMPLITUDE*1000))
time_evolved_EIS = plot_lib.load_time_evolved_EIS(path_list)
time_evolved_impedance = plot_lib.list_EIS_to_2d_array(time_evolved_EIS)

omega = ec.FrequencyArray(time_evolved_EIS[0].frequency*2*np.pi)

meshed_omega_tmp, meshed_E_tmp = np.meshgrid(omega.float_array(), EQ_POTENTIALS.float_array())
meshed_omega = qa.QArray[ec.FrequencyArray](meshed_omega_tmp)
meshed_E = qa.QArray[ec.PotentialArray](meshed_E_tmp)

analytical_imedance:np.ndarray = np.array([
    ic.calculate_analytical_impedance(
        step1,
        step2,
        ec.PotentialArray([APPLIED_POTENTIAL]),
        omega,
        ACTIVESITES_NUM
    )[0] for step1, step2 in zip(step1_ary, step2_ary)
])

full_analytical_impedance = plot_lib.add_non_Faraday_impedance(
    SOLUTION_RESISTANCE,
    DOUBLE_LAYER_CAPACITANCE,
    meshed_omega,
    analytical_imedance
)

full_time_evolved_impedance = plot_lib.add_non_Faraday_impedance(
    SOLUTION_RESISTANCE,
    DOUBLE_LAYER_CAPACITANCE,
    meshed_omega,
    time_evolved_impedance    
)

fig, ax_Nyquist, ax_Bode_R, ax_Bode_ph = plot_lib.common_canvas_for_impedance_plot()

plot_num = len(full_analytical_impedance)

for i, imp in enumerate(zip(full_analytical_impedance, full_time_evolved_impedance)):
    
    analytical:nptyping.NDArray[np.complex128] = imp[0]
    time_evolved:nptyping.NDArray[np.complex128] = imp[1]

    color = cm.winter(0.2+i/plot_num*0.7)
    z = plot_num - i

    ax_Nyquist.scatter(
        np.real(analytical[::2]),
        -np.imag(analytical[::2]),
        s = MARKER_SIZE, 
        facecolors = "none", 
        edgecolors = color, 
        zorder = z,
        label = r"$E_1$ = {} V".format(EQ_POTENTIALS[i])
    )
    ax_Bode_R.scatter(
        omega.float_array()[::2],
        np.abs(analytical)[::2],
        s = MARKER_SIZE, 
        facecolors = "none", 
        edgecolors = color, 
        zorder = z
    )
    ax_Bode_ph.scatter(
        omega.float_array()[::2],
        np.angle(analytical)[::2]/np.pi*180,
        s = MARKER_SIZE, 
        facecolors = "none", 
        edgecolors = color, 
        zorder = z,
        #label = r"$E_1$ = {} V".format(EQ_POTENTIALS[i])
    )

    ax_Nyquist.plot(
        np.real(time_evolved),
        -np.imag(time_evolved),
        zorder = z,
        c = color,
        ls = ls_list[i],
        lw = LINE_WIDTH,
        #label = r"$E_1$ = {} V".format(EQ_POTENTIALS[i])
    )
    ax_Bode_R.plot(
        omega.float_array(),
        np.abs(time_evolved),
        zorder = z,
        c = color,
        ls = ls_list[i],
        lw = LINE_WIDTH
    )
    ax_Bode_ph.plot(
        omega.float_array(),
        np.angle(time_evolved)/np.pi*180,
        zorder = z,
        c = color,
        ls = ls_list[i],
        lw = LINE_WIDTH,
        label = r"$E_1$ = {} V".format(EQ_POTENTIALS[i])
    )
ax_Nyquist.text(-0.2, 1.0, "(A)", transform = ax_Nyquist.transAxes)
ax_Nyquist.set_xlim(0, 50)
ax_Nyquist.set_ylim(0, 50)
ax_Nyquist.legend()
ax_Bode_R.text(-0.2, 1.0, "(B)", transform = ax_Bode_R.transAxes)
ax_Bode_ph.legend()

fig.savefig(common.FIGURE_EXPORT_DIR+"E1_dependence.png", dpi = 600)
