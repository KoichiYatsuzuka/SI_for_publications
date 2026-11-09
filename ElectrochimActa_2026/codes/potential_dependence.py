#%%
import impedance_calculater as ic
import BFC_libs.electrochemistry as ec

import matplotlib.cm as cm
import numpy as np

from typing import Final
import numpy.typing as nptyping

from common import TIME_EVOLUTION_OUTPUT_DIR, TIME_EVOLUTION_FILE_FORMAT, FIGURE_EXPORT_DIR
from plot_lib import SOLUTION_RESISTANCE, DOUBLE_LAYER_CAPACITANCE, ACTIVESITES_NUM
from plot_lib import MARKER_SIZE, LINE_WIDTH
from plot_lib import step1, step2
import plot_lib as plot_lib

applied_potential = ec.PotentialArray([0.025, 0.05, 0.075, 0.1])
POTENTIL_AMPLITUDE: Final[float] = 0.005

path_list: list[str] = []

for potential in applied_potential:
    path_list.append(TIME_EVOLUTION_OUTPUT_DIR+TIME_EVOLUTION_FILE_FORMAT.format(E1 = -0.05 ,potential_center = potential.value, potential_amp = POTENTIL_AMPLITUDE*1000))
time_evolved_EIS = plot_lib.load_time_evolved_EIS(path_list)
time_evolved_impedance = plot_lib.list_EIS_to_2d_array(time_evolved_EIS)

omega = ec.FrequencyArray(time_evolved_EIS[0].frequency*2*np.pi)
meshed_omega, meshed_E = np.meshgrid(omega, applied_potential)

analytical_imedance = ic.calculate_analytical_impedance(
    step1,
    step2,
    applied_potential,
    omega,
    ACTIVESITES_NUM
)

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
        np.real(analytical),
        -np.imag(analytical),
        s = MARKER_SIZE, 
        facecolors = "none", 
        edgecolors = color, 
        zorder = z,
    )
    ax_Bode_R.scatter(
        omega.float_array(),
        np.abs(analytical),
        s = MARKER_SIZE, 
        facecolors = "none", 
        edgecolors = color, 
        zorder = z,
    )
    ax_Bode_ph.scatter(
        omega.float_array(),
        np.angle(analytical)/np.pi*180,
        s = MARKER_SIZE, 
        facecolors = "none", 
        edgecolors = color, 
        zorder = z,
    )

    ax_Nyquist.plot(
        np.real(time_evolved),
        -np.imag(time_evolved),
        zorder = z,
        c = color,
        lw = LINE_WIDTH
    )
    ax_Bode_R.plot(
        omega.float_array(),
        np.abs(time_evolved),
        zorder = z,
        c = color,
        lw = LINE_WIDTH
    )
    ax_Bode_ph.plot(
        omega.float_array(),
        np.angle(time_evolved)/np.pi*180,
        zorder = z,
        c = color,
        lw = LINE_WIDTH
    )
ax_Nyquist.text(-0.2, 1.0, "(A)", transform = ax_Nyquist.transAxes)
ax_Nyquist.set_xlim(0, 100)
ax_Nyquist.set_ylim(0, 100)
ax_Nyquist.legend()
ax_Bode_R.text(-0.2, 1.0, "(B)", transform = ax_Bode_R.transAxes)
ax_Bode_ph.legend()

fig.savefig(FIGURE_EXPORT_DIR+"potential_dependence.png", dpi = 600)

# %%
