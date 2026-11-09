#%%
import impedance_calculater as ic
import BFC_libs.electrochemistry as ec
import BFC_libs.quantity_array as qa

import matplotlib.cm as cm
import numpy as np

from common import TIME_EVOLUTION_OUTPUT_DIR, TIME_EVOLUTION_FILE_FORMAT, FIGURE_EXPORT_DIR
from plot_lib import SOLUTION_RESISTANCE, DOUBLE_LAYER_CAPACITANCE, ACTIVESITES_NUM
from plot_lib import MARKER_SIZE, LINE_WIDTH
from plot_lib import step1, step2
import plot_lib as plot_lib 


applied_potential = ec.Potential(0.075)
potential_amp = ec.PotentialArray([0.005, 0.01, 0.025])

path_list:list[str] = []
for amp in potential_amp:
    path_list.append(TIME_EVOLUTION_OUTPUT_DIR+TIME_EVOLUTION_FILE_FORMAT.format(
        E1 = -0.05,
        potential_center = applied_potential.value, 
        potential_amp = amp.value*1000)
        )
time_evolved_EIS = plot_lib.load_time_evolved_EIS(path_list)

omega:ec.FrequencyArray = ec.FrequencyArray(time_evolved_EIS[0].frequency*2*np.pi)
meshed_omega, meshed_E = np.meshgrid(omega, ec.PotentialArray([applied_potential]))

impedance = ic.calculate_analytical_impedance(
    step1,
    step2,
    qa.QArray[ec.PotentialArray](meshed_E),
    angular_frequency=qa.QArray[ec.FrequencyArray](meshed_omega),
    activesites_num=ACTIVESITES_NUM
)

full_impedance = plot_lib.add_non_Faraday_impedance(
    SOLUTION_RESISTANCE,
    DOUBLE_LAYER_CAPACITANCE,
    meshed_omega,
    impedance
)

fig, ax_Nyquist, ax_Bode_R, ax_Bode_ph = plot_lib.common_canvas_for_impedance_plot()

ax_Nyquist.plot(
    full_impedance[0].real, 
    -full_impedance[0].imag, 
    c="#707070", 
    ls="-", 
    zorder = -10, 
    lw= LINE_WIDTH
)
ax_Bode_R.plot(
    omega,
    np.abs(full_impedance[0]),
    c="#707070", 
    ls="-", 
    zorder = -10, 
    lw= LINE_WIDTH
)
ax_Bode_ph.plot(
    omega,
    np.angle(full_impedance[0])/np.pi*180,
    c="#707070", 
    ls="-", 
    zorder = -10, 
    lw= LINE_WIDTH
)
plot_num = len(time_evolved_EIS)
for i, eis in enumerate(time_evolved_EIS):
    full_time_evolved_impedance = plot_lib.add_non_Faraday_impedance(
        SOLUTION_RESISTANCE,
        DOUBLE_LAYER_CAPACITANCE,
        omega,
        eis.impedance.complex_array()
    )
    color = cm.winter(0.2+i/plot_num*0.7)
    label_str = r"$\Delta E = $" + "{} mV".format(potential_amp[i]*1000)

    ax_Nyquist.scatter(
        full_time_evolved_impedance.real, 
        -full_time_evolved_impedance.imag, 
        s = MARKER_SIZE, 
        facecolors = "none", 
        edgecolors = color, 
        zorder = plot_num - i,
        label = label_str
        )
    ax_Bode_R.scatter(
        omega,
        np.abs(full_time_evolved_impedance),
        s = MARKER_SIZE, 
        facecolors = "none", 
        edgecolors = color, 
        zorder = plot_num - i
    )
    ax_Bode_ph.scatter(
        omega,
        np.angle(full_time_evolved_impedance)/np.pi*180,
        s = MARKER_SIZE, 
        facecolors = "none", 
        edgecolors = color, 
        zorder = plot_num - i,
        label = label_str
    )
ax_Nyquist.text(-0.2, 1.0, "(A)", transform = ax_Nyquist.transAxes)
ax_Nyquist.set_xlim(-1, 150)
ax_Nyquist.set_ylim(-1, 150)
ax_Nyquist.legend()
ax_Bode_R.text(-0.2, 1.0, "(B)", transform = ax_Bode_R.transAxes)
ax_Bode_ph.legend()

fig.savefig(FIGURE_EXPORT_DIR+"amplitude_dependence.png", dpi = 600)
#%%

