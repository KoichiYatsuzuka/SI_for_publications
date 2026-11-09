#%%
import impedance_calculater as ic
import BFC_libs.electrochemistry as ec
from BFC_libs.physical_constants import F
#import BFC_libs.common as cmn

#import matplotlib.cm as cm
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import matplotlib.cm as cm
import numpy as np

import common
#from common import TIME_EVOLUTION_OUTPUT_DIR, TIME_EVOLUTION_FILE_FORMAT, FIGURE_EXPORT_DIR
from plot_lib import ACTIVESITES_NUM
from plot_lib import MARKER_SIZE, LINE_WIDTH
from plot_lib import step1, step2
import plot_lib as plot_lib

#Y_TO_K_COEFF = 1/(ACTIVESITES_NUM * F)


applied_potential = ec.PotentialArray(np.linspace(0.0, 0.125, 25))

r_0 = np.array([
    ic.calculate_analytical_impedance(
        step1,
        step2,
        ec.PotentialArray([e]),
        ec.FrequencyArray([0]),
        ACTIVESITES_NUM
    )[0][0].real for e in applied_potential
])

r1 = np.array([
    ic.calculate_analytical_impedance(
        step1,
        step2,
        ec.PotentialArray([e]),
        ec.FrequencyArray([np.inf]),
        ACTIVESITES_NUM
    )[0][0].real for e in applied_potential
])

r2 = r_0-r1

k1 = step1.rate_fw(applied_potential)

k2 = step2.rate_fw(applied_potential)

y1 = 1/r1
y2 = 1/r2

fig = plt.figure(figsize = (4,8))
ax1 = fig.add_axes((0.3,0.6,0.7,0.35))
ax2 = fig.add_axes((0.3,0.1,0.7,0.35))

ax1.set_xscale("log")
ax1.set_yscale("log")
# ax1.set_xlim(250, 580)
# ax1.set_ylim(3.3E-3, 5.5E-3)

ax2.set_xscale("log")
ax2.set_yscale("log")
# ax2.set_xlim(0.38, 1.05)
# ax2.set_ylim(0.045, 0.09)

ax1.set_xlabel(r"$k_1$ [1/s]")
ax1.set_ylabel(r"$1/R_{\mathrm{ct1}}$ [S]")

ax2.set_xlabel(r"$k_2$ [1/s]")
ax2.set_ylabel(r"$1/R_{\mathrm{ct2}}$ [S]")

# The x data of both panels span less than one decade, so the default
# LogLocator finds only a single major tick (10^3 in A, 10^0 in B) and the
# axis ends up with one lone label. Labelling the minor ticks as well
# restores a readable scale: LogLocator(subs) puts minor ticks at 2,3,...,9
# times each decade, and LogFormatterSciNotation with a wide
# minor_thresholds forces those minor ticks to be labelled instead of being
# blanked out (matplotlib suppresses minor labels by default once the axis
# covers more than a fraction of a decade).
def log_formatter(x, pos):
    if x <= 0:
        return ""

    exp = int(np.floor(np.log10(x)))
    mant = x / 10**exp

    if np.isclose(mant, 1):
        return rf"$10^{{{exp}}}$"
    elif np.isclose(mant, 5):
        return rf"$5\times10^{{{exp}}}$"
    elif np.isclose(mant, 2):
        return rf"$2\times10^{{{exp}}}$"
    else:
        return ""

_major_subs = (2.0,5.0, 1.0)
for _ax in (ax1, ax2):
    _ax.xaxis.set_major_locator(tck.LogLocator(base=10.0, subs=_major_subs))
    #_ax.xaxis.set_minor_formatter(LogFormatterSciNotation(minor_thresholds=(np.inf, np.inf)))
    _ax.xaxis.set_major_formatter(tck.FuncFormatter(log_formatter))
    #_ax.tick_params(axis="x", which="minor", labelsize="small")


ax1.scatter(
    k1,
    y1,
    zorder = 1,
    c = cm.winter(0.2),
    s = MARKER_SIZE*2
)

# ax1.plot(
#     ax1.get_xlim(),
#     ax1.get_ylim(),
#     c = "#808080",
#     ls = "--",
#     zorder = 0
# )

ax2.scatter(
    k2,
    y2,
    zorder = 1,
    c = cm.winter(0.2+2/3*0.7),
    s = MARKER_SIZE*2
)

# ax2.plot(
#     ax2.get_xlim(),
#     ax2.get_ylim(),
#     c = "#808080",
#     ls = "--",
#     zorder = 0
# )

ax1.text(-0.2, 1.0, "(A)", transform = ax1.transAxes)
ax2.text(-0.2, 1.0, "(B)", transform = ax2.transAxes)

fig.savefig(common.FIGURE_EXPORT_DIR+"Rct_vs_E.png", dpi = 600)