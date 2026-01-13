"""電子的な性質の変化の追跡のためのグラフ
"""

#%% UV-vis Mn(III)の比較
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt
import os

from BFC_libs import common as cmn
import BFC_libs.UV_vis.Shimadzu as smd
import BFC_libs.UV_vis as spc
import BFC_libs.colors as clr

try:
    from . import lib
except(ImportError):
    import lib

dir = "./data/UV-vis/20240325_2-127_K_Na_diff"
spectra_K = smd.load_spectrum_data(dir + "/2-127_in_K.csv")
spectra_Na = smd.load_spectrum_data(dir + "/2-127_in_Na.csv")

diff_K = spc.UV_VisSpectrum(
    _comment = ["2-127", "300mV - 0 mV vs SSCE"],
    _condition = ["in K+"],
    _original_file_path = dir,
    _data_name = "300 mV - 0 mV in K",
    _wavelength = spectra_K.data[0].wavelength,
    _absorption = spectra_K.data[13].absorption - spectra_K.data[0].absorption #index*50 = applied potential in mV vs SSCE
)

diff_Na = spc.UV_VisSpectrum(
    _comment = ["2-127", "300mV - 0 mV vs SSCE"],
    _condition = ["in Na+"],
    _original_file_path = dir,
    _data_name = "300 mV - 0 mV in K",
    _wavelength = spectra_Na.data[0].wavelength,
    _absorption = spectra_Na.data[13].absorption.float_array() - spectra_Na.data[0].absorption.float_array()#index*50 = applied potential in mV vs SSCE
)

fig, ax = cmn.create_standard_matplt_canvas()

ax.set_xlim(350, 800)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"$\Delta$Absorbance (a.u.)")

ax.plot(
    diff_K.wavelength,
    scipy.signal.savgol_filter(diff_K.absorption, 10, 2),
    c = lib.COLOR_K(),
    label = "After electrolysis in K$^+$ solution"
)

ax.plot(
    diff_Na.wavelength,
    scipy.signal.savgol_filter(diff_Na.absorption, 10, 2),
    c = lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$ solution"
)

ax.legend()

plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "UV-vis_Mn(III)_comparison.png")

# ---------------y軸ずらした図----------------------
"""fig, ax = cmn.create_standard_matplt_canvas()

ax.set_xlim(350, 800)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"$\Delta$Absorbance (a.u.)")

ax.plot(
    diff_K.wavelength,
    scipy.signal.savgol_filter(diff_K.absorption, 10, 2),
    c = clr.blue
)

ax.plot(
    diff_Na.wavelength,
    scipy.signal.savgol_filter(diff_Na.absorption, 10, 2) + 0.01,
    c = clr.red    
)

plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "UV-vis_Mn(III)_comparison_y-mod.png")
"""
"""ax.plot(
    diff_K.wavelength,
    scipy.signal.savgol_filter(diff_K.absorption, 10, 4) - scipy.signal.savgol_filter(diff_Na.absorption, 10, 4)
)"""

#%% UV-vis 電位依存性
import numpy as np
import scipy.signal
import matplotlib.pyplot as plt

from enum import Enum

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.UV_vis.Shimadzu as smd
import BFC_libs.UV_vis as spc

dir = "C:/Users/Koichi Yatsuzuka/Documents/2_MnO2_Tafel_Na_K/paper_MnO2_Tafel_Na_K/Figures_Na_K/main_figures/data/UV-vis/20240325_2-127_K_Na_diff"
spectra_K = smd.load_spectrum_data(dir + "/2-127_in_K.csv")
spectra_Na = smd.load_spectrum_data(dir + "/2-127_in_Na.csv")


class Electrolyte(Enum):
    K = 0
    Na = 1

class Abs(Enum):
    abs_600 = 0
    abs_780 = 1
    abs_450 = 2

fig, ax = cmn.create_standard_matplt_canvas()

ax.set_xlim(350, 800)

ax.set_xlabel("Wavelength (nm)")
ax.set_ylabel(r"$\Delta$Absorbance (a.u.)")

for i in range(1,14):
    if i != 13:
        continue
    ax.plot(
        spectra_K.data[0].wavelength,
        spectra_K.data[i].absorption - spectra_K.data[0].absorption,
        c = clr.blue
    )
    ax.plot(
        spectra_Na.data[0].wavelength,
        spectra_Na.data[i].absorption - spectra_Na.data[0].absorption,
        c = clr.red
    )

abs: dict[tuple[Electrolyte, Abs], list[spc.Absorption]] = {
    (Electrolyte.K, Abs.abs_600): [], 
    (Electrolyte.K, Abs.abs_780): [],
    (Electrolyte.K, Abs.abs_450): [],
    (Electrolyte.Na, Abs.abs_600): [], 
    (Electrolyte.Na, Abs.abs_780): [],
    (Electrolyte.Na, Abs.abs_450): [],
}

index_600_K = spectra_K.data[0].wavelength.find(600.0)
index_780_K = spectra_K.data[0].wavelength.find(780.0)
index_450_K = spectra_K.data[0].wavelength.find(450.0)
index_600_Na = spectra_Na.data[0].wavelength.find(600.0)
index_780_Na = spectra_Na.data[0].wavelength.find(780.0)
index_450_Na = spectra_Na.data[0].wavelength.find(450.0)

for i in range(14):
    abs[Electrolyte.K, Abs.abs_600].append(spectra_K.data[i].absorption[index_600_K] \
                                           - spectra_K.data[i].absorption[index_450_K])
    abs[Electrolyte.K, Abs.abs_780].append(spectra_K.data[i].absorption[index_780_K]\
                                           - spectra_K.data[i].absorption[index_450_K])
    abs[Electrolyte.K, Abs.abs_450].append(spectra_K.data[i].absorption[index_450_K])
    
    abs[Electrolyte.Na, Abs.abs_600].append(spectra_Na.data[i].absorption[index_600_Na] \
                                            -spectra_Na.data[i].absorption[index_450_Na])
    abs[Electrolyte.Na, Abs.abs_780].append(spectra_Na.data[i].absorption[index_780_Na]\
                                            -spectra_Na.data[i].absorption[index_450_Na])

fig2, ax2 = cmn.create_standard_matplt_canvas()
ax2.set_xlabel("Potential (V vs SHE)")
ax2.set_ylabel(r"$\Delta$ Absorbance")
ax2.scatter(
    np.linspace(0.2, 0.9, 14),
    abs[Electrolyte.K, Abs.abs_600] - abs[Electrolyte.K, Abs.abs_600][0],
    label = "K+ 600 nm"
)
ax2.scatter(
    np.linspace(0.2, 0.9, 14),
    abs[Electrolyte.K, Abs.abs_780] - abs[Electrolyte.K, Abs.abs_780][0],
    label = "K+ 780 nm"
)
"""ax2.scatter(
    np.linspace(0.2, 0.9, 14),
    abs[Electrolyte.K, Abs.abs_450] - min(abs[Electrolyte.K, Abs.abs_450]),
    label = "K+ 450 nm"
)"""
ax2.scatter(
    np.linspace(0.2, 0.9, 14),
    abs[Electrolyte.Na, Abs.abs_600] - abs[Electrolyte.Na, Abs.abs_600][0],
    label = "Na+ 600 nm"
)
ax2.scatter(
    np.linspace(0.2, 0.9, 14),
    abs[Electrolyte.Na, Abs.abs_780] - abs[Electrolyte.Na, Abs.abs_780][0],
    label = "Na+ 780 nm"
)

ax2.legend()



import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from typing import Final

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.photoelectron as pe
import BFC_libs.photoelectron.VersaProbe as vp
import BFC_libs.fitting_support as fit

from BFC_libs.photoelectron import HELIUM_UV_ENERGY

df_full_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterK_UPS_full.csv")
df_full_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterNa_UPS_full.csv")
df_fermi_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterK_UPS_fermi.csv")
df_fermi_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterNa_UPS_fermi.csv")

normd_data_K = pe.PhotoelectronSpectrum(
    _comment = [],
    _condition = [],
    _original_file_path = "",
    _data_name = "K normd",
    _photoelectron_energy = df_full_K[0].x  + pe.PhotoelectronEnergy(5.0), 
    _photoelectron_intensity = df_full_K[0].y.normalize()
)

normd_data_Na = pe.PhotoelectronSpectrum(
    _comment = [],
    _condition = [],
    _original_file_path = "",
    _data_name = "Na normd",
    _photoelectron_energy = df_full_Na[0].x + pe.PhotoelectronEnergy(5.0),
    _photoelectron_intensity = df_full_Na[0].y.normalize()
)

K_raise = normd_data_K.slice(pe.PhotoelectronEnergy(16.5), pe.PhotoelectronEnergy(16.4))
Na_raise = normd_data_Na.slice(pe.PhotoelectronEnergy(16.7), pe.PhotoelectronEnergy(16))

initial_fit_params_K = fit.TotalFunc(
    [fit.Constant(3.2*10E7),
    fit.Linear(-2*10E6)],
)
initial_fit_params_Na = fit.TotalFunc(
    [fit.Constant(1.88*10E6),
    fit.Linear(-1.3*10E5)],
)

K_fit_res = fit.fitting(K_raise.x.float_array(), K_raise.y.float_array(),initial_fit_params_K)
K_raise_slope = K_fit_res[0].func_components[1].params()
K_raise_constant = K_fit_res[0].func_components[0].params()

Na_fit_res = fit.fitting(Na_raise.x.float_array(), Na_raise.y.float_array(),initial_fit_params_Na)
Na_raise_slope = Na_fit_res[0].func_components[1].params()
Na_raise_constant = Na_fit_res[0].func_components[0].params()

Na_fermi_potential = HELIUM_UV_ENERGY.value - -1 * Na_raise_constant/Na_raise_slope
K_fermi_potential = HELIUM_UV_ENERGY.value  - -1 * K_raise_constant/K_raise_slope
#
# -------------------------ノイズによってベースラインが上がるので調整----------------------------
"""base_index = (data_K_narrow.x.find(pe.PhotoelectronEnergy(-1))[0], data_K_narrow.x.find(pe.PhotoelectronEnergy(-1.9))[0])
baseline_K = data_K_narrow.y.normalize()[base_index[0]:base_index[1]].float_array()
baseline_Na = data_Na_narrow.y.normalize()[base_index[0]:base_index[1]].float_array()

base_K = np.average(baseline_K)
base_Na = np.average(baseline_Na)"""

#----------------------------描画---------------------------------------------
fig, ax = cmn.create_standard_matplt_canvas()
ax.set_xlabel("Binding energy (eV)")
ax.set_ylabel("Normalized intensity (a.u.)")
ax.set_xlim(18, -2)

ax_ins = ax.inset_axes([0.45, 0.6, 0.45, 0.35])

ax_ins.set_xlim(-0.5, 8.5)
ax_ins.set_ylim(-0.005, 0.3)

ax.indicate_inset_zoom(ax_ins)

ax_ins.set_xlim(8.5, -0.5)
ax_ins.set_ylim(-0.005, 1.05)
ax_ins.set_xlabel("Binding energy (eV)")
ax_ins.set_ylabel("Normalized\nintensity\n(a.u.)")
ax_ins.tick_params(labelleft = False, left = False, right = False)

ax.plot(
    normd_data_K.x,
    normd_data_K.y,
    c = clr.blue
)
ax.plot(
    normd_data_Na.x,
    normd_data_Na.y,
    c=clr.red
)

ax_ins.plot(
    df_fermi_K[0].x + pe.PhotoelectronEnergy(5.0),
    df_fermi_K[0].y.normalize(),
    c = clr.blue
)
ax_ins.plot(
    df_fermi_Na[0].x + pe.PhotoelectronEnergy(5.0),
    df_fermi_Na[0].y.normalize(),
    c = clr.red
)


plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "UPS.png", dpi=600)
#%% UPS
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from typing import Final

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.photoelectron as pe
import BFC_libs.photoelectron.VersaProbe as vp
import BFC_libs.fitting_support as fit

from BFC_libs.photoelectron import HELIUM_UV_ENERGY

try:
    from . import lib
except(ImportError):
    import lib
n = 2

match n:
    case 0:
        df_full_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterK_UPS_full.csv")
        df_full_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterNa_UPS_full.csv")
        df_fermi_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterK_UPS_fermi.csv")
        df_fermi_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterNa_UPS_fermi.csv")

    case 1:
        df_full_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/29.4.NO3-K2 WholeRegion.csv")
        df_full_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/42.2.NO5-Na2 WholeRegion.csv")
        df_fermi_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/30.4.NO3-K2 HOMO.csv")
        df_fermi_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/43.2.NO5-Na2 HOMO.csv")
    case 2:
        df_full_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/24.3.NO2-K1 WholeRegion.csv")
        df_full_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/37.1.NO4-Na1 WholeRegion.csv")
        df_fermi_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/25.3.NO2-K1 HOMO.csv")
        df_fermi_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/38.1.NO4-Na1 HOMO.csv")

df_full_as_made = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/19.2.NO1-bfr WholeRegion.csv")
df_fermi_as_made = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/20.2.NO1-bfr HOMO.csv")

df_full_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/42.2.NO5-Na2 WholeRegion.csv")
df_fermi_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240611_extra_UPS_repr/43.2.NO5-Na2 HOMO.csv")


normd_as_made = pe.PhotoelectronSpectrum(
    _comment = [],
    _condition = [],
    _original_file_path = "",
    _data_name = "As made normd",
    _photoelectron_energy = df_full_as_made[0].x  + pe.PhotoelectronEnergy(5.0), 
    _photoelectron_intensity = df_full_as_made[0].y.normalize()
)

normd_data_K = pe.PhotoelectronSpectrum(
    _comment = [],
    _condition = [],
    _original_file_path = "",
    _data_name = "K normd",
    _photoelectron_energy = df_full_K[0].x  + pe.PhotoelectronEnergy(5.0), 
    _photoelectron_intensity = df_full_K[0].y.normalize()
)

normd_data_Na = pe.PhotoelectronSpectrum(
    _comment = [],
    _condition = [],
    _original_file_path = "",
    _data_name = "Na normd",
    _photoelectron_energy = df_full_Na[0].x + pe.PhotoelectronEnergy(5.0),
    _photoelectron_intensity = df_full_Na[0].y.normalize()
)

K_raise = normd_data_K.slice(pe.PhotoelectronEnergy(16.55), pe.PhotoelectronEnergy(16.5))
Na_raise = normd_data_Na.slice(pe.PhotoelectronEnergy(16.57), pe.PhotoelectronEnergy(16.45))

initial_fit_params_K = fit.FittingParameters(
    [fit.Constant("", fit.ValueWithBounds(3.2*10E7)),
    fit.Linear("",fit.ValueWithBounds(-2*10E6))],
)
initial_fit_params_Na = fit.FittingParameters(
    [fit.Constant("", fit.ValueWithBounds(1.88*10E6)),
    fit.Linear("",fit.ValueWithBounds(-1.3*10E5))],
)

K_fit_res = fit.fitting(K_raise.x.float_array(), K_raise.y.float_array(),initial_fit_params_K)
K_raise_slope = K_fit_res.func_components[1].params().value
K_raise_constant = K_fit_res.func_components[0].params().value

Na_fit_res = fit.fitting(Na_raise.x.float_array(), Na_raise.y.float_array(),initial_fit_params_Na)
Na_raise_slope = Na_fit_res.func_components[1].params().value
Na_raise_constant = Na_fit_res.func_components[0].params().value

Na_fermi_potential = HELIUM_UV_ENERGY.value - -1 * Na_raise_constant/Na_raise_slope
K_fermi_potential = HELIUM_UV_ENERGY.value  - -1 * K_raise_constant/K_raise_slope
#



#----------------------------描画---------------------------------------------
fig = plt.figure(figsize = (8,3))
plt.subplots_adjust(left = 0.1, top = 0.9, bottom = 0.15, wspace=0.4, hspace=0.2)
ax_full_spectra = fig.add_subplot(1,2,1)

ax_full_spectra.set_xlabel("Binding energy (eV)")
ax_full_spectra.set_ylabel("Normalized intensity (a.u.)")

ax_full_spectra.set_xlim(18, -2)



"""ax_ins = ax_full_spectra.inset_axes([0.45, 0.6, 0.45, 0.35])

ax_ins.set_xlim(-0.5, 8.5)
ax_ins.set_ylim(-0.005, 0.3)

ax_full_spectra.indicate_inset_zoom(ax_ins)

ax_ins.set_xlim(8.5, -0.5)
ax_ins.set_ylim(-1, 1)
ax_ins.set_xlabel("Binding energy (eV)")
ax_ins.set_ylabel("Normalized\nintensity\n(a.u.)")
ax_ins.tick_params(labelleft = False, left = False, right = False)
"""

ax_fermi = fig.add_subplot(1, 2, 2)
ax_fermi.set_xlabel("Binding energy (eV)")
ax_fermi.set_ylabel("Normalized intensity (a.u.)")
ax_fermi.set_xlim(8, -0.5)
#ax_fermi.set_ylim(-0.005, 0.1)

"""ax_fermi_right = ax_fermi.twinx()
ax_fermi_right.set_ylim(-0.03, 0.03)"""

ax_full_spectra.plot(
    normd_as_made.x,
    normd_as_made.y,
    c = lib.clr.black,
    label = "As synthesized"
)

ax_full_spectra.plot(
    normd_data_K.x,
    normd_data_K.y,
    c = lib.COLOR_K(),
    label = "After electrolysis in K$^+$ solution"
)
ax_full_spectra.plot(
    normd_data_Na.x,
    normd_data_Na.y,
    c= lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$ solution"
)

ax_full_spectra.legend()

alphabet_pos_a = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos_no_label, ax_full_spectra)
ax_full_spectra.text(alphabet_pos_a.x, alphabet_pos_a.y, "(a)")



#-------------------------------------

# 0-2 eVをプロットする用
df_fermi_K_near_edge = \
    df_fermi_K[0].slice(pe.PhotoelectronEnergy(-3.05), pe.PhotoelectronEnergy(-7.0))
df_fermi_Na_near_edge = \
    df_fermi_Na[0].slice(pe.PhotoelectronEnergy(-3.05), pe.PhotoelectronEnergy(-7.0))

ax_fermi.plot(
    df_fermi_as_made[0].x + pe.PhotoelectronEnergy(5.0),
    df_fermi_as_made[0].y.normalize(),
    c = lib.clr.black,
    label = "As synthesized"
)

ax_fermi.plot(
    df_fermi_K[0].x + pe.PhotoelectronEnergy(5.0),
    df_fermi_K[0].y.normalize(),
    c = lib.COLOR_K(),
    label = "After electrolysis in K$^+$ solution"
)

ax_fermi.plot(
    df_fermi_Na[0].x + pe.PhotoelectronEnergy(5.0),
    df_fermi_Na[0].y.normalize(),
    c = lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$ solution"
)

ax_fermi.set_ylim(-0.005, 1.4)

ax_fermi.legend()

alphabet_pos_b = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos_no_label, ax_fermi)
ax_fermi.text(alphabet_pos_b.x, alphabet_pos_b.y, "(b)")

"""# 0-2 eVをプロットする用
ax_fermi.plot(
    df_fermi_K_near_edge.x + pe.PhotoelectronEnergy(5.0),
    df_fermi_K_near_edge.y.normalize(),
    c = lib.COLOR_K()
)

ax_fermi.plot(
    df_fermi_Na_near_edge.x + pe.PhotoelectronEnergy(5.0),
    df_fermi_Na_near_edge.y.normalize(),
    c = lib.COLOR_Na()
)"""


"""ax_fermi_right.plot(
    df_fermi_K_near_edge.x + pe.PhotoelectronEnergy(5.0),
    (df_fermi_K_near_edge.y.normalize() - df_fermi_Na_near_edge.y.normalize()),
    c = clr.purple
)"""
#ax_fermi_right.set_ylabel(r"$\Delta$Spectrum")

plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "UPS.png", dpi=600)


print("K Fermi: {} eV".format( K_fermi_potential))
print("Na Fermi: {} eV".format(Na_fermi_potential))


#%% XPS比較 
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from typing import Final

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.photoelectron as pe
import BFC_libs.photoelectron.VersaProbe as vp

data_before = vp.VasaProbeData.load_file("./data/XPS_UPS/20240425_2-134_XPS_bfraft/before_electrolysis.csv")
data_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240425_2-134_XPS_bfraft/after_Na.csv") 
data_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240425_2-134_XPS_bfraft/after_K.csv")

MANGANESE_2P = "Mn2p"
CARBON_1S = "C1s"
ESCA = "Su1s"


fig, ax = cmn.create_standard_matplt_canvas()
ax.set_xlabel("Binding energy (eV)")
ax.set_ylabel("Intensity (a.u.)")
ax.set_xlim(pe.PhotoelectronEnergy(672).value, pe.PhotoelectronEnergy(635).value)
ax.set_ylim(pe.PhotoelectronIntensity(-0.025).value, pe.PhotoelectronIntensity(1.025).value)

ax.plot(
    data_K[MANGANESE_2P].x,
    data_K[MANGANESE_2P].y.normalize(),
    c = clr.blue
)
ax.plot(
    data_Na[MANGANESE_2P].x,
    data_Na[MANGANESE_2P].y.normalize(),
    c = clr.red
)

ax.plot(
    data_K[MANGANESE_2P].x,
    (data_K[MANGANESE_2P].y.normalize() - data_Na[MANGANESE_2P].y.normalize()) * 5 + pe.PhotoelectronIntensity(0.5),
    c = clr.purple
)

plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "XPS_Mn2p.png")

#%% UPS比較(旧)