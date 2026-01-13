"""
材料同定のためのGI-XRDとRaman
"""

#%%
"""FTO、合成直後のGI-XRD

"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Final

from BFC_libs import common as cmn
from BFC_libs import colors as clr
from BFC_libs import XRD as xrd
from BFC_libs.XRD import ICDD 
from BFC_libs.XRD import SmartLab as sl

try:
    from . import lib
except(ImportError):
    import lib

dir_2_115 = "./data/XRD/20240229_2-115_MnO2_as_synth_after_electrolysis/"
dir_2_138 = "./data/XRD/20240510_2-138_bare_FTO/"
dir_PDF = "./data/XRD/ICDD/"

data_as_synth: Final = sl.read_smart_lab_1D_data(
    dir_2_115 + "20240229_2-115_004_fine_omega_100mdeg.txt"
)[0]

data_after_K: Final = sl.read_smart_lab_1D_data(
    dir_2_115 + "20240229_2-115_006_after_K_fine_omega_100mdeg.txt"
)[0]

data_after_Na: Final = sl.read_smart_lab_1D_data(
    dir_2_115 + "20240229_2-115_006_after_Na_fine_omega_100mdeg.txt"
)[0]

data_FTO: Final = sl.read_smart_lab_1D_data(
    dir_2_138 + "20240510_2-138_002_bare_FTO_fine.txt"
)[0]

df_pdf_tmp: Final = pd.read_csv(dir_PDF + "PDF_card_alphaMnO2_00-044-0141.csv")
pdf_MnO2: Final = ICDD.ICDD(
    _comment = "PDF: 00-044-0141",
    _data_name = "PDF: 00-044-0141",
    _condition = "",
    _original_file_path = dir_PDF + "PDF_card_alphaMnO2_00-044-0141.csv",
    _two_theta = xrd.ThetaArray(df_pdf_tmp["2theta"].values),
    _intensity = xrd.DiffractionIntensityArray(df_pdf_tmp["intensity"].values),
    d_value= df_pdf_tmp["d value"].values,
    Miller_index= ICDD.MillerIndex(
        df_pdf_tmp["h"].values, 
        df_pdf_tmp["k"].values, 
        df_pdf_tmp["l"].values)
)

fig, ax = cmn.create_standard_matplt_canvas()

x_lim_left = xrd.Theta(5)
x_lim_right = xrd.Theta(70)
ax.set_xlim(x_lim_left.__float__(), x_lim_right.__float__())
ax.set_ylim(-0.2, 1.2)
alphabet_pos = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos_no_label, ax)
ax.set_xlabel(r"2$\theta$ (degree)")
ax.set_ylabel("Diffraction intensity (a.u.)")
ax.tick_params(labelleft=False, left=False, right=False)
ax.text(alphabet_pos.x, alphabet_pos.y, "(a)")

FTO_plot_range = np.where((data_FTO.x<x_lim_right) * (data_FTO.x>x_lim_left))
ax.plot(
    data_FTO.two_theta[FTO_plot_range],
    data_FTO.intensity[FTO_plot_range].normalize()*0.5,
    c=lib.COLOR_K(),
    label = "FTO"
)

synth_plot_range = np.where((data_as_synth.x<x_lim_right) * (data_as_synth.x>x_lim_left))
ax.plot(
    data_as_synth.two_theta[synth_plot_range],
    data_as_synth.intensity[synth_plot_range].normalize()+xrd.DiffractionIntensity(0.25),
    c = lib.COLOR_Na(),
    label = r"Synthesized $\mathrm{\alpha}$-MnO$_2$"
)

ax.vlines(
    x = pdf_MnO2.two_theta,
    ymin = -0.2,
    ymax= pdf_MnO2.intensity * 0.2/100 - xrd.DiffractionIntensity(0.18),
    colors= clr.green,
    lw = 2,
    label = r"Reference ($\mathrm{\alpha}$-MnO$_2$)"
)

ax.legend()

fig.savefig("exported_figures/characterization_GI_XRD", dpi=600)

#%%
"""合成直後のラマン

Todo: ω = 0.1°のFTOのXRDを測定
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import ArrowStyle

from typing import Final
from collections import namedtuple

from BFC_libs import common as cmn
from BFC_libs import colors as clr
from BFC_libs import Raman as rmn
from BFC_libs.Raman import Nanophoton

try:
    from . import lib
except(ImportError):
    import lib

dir_2_107 = "./data/Raman/20240207_2-107_Raman_bfr_aft_repro/"

data = Nanophoton.read_1D_data(
    dir_2_107 + "20240207_2-107_comparison.txt"
)
spectrum_before = data[0]

Point= namedtuple('Point', 'x y')

#振動モード(Ref. Tanja Barudvzija, J. Alloys Compd., 2017)
VibrationAnnotation = \
    namedtuple('VibrationAnnotation', 'wavenumber x_rel_pos y_rel_pos vibration_mode')

Vib = VibrationAnnotation
vibration_annotations = [
    Vib(181, 0, 0.1, "$E_g$"),
    Vib(296, -40, 0.1, "*"),
    Vib(327, 0, -0.05, "*"),
    Vib(387, 10, 0, "$E_g$"),
    Vib(470, -30, 0.3, "*"),
    Vib(513, 0, 0, "$E_g$"),
    Vib(580, 0, 0, "$A_g$"),
    Vib(638, 50, 0, "$A_g$"),
    Vib(745, 0, 0, "$A_g$"),

]
peaktop_list: list[Point] = []
for peakpos in vibration_annotations:
    # データ横軸の配列のうち、もっともピーク位置に近い横軸位置を取得
    nearest_wavenum_index = np.argmin(np.abs(
        spectrum_before.wavenumber.float_array() - peakpos.wavenumber
    ))
    #その時のピークの高さ取得
    peak_hight = spectrum_before.intensity[nearest_wavenum_index].__float__()
    peaktop_list.append(Point(peakpos.wavenumber, peak_hight))


fig, ax = cmn.create_standard_matplt_canvas()
ax.tick_params(labelleft=False, left=False, right=False)
x_lim_left = rmn.Wavenumber(100)
x_lim_right = rmn.Wavenumber(900)
ax.set_xlim(x_lim_left.__float__(), x_lim_right.__float__())
ax.set_ylim(0., 1.5)
alphabet_pos = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos_no_label, ax)
ax.set_xlabel(r"Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Raman intensity (a.u.)")
ax.text(alphabet_pos.x, alphabet_pos.y, "(b)")

ax.plot(
    spectrum_before.wavenumber,
    spectrum_before.intensity,
    c=clr.black
)

STD_ARROW_HIEGHT: Final = 0.2
for i,peaktop in enumerate(peaktop_list):

    ax.annotate(
        text=vibration_annotations[i].vibration_mode+"\n"+\
            str(vibration_annotations[i].wavenumber),
        xy=(peaktop.x,peaktop.y),
        xytext=(
            peaktop.x + vibration_annotations[i].x_rel_pos,
            peaktop.y + vibration_annotations[i].y_rel_pos + STD_ARROW_HIEGHT
        ),
        xycoords="data",
        ha="center",
        arrowprops=dict(\
            arrowstyle = ArrowStyle("->", widthA=0.3, widthB=0.3),
            connectionstyle="arc3"
        )
    )

fig.savefig("./exported_figures/characterization_raman.png", dpi=600)

#%% Elemental analysis (XPS Na1s)
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from typing import Final

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.photoelectron as pe
import BFC_libs.photoelectron.VersaProbe as vp
try:
    from . import lib
except(ImportError):
    import lib

Na1s = "Na1s"

df_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterK_C1s_Na1s.csv")
df_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240520_2-141_XPSUPS_repr/AfterNa_C1s_Na1s.csv")
df_as_made = vp.VasaProbeData.load_file("./data/XPS_UPS/20240528_2-144_XPSUPS_repr/AsMade.csv")

fig, ax = cmn.create_standard_matplt_canvas()
ax.set_xlim(df_K[Na1s].x.max().value, df_K[Na1s].x.min().value)
ax.set_ylim(7000, 15000)

ax.set_xlabel("Biding energy (eV)")
ax.set_ylabel("Intensity (count)")

ax.plot(
    df_as_made[Na1s].x,
    df_as_made[Na1s].y,
    c = lib.clr.black,
    label = "As synthesized"
)

ax.plot(
    df_K[Na1s].x,
    df_K[Na1s].y,
    c = lib.COLOR_K(),
    label = "After electrolysis in K$^+$ solution"
)

ax.plot(
    df_Na[Na1s].x,
    df_Na[Na1s].y,
    c = lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$ solution"
)

ax.legend()

fig.savefig("./exported_figures/characterization_XPS_Na1s.png", dpi=600)


#%% Elemental analysis (XPS K2p)
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from typing import Final

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.photoelectron as pe
import BFC_libs.photoelectron.VersaProbe as vp
try:
    from . import lib
except(ImportError):
    import lib

K2p = "K2p"

df_K = vp.VasaProbeData.load_file("./data/XPS_UPS/20240528_2-144_XPSUPS_repr/AfterK.csv")
df_Na = vp.VasaProbeData.load_file("./data/XPS_UPS/20240528_2-144_XPSUPS_repr/AfterNa.csv")
df_as_made = vp.VasaProbeData.load_file("./data/XPS_UPS/20240528_2-144_XPSUPS_repr/AsMade.csv")

fig, ax = cmn.create_standard_matplt_canvas()
ax.set_xlim(df_K[K2p].x.max().value, df_K[K2p].x.min().value)
ax.set_xlabel("Biding energy (eV)")
ax.set_ylabel("Intensity (count)")

ax.plot(
    df_as_made[K2p].x,
    df_as_made[K2p].y,
    c = lib.clr.black,
    label = "As synthesized"
)


ax.plot(
    df_K[K2p].x,
    df_K[K2p].y,
    c = lib.COLOR_K(),
    label = "After electrolysis in K$^+$ solution"
)

ax.plot(
    df_Na[K2p].x,
    df_Na[K2p].y,
    c = lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$ solution"
)

ax.legend()

fig.savefig("./exported_figures/characterization_XPS_K2p.png", dpi=600)