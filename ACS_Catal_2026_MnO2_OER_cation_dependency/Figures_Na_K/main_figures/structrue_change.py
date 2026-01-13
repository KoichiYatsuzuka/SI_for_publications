#%% ラマン変化（旧）
# 現在未使用だし、動かない
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from enum import Enum

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.Raman as rmn
import BFC_libs.Raman.Nanophoton as nph
import BFC_libs.fitting_support as fit

data_file = nph.read_1D_data("data/Raman/20240207_2-107_Raman_bfr_aft_repro/20240207_2-107_comparison.txt")
diff_data_file = nph.read_1D_data("data/Raman/20240207_2-107_Raman_bfr_aft_repro/20240207_2-107_diff.txt")
class DataNameRaw(Enum):
    K_before = 1
    K_after = 2
    Na_before = 3
    Na_after = 4

class DataNameDiff(Enum):
    K_diff = 11
    Na_diff = 12
    before_diff = 13
    after_diff = 14


spectra:dict[DataNameRaw, rmn.RamanSpectrum] = {
    DataNameRaw.K_before: data_file[0],
    DataNameRaw.K_after: data_file[2],
    DataNameRaw.Na_before: data_file[3],
    DataNameRaw.Na_after: data_file[4]
}

"""diff_spectra:dict[DataNameDiff, rmn.RamanSpectrum] = {
    DataNameDiff.K_diff: diff_data_file[1],
    DataNameDiff.Na_diff: diff_data_file[0],
    DataNameDiff.before_diff: diff_data_file[3],
    DataNameDiff.after_diff: diff_data_file[2]
}"""



class MainPeak(Enum):
    mp580 = 0
    mp630 = 1

#スペクトル分割
splitted_spectra:dict[(DataNameRaw, MainPeak), rmn.RamanSpectrum] = {}
#spectra_630:dict[DataNameRaw, rmn.RamanSpectrum] = {}

for data_series in DataNameRaw:
    
    splitted_spectra[(data_series, MainPeak.mp580)]\
        =spectra[data_series].slice(rmn.Wavenumber(560), rmn.Wavenumber(605))
    
    splitted_spectra[(data_series, MainPeak.mp630)]\
        =spectra[data_series].slice(rmn.Wavenumber(605), rmn.Wavenumber(670))
    
# フィッティング準備
initial_fit_params: dict[MainPeak, fit.TotalFunc] = {}

initial_fit_params[MainPeak.mp580] = fit.TotalFunc(
    func_components=[
        fit.Linear(0.004),
        fit.Constant(-2),
        fit.LorentzPeak(
            _center = 579,
            _width = 15,
            _ampletude = 20
        ),
        fit.LorentzPeak(
            _center = 585,
            _width = 15,
            _ampletude = 20
        )
    ],
    minimum_bounds=[
        fit.Linear(-0.01),
        fit.Constant(-10),
        fit.LorentzPeak(
            _center = 576,
            _width = 1,
            _ampletude = 2
        ),
        fit.LorentzPeak(
            _center = 581,
            _width = 1,
            _ampletude = 0.1
        )
    ],
    maximum_bounds=[
        fit.Linear(0.01),
        fit.Constant(10),
        fit.LorentzPeak(
            _center = 580.9,
            _width = 50,
            _ampletude = 200
        ),
        fit.LorentzPeak(
            _center = 590,
            _width = 50,
            _ampletude = 200
        )
    ]
)

initial_fit_params[MainPeak.mp630] = fit.TotalFunc(
    func_components=[
        fit.Linear(-0.0015),
        fit.Constant(1.5),
        fit.LorentzPeak(
            _center = 635,
            _width = 21,
            _ampletude = 35
        ),
        fit.LorentzPeak(
            _center = 655,
            _width = 12,
            _ampletude = 5
        )
    ],
    minimum_bounds=[
        fit.Linear(-0.01),
        fit.Constant(-10),
        fit.LorentzPeak(
            _center = 630,
            _width = 1,
            _ampletude = 2
        ),
        fit.LorentzPeak(
            _center = 650,
            _width = 1,
            _ampletude = 0.1
        )
    ],
    maximum_bounds=[
        fit.Linear(0.01),
        fit.Constant(10),
        fit.LorentzPeak(
            _center = 640,
            _width = 50,
            _ampletude = 200
        ),
        fit.LorentzPeak(
            _center = 657,
            _width = 50,
            _ampletude = 200
        )
    ]
)

fit_res: dict[tuple[DataNameRaw, MainPeak], fit.FittingResult] = {}

fit_patterns: list[tuple[DataNameRaw, MainPeak]] = [
    (DataNameRaw.K_after, MainPeak.mp580),
    (DataNameRaw.K_after, MainPeak.mp630),
    (DataNameRaw.Na_after, MainPeak.mp580),
    (DataNameRaw.Na_after, MainPeak.mp630)
]

for fit_pattern in fit_patterns:
    fit_res[fit_pattern], _ = \
        fit.fitting(
            splitted_spectra[fit_pattern].x.float_array(),
            splitted_spectra[fit_pattern].y.float_array(),
            initial_fit_params[fit_pattern[1]]
        )
residual: dict[tuple[DataNameRaw, MainPeak], rmn.RamanSpectrum] = {}
for fit_pattern in fit_patterns:
    residual[fit_pattern] = \
        rmn.RamanSpectrum(
            _comment = "residual",
            _condition = "",
            _original_file_path = "",
            _data_name = "res_" + str(fit_pattern),
            _wavenumber = splitted_spectra[fit_pattern].x,
            _intensity = splitted_spectra[fit_pattern].y - \
                rmn.RammanIntensityArray(fit_res[fit_pattern].calc(splitted_spectra[fit_pattern].x.float_array()))
        )


def show_fit_res(spc: rmn.RamanSpectrum, fit_res: fit.FittingResult, ax: plt.Axes, y_offset=0.0):
    #spc = spectra_580[key]
    #res, cov = fit.fitting(spc.x, spc.y, fit_fucs)

    

    def plot_fitting(fit_res: fit.FittingResult, x: np.ndarray, ax: plt.Axes, y_offset = 0.0):


        funcs = fit_res.func_components

        # baselineプロット
        ax.plot(
            x, 
            funcs[0].calc(x)+funcs[1].calc(x) + y_offset,
            c="#AAAAAA"
            )

        #ピークのプロット
        ax.plot(
            x, 
            funcs[0].calc(x)+funcs[1].calc(x) + funcs[2].calc(x)+ y_offset,
            c=clr.green
            )
        ax.plot(
            x, 
            funcs[0].calc(x)+funcs[1].calc(x) + funcs[3].calc(x) + y_offset,
            c=clr.blue
            )

        #フィッティング関数全体のプロット
        ax.plot(
            x, 
            fit_res.calc(x)+ y_offset,
            c=clr.red
            )
    plot_fitting(fit_res, spc.x.float_array(), ax, y_offset)

    return

Y_OFFSET = 1.0

fig = plt.figure(figsize=(6,4))
ax: plt.Axes = fig.add_axes([0.2, 0.2, 0.7, 0.7])
ax.set_xlim(555, 670)
#ax.set_ylim(0.0, 1.8)
ax.set_xlabel("Wavenumber (cm$^{-1}$))")
ax.set_ylabel("Raman intensity (a.u.)")

ax.plot(
        spectra[DataNameRaw.K_after].x,
        spectra[DataNameRaw.K_after].y.float_array(),
        c="black",
        lw = 2.5
    )
ax.plot(
        spectra[DataNameRaw.Na_after].x,
        spectra[DataNameRaw.Na_after].y.float_array() + Y_OFFSET,
        c="black",
        lw = 2.5
    )

class ElemPeak(Enum):
    ep578 = 0
    ep585 = 1
    ep638 = 2
    ep655 = 3

for fit_pattern in fit_patterns:

    show_fit_res(
        splitted_spectra[fit_pattern], 
        fit_res[fit_pattern],
        ax, 
        0.0 if fit_pattern[0]==DataNameRaw.K_after else Y_OFFSET
    )

peak_pos:dict[tuple[DataNameRaw, ElemPeak], cmn.Point] = {}

for fit_pattern in fit_patterns:
    sab_peak1 = fit_res[fit_pattern].func_components[2]
    sab_peak2 = fit_res[fit_pattern].func_components[3]
    
    if isinstance(sab_peak1, fit.PeakFunction):
        x = sab_peak1.center
        peak_pos[
            fit_pattern[0],(ElemPeak.ep578 if fit_pattern[1] == MainPeak.mp580 else ElemPeak.ep638)
        ] = (
            cmn.Point(
                x, 
                fit_res[fit_pattern].func_components[0].calc(x) + \
                    fit_res[fit_pattern].func_components[1].calc(x) + \
                    sab_peak1.calc(x) + \
                    (0.0 if fit_pattern[0]==DataNameRaw.K_after else Y_OFFSET)
                )
        )
    else:
        raise TypeError
    
    if isinstance(sab_peak2, fit.PeakFunction):
        x = sab_peak2.center
        peak_pos[
            fit_pattern[0],(ElemPeak.ep585 if fit_pattern[1] == MainPeak.mp580 else ElemPeak.ep655)
        ] = (
            cmn.Point(
                x, 
                fit_res[fit_pattern].func_components[0].calc(x) + \
                    fit_res[fit_pattern].func_components[1].calc(x) + \
                    sab_peak2.calc(x) + \
                    (0.0 if fit_pattern[0]==DataNameRaw.K_after else Y_OFFSET)
                )
        )
    else:
        raise TypeError

for peak in ElemPeak:
    x = [
        peak_pos[(DataNameRaw.K_after, peak)].x,
        peak_pos[(DataNameRaw.Na_after, peak)].x
    ]
    y = [
        peak_pos[(DataNameRaw.K_after, peak)].y,
        peak_pos[(DataNameRaw.Na_after, peak)].y
    ]

    ax.plot(x, y, c="black", ls = ":")

for fit_pattern in fit_patterns:
    ax.plot(
        residual[fit_pattern].x,
        residual[fit_pattern].y * 50 + \
            (rmn.RammanIntensity(Y_OFFSET) if fit_pattern[0]==DataNameRaw.K_after else rmn.RammanIntensity(0))
        )
#%% EXAFS
import matplotlib.pyplot as plt
import os

from enum import Enum

from BFC_libs import common as cmn
import BFC_libs.XAS as xas
import BFC_libs.XAS.Athena as atn
import BFC_libs.colors as clr

try:
    from . import lib
except(ImportError):
    import lib

data = atn.load_athena_EXAFS_file(
    "./data/XAS/20240410_2-132_CEY_after_electrolysis/alpha-MnO2_Mn-K.chir_mag"
    )

fig,ax = cmn.create_standard_matplt_canvas()

ax.set_xlim(0, 4)
ax.set_ylim(-0.1, 3.2)
ax.set_xlabel("Distance (Å)")
ax.set_ylabel("Intensity (a.u.)")

ax_ins = ax.inset_axes([0.55, 0.5, 0.4, 0.24])

ax_ins.set_xlim(1.3, 1.5)
ax_ins.set_ylim(1.95, 2.25)
ax.indicate_inset_zoom(ax_ins)

ax.plot(
    data["merge_3-K_CEY"].x,
    data["merge_3-K_CEY"].y,
    c= lib.COLOR_K(),
    label = "After electrolysis in K$^+$ solution"
)
"""ax.plot(
    data["merge_3-K_CEY"].x,
    data["merge_3-K_CEY"].chir_real,
    c=clr.blue,
    ls = ":"    
)
ax.plot(
    data["merge_3-K_CEY"].x,
    data["merge_3-K_CEY"].chir_imaginary,
    c=clr.blue,
    ls = "--"    
)"""
ax_ins.plot(
    data["merge_3-K_CEY"].x,
    data["merge_3-K_CEY"].y,
    c=lib.COLOR_K()
)


ax.plot(
    data["merge_3-Na_CEY"].x,
    data["merge_3-Na_CEY"].y,
    c=lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$ solution"
)

"""ax.plot(
    data["merge_3-Na_CEY"].x,
    data["merge_3-Na_CEY"].chir_real,
    c=clr.red,
    ls = ":"
)
ax.plot(
    data["merge_3-Na_CEY"].x,
    data["merge_3-Na_CEY"].chir_imaginary,
    c=clr.red,
    ls = "--"
)"""
ax_ins.plot(
    data["merge_3-Na_CEY"].x,
    data["merge_3-Na_CEY"].y,
    c=lib.COLOR_Na()
)

ax.legend()
pos_b = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax)
ax.text(
    pos_b.x,
    pos_b.y,
    "(c)",
)

plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "EXAFS_invered.png")

#%% XRD
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Final

from BFC_libs import common as cmn
from BFC_libs import colors as clr
from BFC_libs import XRD as xrd
#from BFC_libs.XRD import ICDD 
from BFC_libs.XRD import SmartLab as sl

try:
    from . import lib
except(ImportError):
    import lib

dir = "./data/XRD/20240229_2-115_MnO2_as_synth_after_electrolysis/"

x_min, x_max = xrd.Theta(8), xrd.Theta(69.959)
data_after_K: Final = sl.read_smart_lab_1D_data(
    dir + "20240229_2-115_006_after_K_fine_omega_100mdeg.txt"
)[0].slice(x_min, x_max)

data_after_Na: Final = sl.read_smart_lab_1D_data(
    dir + "20240229_2-115_006_after_Na_fine_omega_100mdeg.txt"
)[0].slice(x_min, x_max)


fig, ax = cmn.create_standard_matplt_canvas()

ax.set_xticks(np.linspace(10, 70, 7, dtype=int))
ax.set_xlim(8, 70)
ax.set_ylim(0, 1.4)
ax.set_xlabel(r"2$\theta$ (deg)")
ax.set_ylabel("Diffraction intensity (a.u.)")
ax.tick_params(labelleft = False, left = False, right = False)

ax.plot(
    data_after_K.x,
    data_after_K.y.normalize(),
    c=lib.COLOR_K(),
    label = "After electrolysis in K$^+$ solution"
)

ax.plot(
    data_after_Na.x,
    data_after_Na.y.normalize(),
    c=lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$ solution"
)


# 座標データは2-138_002_bare_FTO_fineの生データから手打ちで抜き出し
# 文字の幅の分だけ、x方向に約-0.6°
FTO_notation_coords: list[tuple[float, float]] = [
    (33.2, 0.3),
    (25.9, 0.25),
    (51.2, 0.5),
    (61.36, 0.1),
    (63.9, 0.08)
]

for point in FTO_notation_coords:
    ax.text(point[0], point[1], "*")

ax.legend()


plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "XRD.png")

#%% ラマン変化（test）
import numpy as np
import pandas as pd
import scipy.signal
import matplotlib.pyplot as plt

from enum import Enum

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.Raman as rmn
import BFC_libs.Raman.Nanophoton as nph
import BFC_libs.fitting_support as fit

data_file = nph.read_1D_data("data/Raman/20240207_2-107_Raman_bfr_aft_repro/20240207_2-107_comparison.txt")
diff_data_file = nph.read_1D_data("data/Raman/20240207_2-107_Raman_bfr_aft_repro/20240207_2-107_diff.txt")
class DataNameRaw(Enum):
#    K_before = 1
    K_after = 2
#    Na_before = 3
    Na_after = 4

class DataNameDiff(Enum):
    K_diff = 11
    Na_diff = 12
    before_diff = 13
    after_diff = 14


spectra:dict[DataNameRaw, rmn.RamanSpectrum] = {
#    DataNameRaw.K_before: data_file[0],
    DataNameRaw.K_after: data_file[2],
#    DataNameRaw.Na_before: data_file[3],
    DataNameRaw.Na_after: data_file[5]
}

diff_spectra:dict[DataNameDiff, rmn.RamanSpectrum] = {
    DataNameDiff.K_diff: diff_data_file[1],
    DataNameDiff.Na_diff: diff_data_file[0],
    DataNameDiff.before_diff: diff_data_file[3],
    DataNameDiff.after_diff: diff_data_file[2]
}


class MainPeak(Enum):
    mp580 = 0
    mp630 = 1

#スペクトル分割
#splitted_spectra:dict[(DataNameRaw, MainPeak), rmn.RamanSpectrum] = {}
#spectra_630:dict[DataNameRaw, rmn.RamanSpectrum] = {}

"""for data_series in DataNameRaw:
    
    splitted_spectra[(data_series, MainPeak.mp580)]\
        =spectra[data_series].slice(rmn.Wavenumber(560), rmn.Wavenumber(605))
    
    splitted_spectra[(data_series, MainPeak.mp630)]\
        =spectra[data_series].slice(rmn.Wavenumber(605), rmn.Wavenumber(670))
    """
# フィッティング準備
initial_fit_params = fit.TotalFunc(
    func_components=[
        fit.Linear(0.004),
        fit.Constant(-2),
        fit.LorentzPeak(
            _center = 579,
            _width = 15,
            _ampletude = 20
        ),
        fit.LorentzPeak(
            _center = 585,
            _width = 15,
            _ampletude = 20
        ),
        fit.LorentzPeak(
            _center = 635,
            _width = 21,
            _ampletude = 35
        ),
        fit.LorentzPeak(
            _center = 655,
            _width = 12,
            _ampletude = 5
        ),

    ],
    minimum_bounds=[
        fit.Linear(-0.01),
        fit.Constant(-10),
        fit.LorentzPeak(
            _center = 576,
            _width = 1,
            _ampletude = 2
        ),
        fit.LorentzPeak(
            _center = 581,
            _width = 1,
            _ampletude = 0.1
        ),
        fit.LorentzPeak(
            _center = 630,
            _width = 1,
            _ampletude = 2
        ),
        fit.LorentzPeak(
            _center = 650,
            _width = 1,
            _ampletude = 0.1
        ),

    ],
    maximum_bounds=[
        fit.Linear(0.01),
        fit.Constant(10),
        fit.LorentzPeak(
            _center = 580.9,
            _width = 50,
            _ampletude = 200
        ),
        fit.LorentzPeak(
            _center = 590,
            _width = 50,
            _ampletude = 200
        ),
        fit.LorentzPeak(
            _center = 640,
            _width = 50,
            _ampletude = 200
        ),
        fit.LorentzPeak(
            _center = 657,
            _width = 50,
            _ampletude = 200
        ),
        
    ]
)

fit_res: dict[DataNameRaw, fit.FittingResult] = {}

"""fit_patterns: list[tuple[DataNameRaw, MainPeak]] = [
    (DataNameRaw.K_after, MainPeak.mp580),
    (DataNameRaw.K_after, MainPeak.mp630),
    (DataNameRaw.Na_after, MainPeak.mp580),
    (DataNameRaw.Na_after, MainPeak.mp630)
]"""

for fit_pattern in DataNameRaw:
    fit_res[fit_pattern], _ = \
        fit.fitting(
            spectra[fit_pattern].x.float_array(),
            spectra[fit_pattern].y.float_array(),
            initial_fit_params
        )
residual: dict[DataNameRaw, rmn.RamanSpectrum] = {}
for fit_pattern in DataNameRaw:
    residual[fit_pattern] = \
        rmn.RamanSpectrum(
            _comment = "residual",
            _condition = "",
            _original_file_path = "",
            _data_name = "res_" + str(fit_pattern),
            _wavenumber = spectra[fit_pattern].x,
            _intensity = spectra[fit_pattern].y - \
                rmn.RammanIntensityArray(fit_res[fit_pattern].calc(spectra[fit_pattern].x.float_array()))
        )


def show_fit_res(spc: rmn.RamanSpectrum, fit_res: fit.FittingResult, ax: plt.Axes, y_offset=0.0):
    #spc = spectra_580[key]
    #res, cov = fit.fitting(spc.x, spc.y, fit_fucs)

    

    def plot_fitting(fit_res: fit.FittingResult, x: np.ndarray, ax: plt.Axes, y_offset = 0.0):


        funcs = fit_res.func_components

        # baselineプロット
        ax.plot(
            x, 
            funcs[0].calc(x)+funcs[1].calc(x) + y_offset,
            c="#AAAAAA"
            )

        #ピークのプロット
        for func in funcs[2:]:

            ax.plot(
                x, 
                funcs[0].calc(x)+funcs[1].calc(x) + func.calc(x)+ y_offset,
                c=clr.green
                )
        
        #フィッティング関数全体のプロット
        ax.plot(
            x, 
            fit_res.calc(x)+ y_offset,
            c=clr.red
            )
    plot_fitting(fit_res, spc.x.float_array(), ax, y_offset)

    return

Y_OFFSET = 1.0

fig = plt.figure(figsize=(6,4))
ax: plt.Axes = fig.add_axes([0.2, 0.2, 0.7, 0.7])
ax.set_xlim(555, 670)
#ax.set_ylim(0.0, 1.8)
ax.set_xlabel("Wavenumber (cm$^{-1}$))")
ax.set_ylabel("Raman intensity (a.u.)")

ax.plot(
        spectra[DataNameRaw.K_after].x,
        spectra[DataNameRaw.K_after].y.float_array(),
        c="black",
        lw = 2.5
    )
ax.plot(
        spectra[DataNameRaw.Na_after].x,
        spectra[DataNameRaw.Na_after].y.float_array() + Y_OFFSET,
        c="black",
        lw = 2.5
    )

class ElemPeak(Enum):
    ep578 = 0
    ep585 = 1
    ep638 = 2
    ep655 = 3

for fit_pattern in DataNameRaw:

    show_fit_res(
        spectra[fit_pattern], 
        fit_res[fit_pattern],
        ax, 
        0.0 if fit_pattern==DataNameRaw.K_after else Y_OFFSET
    )

peak_pos:dict[tuple[DataNameRaw, ElemPeak], cmn.Point] = {}
"""
for fit_pattern in DataNameRaw:
    sab_peak1 = fit_res[fit_pattern].func_components[2]
    sab_peak2 = fit_res[fit_pattern].func_components[3]
    
    if isinstance(sab_peak1, fit.PeakFunction):
        x = sab_peak1.center
        peak_pos[
            (ElemPeak.ep578 if fit_pattern == MainPeak.mp580 else ElemPeak.ep638)
        ] = (
            cmn.Point(
                x, 
                fit_res[fit_pattern].func_components[0].calc(x) + \
                    fit_res[fit_pattern].func_components[1].calc(x) + \
                    sab_peak1.calc(x) + \
                    (0.0 if fit_pattern==DataNameRaw.K_after else Y_OFFSET)
                )
        )
    else:
        raise TypeError
    
    if isinstance(sab_peak2, fit.PeakFunction):
        x = sab_peak2.center
        peak_pos[
            (ElemPeak.ep585 if fit_pattern == MainPeak.mp580 else ElemPeak.ep655)
        ] = (
            cmn.Point(
                x, 
                fit_res[fit_pattern].func_components[0].calc(x) + \
                    fit_res[fit_pattern].func_components[1].calc(x) + \
                    sab_peak2.calc(x) + \
                    (0.0 if fit_pattern==DataNameRaw.K_after else Y_OFFSET)
                )
        )
    else:
            raise TypeError"""


"""for peak in ElemPeak:
    x = [
        peak_pos[(DataNameRaw.K_after, peak)].x,
        peak_pos[(DataNameRaw.Na_after, peak)].x
    ]
    y = [
        peak_pos[(DataNameRaw.K_after, peak)].y,
        peak_pos[(DataNameRaw.Na_after, peak)].y
    ]

    ax.plot(x, y, c="black", ls = ":")"""


"""for fit_pattern in DataNameRaw:
    ax.plot(
        residual[fit_pattern].x,
        residual[fit_pattern].y + (rmn.RammanIntensity(Y_OFFSET) if fit_pattern == DataNameRaw.K_after else rmn.RammanIntensity(0))
        )
"""
#%% ラマン変化（差スペクトル）
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from copy import deepcopy as copy

from enum import Enum

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.Raman as rmn
import BFC_libs.Raman.Nanophoton as nph
import BFC_libs.fitting_support as fit
try:
    from . import lib
except(ImportError):
    import lib

data_file = nph.read_1D_data("data/Raman/20240207_2-107_Raman_bfr_aft_repro/20240207_2-107_comparison.txt")
diff_data_file = nph.read_1D_data("data/Raman/20240207_2-107_Raman_bfr_aft_repro/20240207_2-107_diff.txt")
class DataNameRaw(Enum):
    K_before = 1
    K_after = 2
    Na_before = 3
    Na_after = 4

class DataNameDiff(Enum):
    K_diff = 11
    Na_diff = 12
    before_diff = 13
    after_diff = 14


spectra:dict[DataNameRaw, rmn.RamanSpectrum] = {
    DataNameRaw.K_before: data_file[0],
    DataNameRaw.K_after: data_file[2],
    DataNameRaw.Na_before: data_file[3],
    DataNameRaw.Na_after: data_file[5]
}

diff_spectra:dict[DataNameDiff, rmn.RamanSpectrum] = {
    DataNameDiff.K_diff: diff_data_file[1],
    DataNameDiff.Na_diff: diff_data_file[0],
    DataNameDiff.before_diff: diff_data_file[3],
    DataNameDiff.after_diff: diff_data_file[2]
}
fig, ax = cmn.create_standard_matplt_canvas()
ax.set_xlim(100, 1100)
ax.set_ylim(0, 1.4)
ax.set_xlabel("Wavenumber (cm$^{-1}$)")
ax.set_ylabel("Intensity(a.u.)")


ax.plot(
    spectra[DataNameRaw.K_after].x,
    spectra[DataNameRaw.K_after].y,
    c = lib.COLOR_K(),
    label = "After electrolysis in K$^+$"
)
ax.plot(
    spectra[DataNameRaw.Na_after].x,
    spectra[DataNameRaw.Na_after].y,
    c = lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$"
)

# figをコピーして二種の似て非なるプロットを作成
#fig_diff = copy(fig)

ax_ins = ax.inset_axes((0.08, 0.4, 0.28, 0.35))

ax_ins.set_xlim(575, 585)
ax_ins.set_ylim(0.8, 1.05)

ax.indicate_inset_zoom(ax_ins, edgecolor="#000000")

ax_ins.tick_params(labelleft = False, left = False, right = False)

ax_ins2 = ax.inset_axes((0.65, 0.4, 0.28, 0.35))

ax_ins2.set_xlim(630, 645)
ax_ins2.set_ylim(0.9, 1.05)

ax.indicate_inset_zoom(ax_ins2, edgecolor="#000000")

ax_ins2.tick_params(labelleft = False, left = False, right = False)

ax_ins.plot(
    spectra[DataNameRaw.K_after].x,
    spectra[DataNameRaw.K_after].y,
    c = lib.COLOR_K()
)
ax_ins.plot(
    spectra[DataNameRaw.Na_after].x,
    spectra[DataNameRaw.Na_after].y,
    c = lib.COLOR_Na()
)
ax_ins2.plot(
    spectra[DataNameRaw.K_after].slice(rmn.Wavenumber(620), rmn.Wavenumber(1100)).x,
    spectra[DataNameRaw.K_after].slice(rmn.Wavenumber(620), rmn.Wavenumber(1100)).y.normalize(),
    c = lib.COLOR_K()
)
ax_ins2.plot(
    spectra[DataNameRaw.Na_after].slice(rmn.Wavenumber(620), rmn.Wavenumber(1100)).x,
    spectra[DataNameRaw.Na_after].slice(rmn.Wavenumber(620), rmn.Wavenumber(1100)).y.normalize(),
    c = lib.COLOR_Na()
)

ax.legend()
fig.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "Raman_change.png", dpi = 600)


#----------------------電解前比較----------------------
#ax_diff = fig_diff.axes[0]



"""ax.plot(
    diff_spectra[DataNameDiff.after_diff].x,
    diff_spectra[DataNameDiff.after_diff].y*10 + rmn.RammanIntensity(0.5),
    c = clr.purple
)"""

"""K_600 = spectra[DataNameRaw.K_after].slice(rmn.Wavenumber(600), rmn.Wavenumber(675))
Na_600 = spectra[DataNameRaw.Na_after].slice(rmn.Wavenumber(600), rmn.Wavenumber(675))
fig, ax = cmn.create_standard_matplt_canvas()

ax.plot(
    K_600.x,
    K_600.y.normalize(),
    c = lib.COLOR_K
)
ax.plot(
    Na_600.x,
    Na_600.y.normalize(),
    c = clr.red
)
ax.plot(
    K_600.x,
    (Na_600.y.normalize()[1:] - K_600.y.normalize()) * 10 + rmn.RammanIntensity(0.5),
    c = clr.purple
)"""

#%% フーリエ変換前のEXAFS
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

from BFC_libs import common as cmn
import BFC_libs.colors as clr

try:
    from . import lib
except(ImportError):
    import lib

column_E = "e"
column_kai_mu = "flat"
column_obsd = "xmu"

df_K = pd.read_csv(
    "./data/XAS/20240410_2-132_CEY_after_electrolysis/merge_4-K_CEY.csv"
    )

df_K_non_normd = pd.read_csv(
    "./data/XAS/20240410_2-132_CEY_after_electrolysis/merge_4-K_CEY_non_normalized.csv"
)

df_Na = pd.read_csv(
    "./data/XAS/20240410_2-132_CEY_after_electrolysis/merge_4-Na_CEY.csv"
)

df_Na_non_normd = pd.read_csv(
    "./data/XAS/20240410_2-132_CEY_after_electrolysis/merge_4-Na_CEY_non_normalized.csv"
)


fig,ax = cmn.create_standard_matplt_canvas()

ax.set_xlim(6350, 7300)

ax.set_xlabel("Energy (eV)")
ax.set_ylabel(r"Normalized $\chi\mu$ (a.u.)")

ax.plot(
    df_K[column_E],
    df_K[column_kai_mu],
    c= lib.COLOR_K(),
    label = "After electrolysis in K$^+$ solution"
)

ax.plot(
    df_Na[column_E],
    df_Na[column_kai_mu],
    c= lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$ solution"
)
ax.legend()

pos_a = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax)
ax.text(
    pos_a.x,
    pos_a.y,
    "(b)",
)

fig.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "_XAS.png", dpi = 600)

fig_non_normd, ax_non_normd = cmn.create_standard_matplt_canvas()

ax_non_normd.set_xlim(6350, 7300)

ax_non_normd.set_xlabel("Energy (eV)")
ax_non_normd.set_ylabel(r"$\chi\mu$ (a.u.)")

ax_non_normd.plot(
    df_K_non_normd[column_E],
    df_K_non_normd[column_obsd],
    c= lib.COLOR_K(),
    label = "After electrolysis in K$^+$ solution"
)

ax_non_normd.plot(
    df_Na_non_normd[column_E],
    df_Na_non_normd[column_obsd],
    c= lib.COLOR_Na(),
    label = "After electrolysis in Na$^+$ solution"
)
ax_non_normd.legend()

pos_a = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax_non_normd)
ax_non_normd.text(
    pos_a.x,
    pos_a.y,
    "(a)",
)

fig_non_normd.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "_XAS_non_normd.png", dpi = 600)
