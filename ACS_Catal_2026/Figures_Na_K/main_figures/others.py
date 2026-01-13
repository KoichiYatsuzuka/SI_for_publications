#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import os

from BFC_libs import common as cmn
import BFC_libs.colors as clr
import BFC_libs.Raman as rmn
import BFC_libs.Raman.Nanophoton as nph
import BFC_libs.fitting_support as fit

Si_fit_data = nph.read_peakfit_result(
    "data/Raman/20230927_2-56_001_silicon_after_calibration_once_fitting.txt"
)

positions: np.ndarray = Si_fit_data.data[:, 0, 1]

numbering = np.linspace(1, len(positions), len(positions))

fig, ax = cmn.create_standard_matplt_canvas()


ax.set_xlabel("Measurement times")
ax.set_ylabel("Peak top position (cm$^{-1}$)")
ax.scatter(numbering, positions, c="#444")

print(positions.std())

plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "Nanophoton_deviation.png", dpi=600)