#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import Final

from BFC_libs import common as cmn
from BFC_libs import colors as clr
from BFC_libs import electrochemistry as ec
from BFC_libs.electrochemistry import biologic as bl
from BFC_libs import fitting_support as fit

try:
    from . import lib
except(ImportError):
    import lib


data = bl.BiologicCAData.load_file("./data/EC/20240514_2-139_kinetics_test/20240514_2-139_kinetic_test_2_C01.mpt")

double_exponential = fit.TotalFunc(
    [fit.Exponential(
        tau = 3.0,
        amplitude = 5,
    ),
    fit.Exponential(
        tau = 5.0,
        amplitude = 1,
    ),
    fit.Constant(0.0)],

    [fit.Exponential(
        tau = 0.0,
        amplitude = 0,
    ),
    fit.Exponential(
        tau = 0.0,
        amplitude = 0,
    ),
    fit.Constant(-1.0)],

    [fit.Exponential(
        tau = 30.0,
        amplitude = 10,
    ),
    fit.Exponential(
        tau = 30.0,
        amplitude = 10,
    ),
    fit.Constant(6.0)],

)

single_exponential = fit.TotalFunc(
    [fit.Exponential(
        tau = 3.0,
        amplitude = 5,
    ),
    fit.Constant(0.0)],

    [fit.Exponential(
        tau = 0.0,
        amplitude = 0,
    ),
    fit.Constant(-1.0)],

    [fit.Exponential(
        tau = 30.0,
        amplitude = 10,
    ),
    fit.Constant(6.0)],

)



fig, ax = cmn.create_standard_matplt_canvas()
ax.set_xlim(89, 100)

res_array:list[fit.TotalFunc] = []
res_1exp_array:list[fit.TotalFunc] = []

potential_array: list[ec.Potential] = []
tau1_array: list[float] = []
tau2_array: list[float] = []

for i, ca in enumerate(data.data[1:]):
    if isinstance(ca, ec.ChronoAmperogram):
        res, cov = fit.fitting(
            ca.x - ca.x[0],
            ca.y,
            double_exponential)
        res_array.append(res)
        
        potential_array.append(ca.potential[1])
        tau1_array.append(res.func_components[0].tau)
        tau2_array.append(res.func_components[1].tau)

        res_single, cov_single = fit.fitting(
            ca.x - ca.x[0],
            ca.y,
            single_exponential)
        res_1exp_array.append(res_single)



        ax.plot(
            ca.x,
            ca.y,
            c = clr.red,
            lw = 0.5
        )
        """ax.plot(
            ca.x,
            res.calc(ca.x.float_array() - ca.x[0].value),
            c = clr.black
        )"""
        ax.plot(
            ca.x,
            res_single.calc(ca.x.float_array() - ca.x[0].value),
            c = clr.blue,
            ls = ":"
        )

fig, ax = cmn.create_standard_matplt_canvas()

ax.scatter(
    potential_array,
    tau1_array,
    label = "tau1"
    )

ax.scatter(
    potential_array,
    tau2_array,
    label = "tau2"
    )

ax.legend()