import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
import numpy as np
import numpy.typing as nptyping
import pandas as pd

import BFC_libs.electrochemistry as ec
import BFC_libs.quantity_array as qa
import general_kinetics as kin
import common
from typing import Final

# TIME_EVOLUTION_OUTPUT_DIR = "./time_evolution/"
# TIME_EVOLUTION_FILE_FORMAT = "te_impedance_Ec_{potential_center}V_Ea{potential_amp}mV.csv"
# FIGURE_EXPORT_DIR = "../fig/"

MARKER_SIZE: Final[int] = 6
LINE_WIDTH: Final[float] = 0.75

SOLUTION_RESISTANCE: Final[float] = 10.0
DOUBLE_LAYER_CAPACITANCE: Final[float] = 1E-5
ACTIVESITES_NUM: Final[float] = 1E-9

step1 = kin.ElementaryStep_ET(
    common.ElectronTransferCoefficient(0.5),
    common.ReactionRateCoefficient(8E1),
    ec.Potential(-0.05),
    kin.Direction.ANODIC
)

step2 = kin.ElementaryStep_ET(
    common.ElectronTransferCoefficient(0.6),
    common.ReactionRateCoefficient(1E0),
    ec.Potential(0.05),
    kin.Direction.ANODIC
)

std_omega = ec.FrequencyArray(np.logspace(0, 6, 40)*2*np.pi)
"""ec.FrequencyArray(np.logspace(0, 6, 40) * 2 * np.pi)"""

def common_canvas_for_impedance_plot()->tuple[Figure, Axes, Axes, Axes]:
    """
    returns: fig, ax_Nyquist, ax_Bode_R, ax_Bode_ph
    """

    fig = plt.figure(figsize=(4, 8))
    ax_Nyquist = fig.add_axes((0.2, 0.6, 0.7, 0.35))
    ax_Nyquist.set_xlabel(r"$Z' [\mathrm{\Omega}]$")
    ax_Nyquist.set_ylabel(r"$-Z'' [\mathrm{\Omega}]$")

    ax_Bode_R = fig.add_axes((0.2,0.3,0.7,0.2))
    ax_Bode_ph = fig.add_axes((0.2,0.1,0.7,0.2))

    ax_Bode_R.tick_params(labelbottom=False)
    ax_Bode_R.set_xscale("log")
    ax_Bode_R.set_yscale("log")
    ax_Bode_ph.set_xscale("log")
    ax_Bode_ph.set_xlabel(r"$\omega$ [rad/s]")
    ax_Bode_ph.set_ylabel("Phase [degree]")
    ax_Bode_R.set_ylabel(r"$|~Z~|$ [$\mathrm{\Omega}$]")

    return fig, ax_Nyquist, ax_Bode_R, ax_Bode_ph


def add_non_Faraday_impedance(
    solution_resistance: float,
    double_layer_capaciternce: float,
    omega_array: np.ndarray,
    Faraday_impedance: np.ndarray
)->nptyping.NDArray[np.complex128]:
    return Faraday_impedance/(1+1j*omega_array*Faraday_impedance*double_layer_capaciternce)+solution_resistance

def load_time_evolved_EIS(
    file_pathes: list[str]
)->list[ec.EIS]:
    """
    file_pathのリストを受け取ったら、その順番の通りにEISの時間発展計算を読んで、list[ec.EIS]として返す。
    多次元には対応していない。
    エラーはキャッチしていないのでそのまま上がってくる。
    """
    tmp:list[ec.EIS] = []

    for file_path in file_pathes:
        df = pd.read_csv(file_path)
        tmp.append(ec.EIS.from_data_frame(df))

    return tmp


def list_EIS_to_2d_array(
    EIS_list: list[ec.EIS]
)->nptyping.NDArray:
    """
    listのEISを二次元np.NDArrayへと変換する関数。
    二時限目が周波数になる。
    """
    tmp = []
    for eis in EIS_list:
        tmp.append(eis.impedance.complex_array())
    return np.stack(tmp)
    tmp = EIS_list[0].impedance.complex_array()

    for eis in EIS_list[1:]:
        tmp = np.stack([tmp, eis.impedance.complex_array()])
    return tmp