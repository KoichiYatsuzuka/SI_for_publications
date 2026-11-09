#%%
import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as integ
import scipy.fft as fft
from functools import partial
import matplotlib.cm as cm

from typing import Final

import BFC_libs.electrochemistry as ec
from BFC_libs.physical_constants import F

import general_kinetics as kin

import plot_lib
from common import TIME_EVOLUTION_FILE_FORMAT, TIME_EVOLUTION_OUTPUT_DIR, FIGURE_EXPORT_DIR
from plot_lib import SOLUTION_RESISTANCE, DOUBLE_LAYER_CAPACITANCE, ACTIVESITES_NUM
from plot_lib import step1, step2
from plot_lib import std_omega
from plot_lib import MARKER_SIZE, LINE_WIDTH

TIME_EVOLUTION_PERIOD: Final[float] = 10.0
TIME_EVOLUTION_POINT_NUM_PER_PERIOD: Final[int] = 3600
TOTAL_POINT_NUM: Final[int] = int(TIME_EVOLUTION_POINT_NUM_PER_PERIOD*TIME_EVOLUTION_PERIOD)
WARMUP_PERIOD: Final[float] = 5.0

POTENTIAL_AMP: Final[ec.Potential] = ec.Potential(0.01)

COLOUR_OURS:Final[tuple[float, float, float, float]] = cm.winter(0.2)
COLOUR_SCIPY: Final[tuple[float, float, float, float]] = cm.winter(0.8)

Z_OURS: Final[int] = -1
Z_SCIPY: Final[int] = 1

applied_potential = ec.PotentialArray([0.075])
omega = std_omega 

def d_theta_d_time_detailed(
        time_sec: np.ndarray, 
        theta: float, 
        potential_center: ec.Potential,
        potential_amp: ec.Potential,
        period_sec: float,
        stepA: kin.ElementaryStep_ET,
        stepB: kin.ElementaryStep_ET):
    
    angular_time = time_sec/period_sec*2*np.pi
    potential = potential_center + potential_amp*np.sin(angular_time)
    
    return stepA.rate_fw(potential)*(1-theta)\
            -stepA.rate_bw(potential)*theta\
            -stepB.rate_fw(potential)*theta\
            +stepB.rate_bw(potential)*(1-theta)

current_list:list[np.ndarray] = []
for i, E in enumerate(applied_potential):
    
    tmp_list = []
    
    potential:ec.PotentialArray = ec.PotentialArray(POTENTIAL_AMP.value*np.sin(np.linspace(0, TIME_EVOLUTION_PERIOD, TOTAL_POINT_NUM, endpoint=False)*2*np.pi))+E
    k1 = step1.rate_fw(potential)
    k1r = step1.rate_bw(potential)
    k2 = step2.rate_fw(potential)
    k2r = step2.rate_bw(potential)

    for j, o in enumerate(omega):
        
        freq: float = o.value / (2*np.pi)
        period = 1/freq

        d_theta_d_time = partial(
            d_theta_d_time_detailed, 
            potential_center = E, 
            potential_amp = POTENTIAL_AMP,
            period_sec = period, 
            stepA = step1,
            stepB = step2
            )
        
        time = np.linspace(0, TIME_EVOLUTION_PERIOD, TOTAL_POINT_NUM, endpoint=False)*period
        
        # 初期値用に、真の定常状態の時の被覆率計算
        theta_center_val \
            = (step1.rate_fw(E)+step2.rate_bw(E))\
                /(step1.rate_fw(E)+step1.rate_bw(E)+step2.rate_fw(E)+step2.rate_bw(E))        
        theta_center = np.full(1, theta_center_val)
        
        tmp:integ._ivp.ivp.OdeResult = integ.solve_ivp(
            fun = d_theta_d_time,
            t_span=(0, TIME_EVOLUTION_PERIOD*period),
            y0=theta_center,
            t_eval=time,
            vectorized=True,
            method="BDF"
        )
        
        theta = tmp.y[0]
        current = F*ACTIVESITES_NUM*(k1*(1-theta) + k2*theta - k1r*theta - k2r*(1-theta))
        
        tmp_list.append(current)
    current_list.append(np.array(tmp_list))


current_array = np.array(current_list) #一次元目: 電位, 二次元目: 周波数, 三時限目: 時間（周期で規格化）


fft_list: list[np.ndarray] = []
for current_each_E in current_array:
    tmp_list = []
    for current_each_E_o, o in zip(current_each_E, omega):
        freq = o.value/2/np.pi
        point_per_period = TOTAL_POINT_NUM//TIME_EVOLUTION_PERIOD
        skip_points = int(WARMUP_PERIOD * point_per_period)
        current_trimmed: np.ndarray = current_each_E_o[skip_points:]
        n_fft = len(current_trimmed)

        raw_fft = fft.fft(current_trimmed)

        dt = 1/point_per_period/freq # ポイント間の時間差
        deconvoluted_freq: np.ndarray = fft.fftfreq(n_fft, dt)

        # 第一次高調波成分とy-offset部分の抜き出しのためのマスク配列作成、負の数成分は削除して後で二倍
        mask1 = deconvoluted_freq < freq*1.01
        mask2 = deconvoluted_freq > freq*0.99
        mask3 = deconvoluted_freq == 0
        freq_mask = (mask1&mask2)|mask3

        # 生のFFTから、解析すべき部分の抜き出し
        extracted_fft: np.ndarray = raw_fft[freq_mask]

        amp: np.ndarray = extracted_fft/(n_fft/2) #電流の振幅絶対値+定数部分
        extracted_freq: np.ndarray = deconvoluted_freq[freq_mask] # 第一次高調波（と周りの成分）と定数部分(0 Hz)

        # 定数成分の除去（0番目が定数成分）
        freq_non_zero: np.ndarray = extracted_freq[1]
        amp_non_zero: np.ndarray = amp[1]

        amp_real = amp_non_zero.real
        amp_imag = amp_non_zero.imag

        # 理解がめんどくさいパート
        # FFTのreal = cos成分, imag = -sin成分
        # EISのreal = sin成分, imag = cos成分

        current_real = -1*amp_imag
        current_imag = amp_real

        abs_current_amp = np.sqrt(current_real**2 + current_imag**2)
        current_phase = np.arctan(current_imag/current_real)

        abs_impedance_val = POTENTIAL_AMP.value / abs_current_amp

        tmp_list.append(
            abs_impedance_val*np.cos(current_phase) - 1J*abs_impedance_val*np.sin(current_phase)
        )
    fft_list.append(tmp_list)
fft_array = np.array(fft_list) #一次元目: 電位, 二次元目: 周波数

path_list:list[str] = []
for e in applied_potential:
    path_list.append(TIME_EVOLUTION_OUTPUT_DIR+TIME_EVOLUTION_FILE_FORMAT.format(
        E1 = -0.05,
        potential_center = e.value,
        potential_amp = POTENTIAL_AMP.value*1000
        )
    )
time_evolved_EIS_exp = plot_lib.load_time_evolved_EIS(path_list)
time_evolved_impedance_exp = plot_lib.list_EIS_to_2d_array(time_evolved_EIS_exp)

full_impedance_exp = plot_lib.add_non_Faraday_impedance(
    SOLUTION_RESISTANCE,
    DOUBLE_LAYER_CAPACITANCE,
    omega,
    time_evolved_impedance_exp
)

full_impedance_scipy = plot_lib.add_non_Faraday_impedance(
    SOLUTION_RESISTANCE,
    DOUBLE_LAYER_CAPACITANCE,
    omega,
    fft_array
)


for i in range (0, len(full_impedance_scipy)):
    fig, ax_Nyquist, ax_Bode_R, ax_Bode_ph = plot_lib.common_canvas_for_impedance_plot()
    # fig = plt.figure(figsize = (4,2))
    # ax = fig.add_axes((0.2,0.2,0.7,0.7))

    # min_r = min([np.min(full_impedance_scipy[i].real), np.min(full_impedance_scipy[i].imag)])
    # max_r = max([np.max(full_impedance_scipy[i].real), np.max(full_impedance_scipy[i].imag)])
    # delta_r = (max_r - min_r)/2
    # plot_range = max_r + delta_r*0.2
    # ax.set_xlim(0 - plot_range*0.1, plot_range)
    # ax.set_ylim((0 - plot_range*0.1)*0.5, plot_range*0.5)

    # ax.set_xlabel(r"$Z' [\mathrm{\Omega}]$")
    # ax.set_ylabel(r"$-Z'' [\mathrm{\Omega}]$")

    ax_Nyquist.scatter(
        full_impedance_exp[i].real, 
        -full_impedance_exp[i].imag, 
        #c="#005B82", 
        color = COLOUR_OURS,
        label="This study",
        s = MARKER_SIZE,
        zorder = Z_OURS
    )
    ax_Nyquist.plot(
        full_impedance_exp[i].real, 
        -full_impedance_exp[i].imag, 
        #c="#005B82", 
        color = COLOUR_OURS,
        lw = LINE_WIDTH,
        zorder = Z_OURS
    )

    ax_Nyquist.scatter(
        full_impedance_scipy[i].real, 
        -full_impedance_scipy[i].imag, 
        #c="#D51506",  
        color = COLOUR_SCIPY,
        label="SciPy",
        s = MARKER_SIZE,
        zorder = Z_SCIPY
    )
    ax_Nyquist.plot(
        full_impedance_scipy[i].real, 
        -full_impedance_scipy[i].imag, 
        #c="#D51506",  
        color = COLOUR_SCIPY,
        lw = LINE_WIDTH,
        zorder = Z_SCIPY
    )


    ax_Nyquist.set_xlim(
        -5, 
        ax_Nyquist.get_xlim()[1]
    )

    ax_Nyquist.set_ylim(
        -5,
        ax_Nyquist.get_xlim()[1]
    )
    #ax_Nyquist.legend()

    ax_Bode_R.scatter(
        omega,
        np.abs(full_impedance_exp[i]),
        #c="#005B82",
        color = COLOUR_OURS,
        label="This study",
        s = MARKER_SIZE,
        zorder = Z_OURS
    )
    ax_Bode_R.plot(
        omega,
        np.abs(full_impedance_exp[i]),
        #c="#005B82",
        color = COLOUR_OURS,
        lw = LINE_WIDTH,
        zorder = Z_OURS
    )


    ax_Bode_R.scatter(
        omega,
        np.abs(full_impedance_scipy[i]),
        #c="#D51506",
        color = COLOUR_SCIPY,
        label="SciPy",
        s = MARKER_SIZE,
        zorder = Z_SCIPY
    )
    ax_Bode_R.plot(
        omega,
        np.abs(full_impedance_scipy[i]),
        #c="#D51506",
        color = COLOUR_SCIPY,
        #label="SciPy",
        lw = LINE_WIDTH,
        zorder = Z_SCIPY
    )

    ax_Bode_ph.scatter(
        omega,
        np.angle(full_impedance_exp[i])/np.pi*180,
        #c="#005B82",
        color = COLOUR_OURS,
        label="This study",
        s = MARKER_SIZE,
        zorder = Z_OURS
    )
    ax_Bode_ph.plot(
        omega,
        np.angle(full_impedance_exp[i])/np.pi*180,
        #c="#005B82",
        color = COLOUR_OURS,
        #label="Our method",
        lw = LINE_WIDTH,
        zorder = Z_OURS
    )

    ax_Bode_ph.scatter(
        omega,
        np.angle(full_impedance_scipy[i])/np.pi*180,
        #c="#D51506",
        color = COLOUR_SCIPY,
        label="SciPy",
        s = MARKER_SIZE,
        zorder = Z_SCIPY
    )
    ax_Bode_ph.plot(
        omega,
        np.angle(full_impedance_scipy[i])/np.pi*180,
        #c="#D51506",
        color = COLOUR_SCIPY,
        #label="SciPy",
        lw = LINE_WIDTH,
        zorder = Z_SCIPY
    )
    ax_Nyquist.text(-0.2, 1.0, "(A)", transform = ax_Nyquist.transAxes)
    ax_Bode_R.text(-0.2, 1.0, "(B)", transform = ax_Bode_R.transAxes)

    ax_Nyquist.legend()
    ax_Bode_R.legend()

    fig.savefig(
        FIGURE_EXPORT_DIR+"scipy_time_evolution_comparison_E={}V_Amp={}mV.png".format(applied_potential[i].value, POTENTIAL_AMP.value*1000),
        dpi=600
    )


# %%