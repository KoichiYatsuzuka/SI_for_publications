#%%alpha-MnO2のサイクル特性
"""
このコードでαとγの両方のデータを出力する。

"""
import os
from typing import Final, Optional

import numpy as np
#import pandas as pd
#import scipy.signal
import matplotlib.pyplot as plt

import BFC_libs.colors as clr
import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec
import BFC_libs.electrochemistry.biologic as bl
from BFC_libs.electrochemistry import Resistance

try:
    from . import lib
except(ImportError):
    import lib

ELECTRODE_GEO_AREA_WIDE: Final[float] = 0.4 * 0.4 * np.pi #直径が1 cmだとずっと思いこんでいたが、8 mmだと気づいた。定性的な結果に影響なし
ELECTRODE_GEO_AREA_NARROW: Final[float] = 0.25 * 0.25 * np.pi #追加実験用。8 mm未使用パンチが無かったので代替

OVERPOTENTIAL_STD_CURRENT_DENSITY: Final[ec.Current] = ec.Current(0.0005) # 0.5 mA/cm2
OVERPOTENTIAL_STD_CURRENT_DENSITY_LOW: Final[ec.Current] = ec.Current(0.0003) # 0.3 mA/cm2

#OVERPOTENTIAL_STD_CURRENT: Final[ec.Current] = ec.Current(0.0005)*ELECTRODE_GEO_AREA_WIDE # 0.5 mA/cm2

#TAFEL_SLOPE_STD_POTENTIAL: Final[ec.Potential] = ec.Potential(0.8) # V vs SHE

#TAFEL_SLOPE_CALCULATION_WIDTH: Final[ec.Potential] = ec.Potential(0.05) # 50 mV

BACKGROUND_COLOUR_K = lib.COLOR_K
BACKGROUND_COLOUR_Na = lib.COLOR_Na
BACKGROUND_TRANSMITION: Final[float] = 0.25

#alphaのデータ読み込み
#ボルタモグラムデータ読み込みしつつiR補正
# voltammograms_alpha: dict[str, bl.BioLogicVoltammogramData] = {}
# for key in keys:
#     resistance = Resistance(
#         bl.BiologicEISData.load_file(dir_alpha + file_base_alpha.format(*file_identifiers_alpha[key], "02_PEIS")).data[0].get_resistance()
#     )
#     voltammograms_alpha[key] = \
#         bl.load_biologic_CV(dir_alpha + file_base_alpha.format(*file_identifiers_alpha[key], "03_CVA"))\
#             .iR_correction(resistance)\
#             .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)
# del resistance

#gammaのデータ読み込み
#gammaは抵抗値を手入力（Im(z) = 0を通らないため）
resistances_gamma: dict[str, Resistance] = {
    "K_1": Resistance(43.1),
    "Na_1": Resistance(25.6),
    "K_2": Resistance(25.6),
    "Na_2": Resistance(25.8)
}
# voltammograms_gamma: dict[str, bl.BioLogicVoltammogramData] = {}
# for key in keys:
#     voltammograms_gamma[key] = \
#     bl.load_biologic_CV(dir_gamma + file_base_gamma.format(*file_identifiers_gamma[key], "03_CVA"))\
#         .iR_correction(resistances[key])\
#         .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)
# del resistances

#---------------------------------データ読み込み---------------------------------

def cv_analysis(file_path_base: str, #ファイルパス、format文字列
                file_identifiers: dict[str, tuple[str, ...]], #format文字列に入れる文字列
                keys: list[str], #解析するkeyの順番
                geo_surface_area: float, #電極の面積
                std_current_density: ec.Current = OVERPOTENTIAL_STD_CURRENT_DENSITY,
                resistances: Optional[dict[str, ec.Resistance]] = None #EISから抵抗値が自動的に算出されない場合、手打ち抵抗値
                ):
    #---------------------------------データ解析---------------------------------
    raw_voltammograms: dict[str, bl.BioLogicVoltammogramData] = {}
    iR_corrected_voltammograms: dict[str, bl.BioLogicVoltammogramData] = {}

    #---------------------------------生データ読み込み---------------------------------
    for key in keys:
        raw_voltammograms[key] = \
            bl.load_biologic_CV(file_path_base.format(*file_identifiers[key], "03_CVA"))\
                .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)

    #---------------------------------iR補正---------------------------------
    #---------------------------------手打ち抵抗値がないなら、抵抗値読み込み---------------------------------
    if resistances is None:
        resistances :dict[str, ec.Resistance] = {}
        for key in keys:
            resistances[key] = bl.BiologicEISData.load_file(file_path_base.format(*file_identifiers[key], "02_PEIS")).data[0].get_resistance()
    
    for key in keys:
        iR_corrected_voltammograms[key] = \
            raw_voltammograms[key].iR_correction(resistances[key])

    #tafel_slope_begin = ec.Potential(0.58) + (ec.SSCE - ec.SHE)
    #tafel_slope_end = ec.Potential(0.62) + (ec.SSCE - ec.SHE)

    overpotential_list:list[ec.Potential] = []
    tafel_slope_list: list[float] = []

    for key in keys:
        for i, cv in enumerate(iR_corrected_voltammograms[key].data):
            if isinstance(cv, ec.Voltammogram):
                

                op_index = cv.current.find(std_current_density*geo_surface_area)[0]
                ts_index = (
                    #cv.potential.find(tafel_slope_begin)[0],
                    #cv.potential.find(tafel_slope_end)[0]
                    op_index - 25,
                    op_index + 25
                )

                # TS calculation
                delta_i = (cv.current[ts_index[1]].log10() - cv.current[ts_index[0]].log10())
                delta_E = (cv.potential[ts_index[1]] - cv.potential[ts_index[0]])

                #append
                overpotential_list.append(cv.potential[op_index])
                tafel_slope_list.append((float(delta_E)/float(delta_i))*1000) # V/dec -> mV/dec

    return (iR_corrected_voltammograms, overpotential_list, tafel_slope_list)
    

def make_op_ts_profile(
    voltammograms: dict[str, bl.BioLogicVoltammogramData], #ボルタモグラムデータ辞書,
    geo_surface_area: float, #電極の面積,
    overpotential_list: list[ec.Potential], #過電位リスト,
    tafel_slope_list: list[float], #Tafel slopeリスト,
    fig_suffix: str, #出力フォルダ接尾文字列,
):
    keys = list(voltammograms.keys())
    #---------------------------------描画設定---------------------------------
    # Kは青、Naは赤
    color_array = [
        lib.COLOR_K, #blue
        lib.COLOR_K, #blue
        lib.COLOR_Na, #red
        lib.COLOR_K, #blue
        lib.COLOR_Na, #red
    ]
    
    # 以降、旧コード
    
    #軸設定
    fig = plt.figure(figsize = (8,9))
    plt.subplots_adjust(wspace=0.3, hspace=0.4)
    text_pos = lib.figure_alphabet_pos
    fig_id_dict: dict[str, str] = {
        "K_1": "(a)",
        "Na_1": "(b)",
        "K_2": "(c)",
        "Na_2": "(d)",
    }

    #(a) - (d)
    ax_dict:dict[str, plt.Axes] = {}

    cv_x_lim_min = 0.65
    cv_x_lim_max = 0.95
    #cv_y_lim_min = -0.2 if fig_suffix == "_alpha" else -0.1
    #cv_y_lim_max = 6 if fig_suffix == "_alpha" else 2.5
    cv_y_lim_min = -0.2
    cv_y_lim_max = 6

    if fig_suffix == "_alpha_reproducibility":
        cv_y_lim_max = 10
    
    
    for i,key in enumerate(keys):
        ax_dict[key] = fig.add_subplot(3,2, i+1)
        ax_dict[key].set_xlabel("Potential (V vs SHE)")
        ax_dict[key].set_ylabel("Current density (mA/cm$^2$)")
        ax_dict[key].set_xlim(cv_x_lim_min, cv_x_lim_max)
        ax_dict[key].set_ylim(cv_y_lim_min, cv_y_lim_max)
        ax_dict[key].text(
            text_pos.x*(cv_x_lim_max - cv_x_lim_min) + cv_x_lim_min,
            text_pos.y*(cv_y_lim_max - cv_y_lim_min) + cv_y_lim_min,
            fig_id_dict[key]
            )

    # (e) - (f)
    
    # 図中に"in K+"のように入れる注釈（in 0.0 - 1.0 scale）
    match fig_suffix:
        case "_alpha":

            annotation_pos_op = [
                (0.05, 0.85), #alpha
                (0.3, 0.08),
                (0.55, 0.85),
                (0.8, 0.08),
            ]
            annotation_pos_ts = [
                (0.05, 0.85), #alpha
                (0.3, 0.08),
                (0.55, 0.85),
                (0.8, 0.08),
            ]
        case "_gamma":
            annotation_pos_op =[
                (0.05, 0.85), #gamma
                (0.3, 0.08),
                (0.55, 0.08),
                (0.8, 0.08),
            ]
            annotation_pos_ts = [
                (0.05, 0.85), #gamma
                (0.3, 0.85),
                (0.55, 0.08),
                (0.8, 0.08),
            ]

        case "_alpha_reproducibility":
            annotation_pos_op = [
                (0.05, 0.85), #alpha
                (0.3, 0.08),
                (0.55, 0.85),
                (0.8, 0.08),
            ]
            annotation_pos_ts = [
                (0.05, 0.85), #alpha
                (0.3, 0.08),
                (0.55, 0.85),
                (0.8, 0.08),
            ]
        case _:
            annotation_pos_ts = []
            annotation_pos_op = []

    # (e): overpotential
    ax_op = fig.add_subplot(3,2,5)
    #op_low_edge = float(min(overpotential_list)) - 0.01
    #op_high_edge = float(max(overpotential_list)) + 0.01
    op_low_edge = 0.79
    op_high_edge = 0.87
    ax_op.set_xlim(0, 60)
    ax_op.set_ylim(op_low_edge, op_high_edge)
    ax_op.axvspan(0, 15, color = BACKGROUND_COLOUR_K(), alpha=BACKGROUND_TRANSMITION)
    ax_op.axvspan(15, 30, color = BACKGROUND_COLOUR_Na(), alpha=BACKGROUND_TRANSMITION)
    ax_op.axvspan(30, 45, color = BACKGROUND_COLOUR_K(), alpha=BACKGROUND_TRANSMITION)
    ax_op.axvspan(45, 60, color = BACKGROUND_COLOUR_Na(), alpha=BACKGROUND_TRANSMITION)
    ax_op.set_xlabel("Cycles")
    ax_op.set_ylabel(r"$E_{0.5}$ (V vs SHE)")
    ax_op.text(
        text_pos.x * (len(overpotential_list) + 1),
        text_pos.y * float(op_high_edge - op_low_edge) + float(op_low_edge),
        "(e)"
    )
    for i, pos in enumerate(annotation_pos_op):
        ax_op.text(
            pos[0] * (len(overpotential_list) + 1),
            pos[1] * float(op_high_edge - op_low_edge) + float(op_low_edge),
            "in K$^+$" if i%2==0 else "in Na$^+$"
        )

    # (f): Tafel slope
    ax_ts = fig.add_subplot(3,2,6)
    #ts_high_edge = float(max(tafel_slope_list)) + 2
    #ts_low_edge = float(min(tafel_slope_list)) - 2
    ts_high_edge = 80
    ts_low_edge = 40
    ax_ts.set_xlim(0, 60)
    ax_ts.set_ylim(ts_low_edge, ts_high_edge)
    #ax_ts.set_ylim(40, 80)
    ax_ts.axvspan(0, 15, color = BACKGROUND_COLOUR_K(), alpha=BACKGROUND_TRANSMITION)
    ax_ts.axvspan(15, 30, color = BACKGROUND_COLOUR_Na(), alpha=BACKGROUND_TRANSMITION)
    ax_ts.axvspan(30, 45, color = BACKGROUND_COLOUR_K(), alpha=BACKGROUND_TRANSMITION)
    ax_ts.axvspan(45, 60, color = BACKGROUND_COLOUR_Na(), alpha=BACKGROUND_TRANSMITION)
    ax_ts.set_xlabel("Cycles")
    ax_ts.set_ylabel(r"TS$_{0.5}$ (mV/dec)")
    ax_ts.text(
        text_pos.x * (len(tafel_slope_list) + 1),
        text_pos.y * float(ts_high_edge - ts_low_edge) + float(ts_low_edge),
        "(f)"
    )
    for i, pos in enumerate(annotation_pos_ts):
        ax_ts.text(
            pos[0] * (len(overpotential_list) + 1),
            pos[1] * float(ts_high_edge - ts_low_edge) + float(ts_low_edge),
            "in K$^+$" if i%2==0 else "in Na$^+$"
        )

    #---------------------------------描画---------------------------------

    current_multipilication = 1000 / (geo_surface_area) #A -> mA/cm2

    
    #溶液入れ替え前のボルタモグラム描画
    ax_dict["Na_1"].plot(
        voltammograms["K_1"].data[-1].potential.float_array(),
        voltammograms["K_1"].data[-1].current.float_array() * current_multipilication,
        c = color_array[1](),
        ls = ":"
    )
    ax_dict["K_2"].plot(
        voltammograms["Na_1"].data[-1].potential.float_array(),
        voltammograms["Na_1"].data[-1].current.float_array() * current_multipilication,
        c = color_array[2](),
        ls = ":"
    )
    ax_dict["Na_2"].plot(
        voltammograms["K_2"].data[-1].potential.float_array(),
        voltammograms["K_2"].data[-1].current.float_array() * current_multipilication,
        c = color_array[3](),
        ls = ":"
    )

    #メインの描画
    # (a) - (d)
    for i, key in enumerate(keys):
        #まず0.5 mA/cm2を表す破線
        ax_dict[key].hlines(0.5, 0.0, 1.0, linestyles="dashed", colors= "#AAAAAA")


        for j, cv in enumerate(voltammograms[key].data):
            
            #最初と最後だけ太め、それ以外は細め
            if j == 0 or j == len(voltammograms[key].data)-1:
                line_width = 1.5
            else:
                line_width = 0.5

            ax_dict[key].plot(
                cv.potential.float_array(),
                cv.current.float_array() * current_multipilication,
                c = clr.color_map(j/len(voltammograms[key].data), color_array[i], color_array[i+1]),
                lw = line_width
            )

                
    
    # サイクル番号の挿入
    match fig_suffix:
        case "_alpha":
            arrow_kargs = dict(arrowstyle = "-|>")
            # (a)
            ax_dict["K_1"].text(0.83, 4.7, "1st")
            ax_dict["K_1"].text(0.87, 3.3, "15th")
            ax_dict["K_1"].annotate("", (0.875, 3.8), (0.86, 4.6), arrowprops=arrow_kargs)
            
            # (b)
            ax_dict["Na_1"].text(0.83, 4.2, "15th")
            ax_dict["Na_1"].text(0.86, 2.5, "16th")
            ax_dict["Na_1"].text(0.89, 1.3, "30th")
            ax_dict["Na_1"].annotate("", (0.895, 1.65), (0.885, 2.4), arrowprops = arrow_kargs)

            # (c)
            ax_dict["K_2"].text(0.87, 2, "30th")
            ax_dict["K_2"].text(0.825, 4, "31st-45th")
            
            # (d)
            ax_dict["Na_2"].text(0.84, 4, "45th")
            ax_dict["Na_2"].text(0.86, 2.2, "46th-60th")
        
        case "_gamma":
            arrow_kargs = dict(arrowstyle = "-|>")
            # (a)
            ax_dict["K_1"].text(0.85, 2.2, "1st")
            ax_dict["K_1"].text(0.88, 1.4, "15th")
            ax_dict["K_1"].annotate("", (0.88, 1.5), (0.865, 2.1), arrowprops=arrow_kargs)
            
            # (b)
            ax_dict["Na_1"].text(0.84, 1.5, "15th")
            ax_dict["Na_1"].text(0.89, 1.4, "16th-\n30th")
            #ax_dict["Na_1"].text(0.89, 1.3, "30th")
            #ax_dict["Na_1"].annotate("", (0.895, 1.65), (0.885, 2.4), arrowprops = arrow_kargs)

            # (c)
            ax_dict["K_2"].text(0.895, 1.3, "30th")
            ax_dict["K_2"].text(0.84, 1.9, "31st-45th")
            
            # (d)
            ax_dict["Na_2"].text(0.83, 1.3, "45th")
            ax_dict["Na_2"].text(0.89, 1.0, "46th")
            ax_dict["Na_2"].text(0.86, 2.1, "60th")
            ax_dict["Na_2"].annotate("", (0.89, 1.95), (0.895, 1.35), arrowprops=arrow_kargs)
            
        case "_alpha_reproducibility":
            arrow_kargs = dict(arrowstyle = "-|>")
            # (a)
            ax_dict["K_1"].text(0.85, 9.2, "1st")
            ax_dict["K_1"].text(0.9, 7.2, "15th")
            ax_dict["K_1"].annotate("", (0.9, 8), (0.89, 10), arrowprops=arrow_kargs)
            
            # (b)
            ax_dict["Na_1"].text(0.9, 9, "15th")
            ax_dict["Na_1"].text(0.905, 3, "16th")
            ax_dict["Na_1"].text(0.91, 1.5, "30th")
            ax_dict["Na_1"].annotate("", (0.905, 2), (0.9, 3), arrowprops = arrow_kargs)

            # (c)
            ax_dict["K_2"].text(0.9, 2, "30th")
            ax_dict["K_2"].text(0.88, 9, "31st")
            ax_dict["K_2"].text(0.9, 6.5, "45th")
            ax_dict["K_2"].annotate("", (0.9, 7), (0.895, 8.3), arrowprops = arrow_kargs)
            
            # (d)
            ax_dict["Na_2"].text(0.87, 7.5, "45th")
            ax_dict["Na_2"].text(0.9, 2, "46th-\n60th")
        
    # (e) - (f)
    cycles = np.linspace(1, len(overpotential_list), len(overpotential_list))
    ax_op.scatter(cycles, overpotential_list, s= 3, c=clr.black)
    ax_op.plot(cycles, overpotential_list, c=clr.black, lw=1)

    ax_ts.scatter(cycles, tafel_slope_list, s=3, c=clr.black)
    ax_ts.plot(cycles, tafel_slope_list, lw=1, c=clr.black)


    plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + fig_suffix + "_CV_tracking.png", dpi = 600)
    
def op_ts_comparison(
    op1: list[ec.Potential],
    ts1: list[float],
    label1: str,
    op2: list[ec.Potential],
    ts2: list[float],
    label2: str,
    fig_suffix: str,
):
    #軸設定
    fig = plt.figure(figsize = (8,3))
    #plt.subplots_adjust(wspace=0.3, hspace=0.2)
    #text_pos: cmn.Point = lib.figure_alphabet_pos
    ax_op = fig.add_axes((0.1, 0.2, 0.35, 0.7))
    ax_ts = fig.add_axes((0.6, 0.2, 0.35, 0.7))

    # x軸: cycles
    cycles = np.linspace(1, len(op1), len(op1))

    ax_op.scatter(cycles, op1, s= 3, c=clr.black)
    ax_op.plot(cycles, op1, c=clr.black, lw=1, label= label1)
    ax_op.scatter(cycles, op2, s= 3, c=clr.red)
    ax_op.plot(cycles, op2, c=clr.red, lw=1, label= label2)

    ax_op.set_xlim(0, 60)
    ax_op.set_ylim(0.79, 0.9)
    ax_op.set_xlabel("Cycles")
    pos_a = \
        cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos_no_label, ax_op)
    ax_op.text(pos_a.x, pos_a.y, "(a)")
    ax_op.legend()

    ax_ts.scatter(cycles, ts1, s=3, c=clr.black)
    ax_ts.plot(cycles, ts1, lw=1, c=clr.black, label= label1)
    ax_ts.scatter(cycles, ts2, s=3, c=clr.red)
    ax_ts.plot(cycles, ts2, lw=1, c=clr.red, label= label2)
    
    ax_ts.set_xlim(0, 60)
    ax_ts.set_ylim(40, 80)
    pos_b = \
        cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos_no_label, ax_ts)
    
    ax_ts.text(pos_b.x, pos_b.y, "(b)")
    ax_ts.set_xlabel("Cycles")

    ax_ts.legend()
    
    return fig
    #plt.savefig(\)
    


#---------------------------------パラメータ設定---------------------------------

#描画する順番
normal_keys: Final[list[str]] = ["K_1", "Na_1", "K_2", "Na_2"]
keys_Na_first: Final[list[str]] = ["Na_1", "K_1", "Na_2", "K_2"]


#---------------------------------file path base---------------------------------
dir_alpha = "./data/EC/20240401_2-128_CV_cycles_alpha_gamma/alpha/"
dir_gamma = "./data/EC/20240401_2-128_CV_cycles_alpha_gamma/gamma/"

dir_addtional_exp = "./data/EC/20251112_2-148_alpha_reproducibility/"#revision追加実験のフォルダ

file_base_alpha = "20240328_2-128_{}_alpha_{}_{}_C01.mpt"
file_base_gamma = "20240328_2-128_{}_gamma_{}_{}_C02.mpt"

# revision実験用のファイルbase
file_base_alpha_additional_C1 = "20251112_2-148_{}_alpha_reproducibility_{}_{}_{}_C01.mpt"
file_base_alpha_additional_C2 = "20251112_2-148_{}_alpha_reproducibility_{}_{}_{}_C02.mpt"
file_base_gamma_additional_C1 = "20251112_2-148_{}_gamma_reproducibility_{}_{}_C01.mpt"
file_base_gamma_additional_C2 = "20251112_2-148_{}_gamma_reproducibility_{}_{}_{}_C02.mpt"
file_base_Na_first_C1 = "20251112_2-148_{}_alpha_Na_first_{}_{}_{}_C01.mpt"
file_base_Na_first_C2 = "20251112_2-148_{}_Na_first_{}_{}_{}_C02.mpt"


#---------------------------------ファイル名用---------------------------------
file_identifiers_alpha = {
    "K_1": ("001", "K-1"),
    "Na_1": ("003", "Na-1"),
    "K_2": ("005", "K-2"),
    "Na_2": ("007", "Na-2"),
}
file_identifiers_gamma = {
    "K_1": ("002", "K-1"),
    "Na_1": ("004", "Na-1"),
    "K_2": ("006", "K-2"),
    "Na_2": ("008", "Na-2"),
}

file_identifiers_alpha_additional_1 = {
    "K_1": ("001", "1", "K1"),
    "Na_1": ("001", "1", "Na1"),
    "K_2": ("001", "1", "K2"),
    "Na_2": ("001", "1", "Na2"),
}
file_identifiers_alpha_additional_2 = {
    "K_1": ("002", "2", "K1"),
    "Na_1": ("002", "2", "Na1"),
    "K_2": ("002", "2", "K2"),
    "Na_2": ("002", "2", "Na2"),
}

file_identifiers_gamma_additional_1 = {
    "K_1": ("005", "K1"),
    "Na_1": ("005", "Na1"),
    "K_2": ("005", "K2"),
    "Na_2": ("005", "Na2"),
}
file_identifiers_gamma_additional_2 = {
    "K_1": ("006", "2", "K1"),
    "Na_1": ("006", "2", "Na1"),
    "K_2": ("006", "2", "K2"),
    "Na_2": ("006", "2", "Na2"),
}
file_identifiers_Na_first_1 = {
    "Na_1": ("003", "1", "Na1"),
    "K_1": ("003", "1", "K1"),
    "Na_2": ("003", "1", "Na2"),
    "K_2": ("003", "1", "K2"),
}
file_identifiers_Na_first_2 = {
    "Na_1": ("004", "2", "Na1"),
    "K_1": ("004", "2", "K1"),
    "Na_2": ("004", "2", "Na2"),
    "K_2": ("004", "2", "K2"),
}

#----------------------------------解析実行---------------------------------
cv_alpha, op_alpha, ts_alpha = cv_analysis(
    dir_alpha + file_base_alpha,
    file_identifiers_alpha,
    normal_keys,
    ELECTRODE_GEO_AREA_WIDE,
    OVERPOTENTIAL_STD_CURRENT_DENSITY,
    None
)
cv_alpha_low, op_alpha_low, ts_alpha_low = cv_analysis(
    dir_alpha + file_base_alpha,
    file_identifiers_alpha,
    normal_keys,
    ELECTRODE_GEO_AREA_WIDE,
    OVERPOTENTIAL_STD_CURRENT_DENSITY_LOW,
    None
)

resistances_gamma: dict[str, Resistance] = {
    "K_1": Resistance(43.1),
    "Na_1": Resistance(25.6),
    "K_2": Resistance(25.6),
    "Na_2": Resistance(25.8)
}
cv_gamma, op_gamma, ts_gamma = cv_analysis(
    dir_gamma + file_base_gamma,
    file_identifiers_gamma,
    normal_keys,
    ELECTRODE_GEO_AREA_WIDE,
    OVERPOTENTIAL_STD_CURRENT_DENSITY,
    resistances_gamma
)

cv_gamma_lw_std, op_gamma_lw_std, ts_gamma_lw_std = cv_analysis(
    dir_gamma + file_base_gamma,
    file_identifiers_gamma,
    normal_keys,
    ELECTRODE_GEO_AREA_WIDE,
    OVERPOTENTIAL_STD_CURRENT_DENSITY_LOW,
    resistances_gamma
)

cv_alpha2, op_alpha_2, ts_alpha_2 = cv_analysis(
    dir_addtional_exp+file_base_alpha_additional_C1,
    file_identifiers_alpha_additional_1,
    normal_keys,
    ELECTRODE_GEO_AREA_NARROW,
    OVERPOTENTIAL_STD_CURRENT_DENSITY_LOW,
    None,
)

cv_alpha3, op_alpha_3, ts_alpha_3 = cv_analysis(
    dir_addtional_exp+file_base_alpha_additional_C2,
    file_identifiers_alpha_additional_2,
    normal_keys,
    ELECTRODE_GEO_AREA_NARROW,
    OVERPOTENTIAL_STD_CURRENT_DENSITY,
    None,
)

cv_Na_first1, op_Na_first1, ts_Na_first1 = cv_analysis(
    dir_addtional_exp+file_base_Na_first_C1,
    file_identifiers_Na_first_1,
    keys_Na_first,
    ELECTRODE_GEO_AREA_NARROW,
    OVERPOTENTIAL_STD_CURRENT_DENSITY,
    None
    )

cv_Na_first2, op_Na_first2, ts_Na_first2 = cv_analysis(
    dir_addtional_exp+file_base_Na_first_C2,
    file_identifiers_Na_first_2,
    keys_Na_first,
    ELECTRODE_GEO_AREA_NARROW,
    OVERPOTENTIAL_STD_CURRENT_DENSITY,
    None
    )

cv_gamma2, op_gamma2, ts_gamma2 = cv_analysis(
    dir_addtional_exp+file_base_gamma_additional_C1,
    file_identifiers_gamma_additional_1,
    normal_keys,
    ELECTRODE_GEO_AREA_NARROW,
    OVERPOTENTIAL_STD_CURRENT_DENSITY_LOW,
    None
    )

cv_gamma3, op_gamma3, ts_gamma3 = cv_analysis(
    dir_addtional_exp+file_base_gamma_additional_C2,
    file_identifiers_gamma_additional_2,
    normal_keys,
    ELECTRODE_GEO_AREA_NARROW,
    OVERPOTENTIAL_STD_CURRENT_DENSITY_LOW,
    None
    )


# 図作成
# 溶液入れ替えCV追跡
make_op_ts_profile(
    cv_alpha,
    ELECTRODE_GEO_AREA_WIDE,
    op_alpha,
    ts_alpha,
    "_alpha"
)
make_op_ts_profile(
    cv_gamma,
    ELECTRODE_GEO_AREA_WIDE,
    op_gamma,
    ts_gamma,
    "_gamma"
)

make_op_ts_profile(
    cv_alpha3,
    ELECTRODE_GEO_AREA_NARROW,
    op_alpha_3,
    ts_alpha_3,
    "_alpha_reproducibility"
)

make_op_ts_profile(
    cv_gamma3,
    ELECTRODE_GEO_AREA_NARROW,
    op_gamma3,
    ts_gamma3,
    "_gamma_reproducibility"
)

# 再現性
fig_alpha_repr = op_ts_comparison(
    op_alpha,
    ts_alpha,
    "run 1",
    op_alpha_3,
    ts_alpha_3,
    "run 2",
    "_alpha_reproducibility"
)
fig_alpha_repr.get_axes()[0].set_ylabel("E$_{0.5}$ (V vs SHE)")
fig_alpha_repr.get_axes()[1].set_ylabel("TS$_{0.5}$ (mV/dec)")

fig_alpha_repr.get_axes()[0].axvspan(0, 15, color = BACKGROUND_COLOUR_K(), alpha=BACKGROUND_TRANSMITION)
fig_alpha_repr.get_axes()[0].axvspan(15, 30, color = BACKGROUND_COLOUR_Na(), alpha=BACKGROUND_TRANSMITION)
fig_alpha_repr.get_axes()[0].axvspan(30, 45, color = BACKGROUND_COLOUR_K(), alpha=BACKGROUND_TRANSMITION)
fig_alpha_repr.get_axes()[0].axvspan(45, 60, color = BACKGROUND_COLOUR_Na(), alpha=BACKGROUND_TRANSMITION)

fig_alpha_repr.get_axes()[1].axvspan(0, 15, color = BACKGROUND_COLOUR_K(), alpha=BACKGROUND_TRANSMITION)
fig_alpha_repr.get_axes()[1].axvspan(15, 30, color = BACKGROUND_COLOUR_Na(), alpha=BACKGROUND_TRANSMITION)
fig_alpha_repr.get_axes()[1].axvspan(30, 45, color = BACKGROUND_COLOUR_K(), alpha=BACKGROUND_TRANSMITION)
fig_alpha_repr.get_axes()[1].axvspan(45, 60, color = BACKGROUND_COLOUR_Na(), alpha=BACKGROUND_TRANSMITION)

fig_alpha_repr.savefig(
    "exported_figures/" +\
    os.path.basename(__file__)[0:-3] + \
    "reproducibility_alpha" + \
    "_OP_TS_comparison.png",
      dpi = 600)


# fig_gamma_repr = op_ts_comparison(
#     op_gamma_lw_std,
#     ts_gamma_lw_std,
#     op_gamma3,
#     ts_gamma3,
#     "_gamma_reproducibility"
# )
# fig_gamma_repr.get_axes()[0].set_ylabel("E$_{0.3}$ (V vs SHE)")
# fig_gamma_repr.get_axes()[1].set_ylabel("TS$_{0.3}$ (mV/dec)")


# Naから開始
fig_Na_first = op_ts_comparison(
    op_alpha,
    ts_alpha,
    "K$^+$ first", 
    op_Na_first2,
    ts_Na_first2,
    "Na$^+$ first", 
    "_Na_first"
)
fig_Na_first.get_axes()[0].set_ylabel("E$_{0.5}$ (V vs SHE)")
fig_Na_first.get_axes()[1].set_ylabel("TS$_{0.5}$ (mV/dec)")

fig_Na_first.get_axes()[0].text(5, 0.80, "K$^+$", c = clr.black)
fig_Na_first.get_axes()[0].text(20, 0.85, "Na$^+$", c = clr.black)
fig_Na_first.get_axes()[0].text(35, 0.81, "K$^+$", c = clr.black)
fig_Na_first.get_axes()[0].text(50, 0.86, "Na$^+$", c = clr.black)

fig_Na_first.get_axes()[0].text(5, 0.86, "Na$^+$", c = clr.red)
fig_Na_first.get_axes()[0].text(20, 0.81, "K$^+$", c = clr.red)
fig_Na_first.get_axes()[0].text(35, 0.865, "Na$^+$", c = clr.red)
fig_Na_first.get_axes()[0].text(50, 0.815, "K$^+$", c = clr.red)

fig_Na_first.get_axes()[1].text(5, 50, "K$^+$", c = clr.black)
fig_Na_first.get_axes()[1].text(20, 63, "Na$^+$", c = clr.black)
fig_Na_first.get_axes()[1].text(35, 45, "K$^+$", c = clr.black)
fig_Na_first.get_axes()[1].text(50, 60, "Na$^+$", c = clr.black)

fig_Na_first.get_axes()[1].text(5, 70, "Na$^+$", c = clr.red)
fig_Na_first.get_axes()[1].text(20, 48, "K$^+$", c = clr.red)
fig_Na_first.get_axes()[1].text(35, 60, "Na$^+$", c = clr.red)
fig_Na_first.get_axes()[1].text(50, 42, "K$^+$", c = clr.red)

fig_Na_first.savefig(
    "exported_figures/" +\
    os.path.basename(__file__)[0:-3] + \
    "Na_first" + \
    "_OP_TS_comparison.png",
    dpi = 600
    )

#%%
fig, ax = cmn.create_standard_matplt_canvas()

ax.set_yscale("log")
ax.set_ylim(10E-5, 0.5*10E-4)
ax.set_xlim(op_gamma[44].value - 0.015, op_gamma[44].value + 0.015)


cv = voltammograms_gamma["K_2"][-1] 
ax.plot(
    cv.x,
    cv.y,
    lw = 1
)
index_1 = cv.x.find(op_gamma[44])[0] - 25
index_2 = cv.x.find(op_gamma[44])[0] + 25

dot_x = [cv.x[index_1], cv.x[index_2]]
dot_y = [cv.y[index_1], cv.y[index_2]]

ax.scatter(dot_x, dot_y)
print(dot_y)
print(dot_x)

TS = (dot_x[1] - dot_x[0]).value/(np.log10(dot_y[1]) - np.log10(dot_y[0])).value
print(TS)

cv = voltammograms_gamma["K_2"][-2] 
ax.plot(
    cv.x,
    cv.y,
    lw = 0.5
)
index_1 = cv.x.find(op_gamma[44])[0] - 25
index_2 = cv.x.find(op_gamma[44])[0] + 25

dot_x = [cv.x[index_1], cv.x[index_2]]
dot_y = [cv.y[index_1], cv.y[index_2]]

ax.scatter(dot_x, dot_y)
print(dot_y)
print(dot_x)
TS = (dot_x[1] - dot_x[0]).value/(np.log10(dot_y[1]) - np.log10(dot_y[0])).value
print(TS)


#%%再添加実験（改）
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Final
from copy import deepcopy as copy

import BFC_libs.colors as clr
import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec
import BFC_libs.electrochemistry.biologic as bl
from BFC_libs.electrochemistry import Resistance

try:
    from . import lib
except(ImportError):
    import lib

CURRENT_COEFFICIENT: Final = 1000/(0.4*0.4*np.pi) #A -> mA/cm2
OVERPOTENTIAL_STD_CURRENT:Final = 0.5 /CURRENT_COEFFICIENT
COLOR_Na_ADDED = "#BB22FF"

dir = "./data/EC/20240527_2-145_re-addition/"

resistance = bl.BiologicEISData.load_file(dir+"20240527_2-145_inK_02_PEIS_C02.mpt").data[0].get_resistance()

df_K = bl.load_biologic_CV(dir+"20240527_2-145_inK_03_CVA_C02.mpt")\
    .iR_correction(resistance)\
    .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)

df_K_added = bl.load_biologic_CV(dir+"20240527_2-145_K_added_C02.mpt")\
    .iR_correction(resistance)\
    .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)

df_Na_added = bl.load_biologic_CV(dir+"20240527_2-145_Na_added_C02.mpt")\
    .iR_correction(resistance)\
    .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)

df_Na_replaced = bl.load_biologic_CV(dir+"20240527_2-145_Na_replaced_C02.mpt")\
    .iR_correction(resistance)\
    .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)

#all_cv = np.append(np.append(np.append(df_K.data, df_K_added.data), df_Na_added.data), df_Na_replaced.data)
#↑Na有り
all_cv = np.append(np.append(df_K.data, df_K_added.data), df_Na_added.data)


#-----------------CV解析----------------------
OP_list: list[ec.Potential] = []
TS_list: list[float] = []
color_list = []

for i, cv in enumerate(all_cv):
    if not isinstance(cv, ec.Voltammogram):
        raise TypeError
    index = cv.current.find(OVERPOTENTIAL_STD_CURRENT)[0]
    op_tmp = cv.potential[index]
    OP_list.append(copy(op_tmp))

    delta_logj = np.log10(cv.current[index + 25]) - np.log10(cv.current[index - 25])
    delta_E = cv.potential[index+25] - cv.potential[index-25]
    tafel_slope_tmp = float(delta_E)/delta_logj * 1000
    TS_list.append(copy(tafel_slope_tmp))
    
    if cv.data_name.find("Na_added") != -1:
        color_list.append(lib.COLOR_Na())
    elif cv.data_name.find("Na_replaced") != -1:
        color_list.append(clr.red)
    elif cv.data_name.find("K_added") != -1:
        color_list.append(lib.COLOR_K())
    else:
        color_list.append(clr.black)

#-----------------軸設定----------------------
fig = plt.figure(figsize = (4,6))
plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.2)
ax_dec = fig.add_subplot(2,1,1)
ax_tafel = fig.add_subplot(2,1,2)
ax_xlim = (0.65, 0.88)
ax_ylim = (-0.2, 4)

ax_dec.set_xlabel("Potential (V vs. SHE)")
ax_dec.set_ylabel("Current density (mA/cm$^2$)")
ax_dec.set_xlim(*ax_xlim)
ax_dec.set_ylim(*ax_ylim)
ax_dec.text(
    *cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax_dec).to_tupple(), 
    "(a)"
    )

ax_tafel.set_xlabel("Potential (V vs. SHE)")
ax_tafel.set_ylabel("log(current density (mA/cm$^2$))")
ax_tafel.set_xlim(*ax_xlim)
ax_tafel.set_ylim(-1.4, np.log10(ax_ylim[1]))
ax_tafel.text(
    *cmn.convert_relative_pos_to_pos_in_axes_label(
        lib.figure_alphabet_pos, 
        ax_tafel
        )\
            .to_tupple(), 
    "(b)"
    )


for i, cv in enumerate(df_K.data):
    if not isinstance(cv, ec.Voltammogram):
        raise TypeError
    if i != len(df_K.data)-1:
        continue
    ax_dec.plot(cv.x,
            cv.y * CURRENT_COEFFICIENT,
            c = clr.black if i == 0 else clr.blue if i == len(df_K.data)-1 else clr.light_gray,
            lw = 1.5 if i == 0 or i == len(df_K.data)-1 else 0.5
            )
ax_dec.plot(df_K.data[-1].x, df_K.data[-1].y * CURRENT_COEFFICIENT, c = clr.black, ls = ":", label = "K$^+$ solution")
ax_dec.plot(df_K_added.data[0].x, df_K_added.data[0].y * CURRENT_COEFFICIENT, c = lib.COLOR_K(), lw = 1.5, label = "After adding K$^+$ solution")
ax_dec.plot(df_Na_added.data[0].x, df_Na_added.data[0].y * CURRENT_COEFFICIENT, c = lib.COLOR_Na(), lw = 1.5, label = "After adding Na$^+$ solution")
#ax_dec.plot(df_Na_replaced.data[0].x, df_Na_replaced.data[0].y * CURRENT_COEFFICIENT, c = clr.red, lw = 1.5)

ax_tafel.plot(df_K.data[-1].x[10:len(df_K.data[-1].x)//2], np.log10(df_K.data[-1].y[10:len(df_K.data[-1].x)//2] * CURRENT_COEFFICIENT), c = clr.black, ls = ":", label = "K$^+$ solution")
ax_tafel.plot(df_K_added.data[0].x[10:len(df_K_added.data[-1].x)//2], np.log10(df_K_added.data[0].y[10:len(df_K_added.data[-1].x)//2] * CURRENT_COEFFICIENT), c = lib.COLOR_K(), lw = 1.5, label = "After adding K$^+$ solution")
ax_tafel.plot(df_Na_added.data[0].x[10:len(df_Na_added.data[-1].x)//2], np.log10(df_Na_added.data[0].y[10:len(df_Na_added.data[-1].x)//2] * CURRENT_COEFFICIENT), c = lib.COLOR_Na(), lw = 1.5, label = "After adding Na$^+$ solution")
#ax_tafel.plot(df_Na_replaced.data[0].x[10:len(df_Na_replaced.data[-1].x)//2], np.log10(df_Na_replaced.data[0].y[10:len(df_Na_replaced.data[-1].x)//2] * CURRENT_COEFFICIENT), c = clr.red, lw = 1.5)

ax_dec.legend()
ax_tafel.legend()
plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "_before_after_solution_addition.png", dpi=600)


#---------------------------SI用------------------------------------
fig = plt.figure(figsize = (4,6))
plt.subplots_adjust(wspace=0.4, hspace=0.4, left=0.2)

ax_op = fig.add_subplot(2,1,1)
ax_op.set_xlabel("Cycle number")
ax_op.set_ylabel("Potential to reach \n0.5 mA/cm$^2$ (V vs. SHE)")
ax_op.set_xlim(0, 18)
ax_op.set_ylim(0.79, 0.83)
x_scatter = np.linspace(1, len(TS_list), len(TS_list))
ax_op.scatter(x_scatter, OP_list, color = color_list)
ax_op.annotate("Adding K$^+$", (15.9, 0.818), (10, 0.8), arrowprops=dict(arrowstyle = "-|>"))
ax_op.annotate("Adding Na$^+$", (16.9, 0.8255), (7, 0.825), arrowprops=dict(arrowstyle = "-|>"))
#ax_op.annotate("Replacing to Na$^+$", (17.9, 0.8435), (5, 0.835), arrowprops=dict(arrowstyle = "-|>"))
ax_op.text(
    *cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax_op).to_tupple(),
    "(a)"
)

ax_ts = fig.add_subplot(2,1,2)
ax_ts.set_xlabel("Cycle number")
ax_ts.set_ylabel("Tafel slope \nat 0.5 mA/cm$^2$ (mV/dec)")
ax_ts.set_xlim(0, 18)
ax_ts.set_ylim(55, 73)
x_scatter = np.linspace(1, len(TS_list), len(TS_list))
ax_ts.text(
    *cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax_ts).to_tupple(),
    "(b)"
)
ax_ts.scatter(x_scatter, TS_list, color = color_list)
ax_ts.annotate("Adding K$^+$", (15.9, 57.6), (10, 60), arrowprops=dict(arrowstyle = "-|>"))
ax_ts.annotate("Adding Na$^+$", (16.9, 60.5), (10, 62.5), arrowprops=dict(arrowstyle = "-|>"))
#ax_ts.annotate("Replacing to Na$^+$", (17.9, 69.5), (5, 67), arrowprops=dict(arrowstyle = "-|>"))




plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "solution_addition_tracking.png", dpi=600)


#%% full potential-range CV
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Final
from copy import deepcopy as copy

import BFC_libs.colors as clr
import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec
import BFC_libs.electrochemistry.biologic as bl
from BFC_libs.electrochemistry import Resistance

try:
    from . import lib
except(ImportError):
    import lib

CURRENT_COEFFICIENT: Final = 1000/(0.4*0.4*np.pi) #A -> mA/cm2


dir = "./data/EC/20240517_2-140_kinetic_CA/"

resistance = bl.load_biologic_EIS(dir+"20240517_2-140_kinetic_CA_K-1_02_PEIS_C02.mpt").data[0].get_resistance()
df = bl.load_biologic_CV(dir+"20240517_2-140_kinetic_CA_K-1_07_CVA_C02.mpt")\
    .iR_correction(resistance)\
    .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)

fig, ax = cmn.create_standard_matplt_canvas()

ax.set_xlabel("Potential (V vs. SHE)")
ax.set_ylabel("Current density (mA/cm$^2$)")

ax.plot(
    df.data[0].x,
    df.data[0].y * CURRENT_COEFFICIENT,
    c = clr.black
)

plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "_CV_redox.png", dpi=600)

#%%溶液追加実験(旧; 投稿時に使わなければ消す)
import os
import numpy as np
import matplotlib.pyplot as plt

from typing import Final

import BFC_libs.colors as clr
import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec
import BFC_libs.electrochemistry.biologic as bl
from BFC_libs.electrochemistry import Resistance

try:
    from . import lib
except(ImportError):
    import lib

ELECTRODE_GEO_AREA:Final = 2.0 # cm2

def load_data_before_after(
        filepath_before_cv: str, 
        filepath_after_cv: str,
        filepath_eis: str,
    )->tuple[bl.BioLogicVoltammogramData, bl.BioLogicVoltammogramData]:

    eis = bl.load_biologic_EIS(filepath_eis)
    resistance = eis[0].get_resistance()

    data_file_before = bl.load_biologic_CV(
        filepath_before_cv        
    ).iR_correction(resistance).map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)

    data_file_after = bl.load_biologic_CV(
        filepath_after_cv
    ).iR_correction(resistance).map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)

    return data_file_before, data_file_after

X_MIN: Final = ec.Potential(0.70)
X_MAX: Final = ec.Potential(0.85)
X_TICK: Final = np.linspace(float(X_MIN), float(X_MAX), 4)

Y_MIN: Final = ec.Current(-0.05)
Y_MAX: Final = ec.Current(3.0)

def plot_before_after(
        data_before: ec.Voltammogram,
        data_after: ec.Voltammogram,
        ax: plt.Axes,
    )->None:
    """This immutates ax"""

    ax.set_xlabel("$E$ (V vs. SHE)")
    ax.set_ylabel("$j$ (mA/cm$^2$)")
    ax.set_xlim(float(X_MIN), float(X_MAX))
    #ax.set_ylim(float(Y_MIN), float(Y_MAX))
    ax.set_xticks(X_TICK)

    ax.plot(
        data_before.potential,
        data_before.current/ELECTRODE_GEO_AREA*1000,
        c = clr.black
    )
    
    ax.plot(
        data_after.potential,
        data_after.current/ELECTRODE_GEO_AREA*1000,
        c = clr.blue
    )

    return



dir = "./data/EC/20240126_2-102_Add_Na/"

K_addition_before, K_addition_after = load_data_before_after(
    dir + "20240125_2-102_009_K_elctrode3_C01.mpt",
    dir + "20240125_2-102_011_add_K_5mL_elctrode3_C01.mpt",
    dir + "20240125_2-102_007_preconditioning_elctrode3_02_PEIS_C01.mpt",
)

Na_addition_before, Na_addition_after = load_data_before_after(
    dir + "20240125_2-102_003_K_elctrode1_C01.mpt",
    dir + "20240125_2-102_005_add_Na_5mL_elctrode1_C01.mpt",
    dir + "20240125_2-102_001_preconditioning_elctrode1_02_PEIS_C01.mpt",
)

fig = plt.figure(figsize = (4,5))
plt.subplots_adjust(left = 0.2, hspace=0.4)
ax_K_add = fig.add_subplot(2,1,1)
ax_Na_add = fig.add_subplot(2,1,2)

plot_before_after(
    K_addition_before[-1], K_addition_after[0], ax_K_add
)

plot_before_after(
    Na_addition_before[-1], K_addition_after[0], ax_Na_add
)

#text_pos = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax_K_add)

ax_K_add.text(
    *cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax_K_add).to_tupple(),
    "(a)"
    )
ax_Na_add.text(
    *cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax_Na_add).to_tupple(),
    "(b)"
    )

ax_K_add.text(
    *cmn.convert_relative_pos_to_pos_in_axes_label(cmn.Point(0.05, 0.85), ax_K_add).to_tupple(),
    "Addition of K$^+$ into K$^+$"
)

ax_Na_add.text(
    *cmn.convert_relative_pos_to_pos_in_axes_label(cmn.Point(0.05, 0.85), ax_Na_add).to_tupple(),    
    "Addition of Na$^+$ into K$^+$"
)

ax_Na_add.annotate(
    "",
    (0.84, 2.2),
    (0.84, 2.7),
    "data",
    arrowprops=dict(
        width=1.0,
        facecolor = "black",
        headwidth = 5.0,
        headlength = 5.0
    )
)

plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "_before_after_solution_addition.png", dpi=600)

#%% Graphical abstract用


import os
from typing import Final

import numpy as np
import matplotlib.pyplot as plt

import BFC_libs.colors as clr
import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec
import BFC_libs.electrochemistry.biologic as bl
from BFC_libs.electrochemistry import Resistance

try:
    from . import lib
except(ImportError):
    import lib
#---------------------------------パラメータ設定---------------------------------

keys: Final[str] = ["K_1", "Na_1", "K_2", "Na_2"]

ELECTRODE_GEO_AREA: Final[float] = 0.4 * 0.4 * np.pi #直径が1 cmだとずっと思いこんでいたが、8 mmだと気づいた。定性的な結果に影響なし

OVERPOTENTIAL_STD_CURRENT: Final[ec.Current] = ec.Current(0.0005*ELECTRODE_GEO_AREA) # 0.5 mA/cm2

#TAFEL_SLOPE_STD_POTENTIAL: Final[ec.Potential] = ec.Potential(0.8) # V vs SHE

#TAFEL_SLOPE_CALCULATION_WIDTH: Final[ec.Potential] = ec.Potential(0.05) # 50 mV

#---------------------------------データ読み込み---------------------------------
dir_alpha = "./data/EC/20240401_2-128_CV_cycles_alpha_gamma/alpha/"
dir_gamma = "./data/EC/20240401_2-128_CV_cycles_alpha_gamma/gamma/"

file_base_alpha = "20240328_2-128_{}_alpha_{}_{}_C01.mpt"
file_base_gamma = "20240328_2-128_{}_gamma_{}_{}_C02.mpt"

file_identifiers_alpha = {
    "K_1": ("001", "K-1"),
    "Na_1": ("003", "Na-1"),
    "K_2": ("005", "K-2"),
    "Na_2": ("007", "Na-2"),
}
file_identifiers_gamma = {
    "K_1": ("002", "K-1"),
    "Na_1": ("004", "Na-1"),
    "K_2": ("006", "K-2"),
    "Na_2": ("008", "Na-2"),
}


#alphaのデータ読み込み
#ボルタモグラムデータ読み込みしつつiR補正
voltammograms_alpha: dict[str, bl.BioLogicVoltammogramData] = {}
for key in keys:
    resistance = Resistance(
        bl.BiologicEISData.load_file(dir_alpha + file_base_alpha.format(*file_identifiers_alpha[key], "02_PEIS")).data[0].get_resistance()
    )
    voltammograms_alpha[key] = \
        bl.load_biologic_CV(dir_alpha + file_base_alpha.format(*file_identifiers_alpha[key], "03_CVA"))\
            .iR_correction(resistance)\
            .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)
del resistance


#gammaのデータ読み込み
#gammaは抵抗値を手入力（Im(z) = 0を通らないため）
resistances: dict[str, Resistance] = {
    "K_1": Resistance(43.1),
    "Na_1": Resistance(25.6),
    "K_2": Resistance(25.6),
    "Na_2": Resistance(25.8)
}
voltammograms_gamma: dict[str, bl.BioLogicVoltammogramData] = {}
for key in keys:
    voltammograms_gamma[key] = \
    bl.load_biologic_CV(dir_gamma + file_base_gamma.format(*file_identifiers_gamma[key], "03_CVA"))\
        .iR_correction(resistances[key])\
        .map(ec.convert_potential_reference, RE_before = ec.SSCE, RE_after = ec.SHE)
del resistances

#---------------------------------データ読み込み---------------------------------

def cv_analysis(voltammograms: dict[str, bl.BioLogicVoltammogramData], fig_suffix: str):
    #---------------------------------データ解析---------------------------------
    
    overpotential_list:list[ec.Potential] = []
    tafe_slope_list: list[cmn.ValueObject] = []

    for key in keys:
        for i, cv in enumerate(voltammograms[key].data):
            if isinstance(cv, ec.Voltammogram):
                

                op_index = cv.current.find(OVERPOTENTIAL_STD_CURRENT)[0]
                ts_index = (
                    #cv.potential.find(tafel_slope_begin)[0],
                    #cv.potential.find(tafel_slope_end)[0]
                    op_index - 25,
                    op_index + 25
                )

                # TS calculation
                delta_i = (cv.current[ts_index[1]].log10() - cv.current[ts_index[0]].log10())
                delta_E = (cv.potential[ts_index[1]] - cv.potential[ts_index[0]])

                #append
                overpotential_list.append(cv.potential[op_index])
                tafe_slope_list.append((float(delta_E)/float(delta_i))*1000) # V/dec -> mV/dec


    #---------------------------------描画設定---------------------------------
    # Kは青、Naは赤
    color_array = [
        clr.Color(0.0, 0.0, 0.0), #black
        lib.COLOR_K, #blue
        lib.COLOR_Na, #red
        lib.COLOR_K, #blue
        lib.COLOR_Na, #red
    ]
        
    #軸設定
    plt.rcParams["font.size"] = 15
    fig, ax = cmn.create_standard_matplt_canvas()
    
    cv_x_lim_min = 0.7
    cv_x_lim_max = 1
    cv_y_lim_min = -0.2
    cv_y_lim_max = 4
    
    ax.set_xlim(cv_x_lim_min, cv_x_lim_max)
    ax.set_ylim(cv_y_lim_min, cv_y_lim_max)

    #ax.tick_params(labelbottom=False, labelleft=False, labelright=False, labeltop=False,
    #               bottom=False, left=False, right=False, top=False)
    
    ax.set_xlabel("$E$ (V vs. SHE)", fontsize=18)
    ax.set_ylabel("OER activity (mA/cm$^2$)   ", fontsize=18)

    #---------------------------------描画---------------------------------

    current_multipilication = 1000 / (ELECTRODE_GEO_AREA) #A -> mA/cm2
    
    ax.plot(
        voltammograms["K_1"].data[-1].potential,
        voltammograms["K_1"].data[-1].current * current_multipilication,
        c = color_array[1](),
    )
    ax.plot(
        voltammograms["Na_1"].data[-1].potential,
        voltammograms["Na_1"].data[-1].current * current_multipilication,
        c = color_array[2](),
    )


    plt.savefig("exported_figures/" + os.path.basename(__file__)[0:-3] + "_for_graohical_abstract.png", dpi = 600)
    return (overpotential_list, tafe_slope_list)

op_alpha, ts_alpha = cv_analysis(voltammograms_alpha, "_alpha")

#%% カチオン依存性定量的評価
import os
from typing import Final

import numpy as np
import matplotlib.pyplot as plt

import BFC_libs.colors as clr
import BFC_libs.common as cmn

# data from Tables S4-S5
# manually input
op_alpha_avr = [
    810.5,
    846.5,
    823.0,
    850.8
]

op_alpha_std = [
    0.8,
    0.7,
    0.2,
    0.4
]

ts_alpha_avr = [
    55.7,
    59.8,
    51.2,
    58.0
]

ts_alpha_std = [
    0.2,
    0.4,
    0.2,
    0.2
]

op_gamma_avr = [
    832,
    856.5,
    843.1,
    844.8
]

op_gamma_std = [
    2,
    0.2,
    0.3,
    0.5
]

ts_gamma_avr = [
    63.1,
    63.4,
    66.9,
    65.0
]

ts_gamma_std = [
    0.2,
    0.2,
    0.3,
    0.1
]


solutions = ["K$^+$ 1st", "Na$^+$ 1st", "K$^+$ 2nd", "Na$^+$ 2nd"]

fig = plt.figure(figsize = (8,3))

ax_op = fig.add_axes((0.1, 0.2, 0.35, 0.7))

#fig_op, ax_op = cmn.create_standard_matplt_canvas()

#ax_right = ax_op.twinx()

ax_op.set_xlabel("Periods")
ax_op.set_ylabel("$E_{0.5}$ (mV vs SHE)")
#ax_right.set_ylabel("Tafel slope (mV/dec)")

ax_op.errorbar(
    solutions,
    op_alpha_avr,
    yerr = op_alpha_std,
    c = clr.red,
    label = r"$\mathrm{\alpha}$-MnO$_2$",
    marker = "o",
    markersize = 5,
    capsize= 4
)

ax_op.errorbar(
    solutions,
    op_gamma_avr,
    yerr = op_gamma_std,
    c = clr.blue,
    label = r"$\mathrm{\gamma}$-MnO$_2$",
    marker = "o",
    markersize = 5,
    capsize= 4
)
ax_op.legend()

pos_a = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax_op)
ax_op.text(pos_a.x, pos_a.y, "(a)")

ax_ts = fig.add_axes((0.6, 0.2, 0.35, 0.7))
ax_ts.set_xlabel("Periods")
ax_ts.set_ylabel("TS$_{0.5}$ (mV/dec)")

ax_ts.errorbar(
    solutions,
    ts_alpha_avr,
    yerr = ts_alpha_std,
    c = clr.red,
    label = r"$\mathrm{\alpha}$-MnO$_2$",
    marker = "o",
    markersize = 5,
    capsize= 4
)

ax_ts.errorbar(
    solutions,
    ts_gamma_avr,
    yerr = ts_gamma_std,
    c = clr.blue,
    label = r"$\mathrm{\gamma}$-MnO$_2$",
    marker = "o",
    markersize = 5,
    capsize= 4
)
ax_ts.legend()

pos_b = cmn.convert_relative_pos_to_pos_in_axes_label(lib.figure_alphabet_pos, ax_ts)
ax_ts.text(pos_b.x, pos_b.y, "(b)")


fig.savefig(
    "exported_figures/" +\
    os.path.basename(__file__)[0:-3] + \
    "_cation_dependence.png",
      dpi = 600)
