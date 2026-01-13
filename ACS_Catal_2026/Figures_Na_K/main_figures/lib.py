#%%
import numpy as np
import math
import matplotlib.pyplot as plt
import copy
import pandas as pd
from collections import namedtuple
from dataclasses import dataclass
from scipy.stats import linregress
from typing import TypeVar, Generic

from BFC_libs.physical_constants import F, R
import BFC_libs.common as cmn 
from BFC_libs import colors as clr

T = 298

#COLOR_K = clr.Color(0.33, 0.33, 0.73)
#COLOR_Na = clr.Color(0.73, 0.33, 0.33)

COLOR_K = clr.Color.from_color_code("#005B82")
COLOR_Na = clr.Color.from_color_code("#D51506")

# user-defined objects
#Point = namedtuple('Point', 'x y')

figure_alphabet_pos = cmn.Point(-0.15, 1.05)
figure_alphabet_pos_no_label = cmn.Point(-0.1, 1.05)

@dataclass
class RedoxKinetics:
    alpha: float
    electron_n: float
    k0: float
    E0: float
    
    def forward_rate_array(self, applied_E_array)->np.ndarray:
        delta_E = applied_E_array - self.E0
        numerator = self.alpha *  F * delta_E
        denominator = R*T
        
        return self.k0*np.exp(numerator/denominator)
    
    def forward_rate(self, applied_E)->float:
        delta_E = applied_E - self.E0
        numerator = self.alpha *  F * delta_E
        denominator = R*T
        return self.k0*np.exp(numerator/denominator)


    def backward_rate_array(self, applied_E_array)->np.ndarray:
        delta_E = applied_E_array - self.E0
        numerator = - (1-self.alpha) *  F * delta_E
        denominator = R*T
        return self.k0*np.exp(numerator/denominator)
    
    def backward_rate(self, applied_E)->float:
        delta_E = applied_E - self.E0
        numerator = - (1-self.alpha) *  F * delta_E
        denominator = R*T
        return self.k0*np.exp(numerator/denominator)


    
#alpha_eff of OER vs EC features of pre-redox peak
def fit_trumpet(
    Ea: np.ndarray,
    Ec: np.ndarray,
    logv: np.ndarray,
    #mask_threshold = 0.1
    mask_point_from_last = 3
    )->tuple[RedoxKinetics, cmn.Point, cmn.Point, cmn.Point]:
    """ 
    単位変換などは行わない。
    この関数にそのような役割は持たせない。
    ## 引数:
        Ea: anode掃引時のピーク[V]のndarray
        Ec: cathode掃引時のピーク[V]のndarray
        logv: 掃引速度(V/s)の常用対数
        mask_point_from_last: 最高掃引速度から何plotを解析するか
    ## 返り値: 
        (alpha,n,k,E0), (x0,y0), (x1,y1), (x2,y2)の順のtuple
        alpha: 0から1の値
        n: 電子移動数
        k: 1/sに換算
        E0: V 参照極は変更しない
        (x0, y0): anode側fittingとcathode側fittingの交点
        (x1, y1): anode側のfittingの端
        (x2, y2): cathode側のfittingの端

    """
    #最後から該当する点だけ抽出
    peak_pos_anod:np.ndarray = Ea[-1*mask_point_from_last:]
    peak_pos_cath:np.ndarray = Ec[-1*mask_point_from_last:]
    ln_v:np.ndarray = (logv[-1*mask_point_from_last:])*np.log(10) 

    # 直線フィッティング
    fit1 = linregress(ln_v,peak_pos_anod)
    slope_anod,intercept_anod = fit1.slope, fit1.intercept 
    fit2 = linregress(ln_v,peak_pos_cath)
    slope_cath,intercept_cath = fit2.slope, fit2.intercept

    #交点とかを数学的に計算
    cross_x = (intercept_cath-intercept_anod)/(slope_anod-slope_cath)/np.log(10)
    cross_y = (slope_anod*intercept_cath-slope_cath*intercept_anod)/(slope_anod-slope_cath)
    cross_point = cmn.Point(cross_x, cross_y)
    
    x1 = ln_v[-1]/np.log(10)
    y1 = slope_anod*ln_v[-1]+intercept_anod
    edge_point_anod = cmn.Point(x1, y1)

    x2 = ln_v[-1]/np.log(10)
    y2 = slope_cath*ln_v[-1]+intercept_cath
    edge_point_cath = cmn.Point(x2, y2)

    # 物理化学パラメータの計算
    alpha = slope_cath/(slope_cath-slope_anod)
    A = slope_anod*alpha # this should be equal to slope_cath*(alpha-1)
    n = 1/(A*F/R/T)

    lnk = \
        alpha*(alpha-1)*(intercept_anod-intercept_cath)/A \
        +(alpha-1)*np.log(slope_anod)\
        -alpha*np.log(-slope_cath)
    k = np.exp(lnk) #

    # 最低掃引速度のピーク位置平均値から計算
    E0 = (Ea[0]+Ec[0])/2

    reaction_kinetics = RedoxKinetics(alpha, n, k, E0)

    return reaction_kinetics, cross_point, edge_point_anod, edge_point_cath

