"""
伝送線モデル込みで逐次電流値計算をすることで、周期電位入力に対する電流応答をシミュレートし、位相と抵抗を返す
時定数が周期に対して十分早いとき、各電位での定常状態を計算する。そうでないときは非定常状態の計算をする。
伝送線モデルのセルを一つずつ計算して、収束した段階で計算を打ち止め、収束値から全体値を逆算する。
異常に時定数が長いとき（100秒以上）は即座に値を返す（360 deg, 10^-7 Ohm）。
"""
import numpy as np
import cmath

from enum import Enum, auto
from dataclasses import dataclass

import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec
from BFC_libs.physical_constants import F, R, T

from general_kinetics import Butler_Volmer_ano, Butler_Volmer_cath

# class CaliculationMethod(Enum):
#     WAIT_STEADY_STATE = auto #各電位で定常になるのを待ち、総電荷と時間から電流値を測定。時定数に対して周期が長いとき
#     SIMPLE_SEQUENCE = auto #時間ピリオド毎に単純計算。時定数にたいして周期が短いとき。
#     CONSTANT_THETA_1 = auto
#     CONSTANT_THETA_0 = auto

# def faraday_impedance_simulation(
#     #input parameters
#     frequency: float,
#     E_center: float,
#     E_amplituide: float,
#     conc_proton_source: float,
#     conc_hydroxide: float,
#     conc_hydrogen: float,
    
#     #reaction1
#     log_k0_1: float,
#     alpha1: float,
#     E0_1: float,

#     #reacton2
#     log_k0_2: float,
#     alpha2: float,
#     E0_2: float,

#     #other conditons
#     SURFACE_AREA = 0.15*0.15*np.pi, #default: RDE disk
#     plot_flag: bool = False
# ):
#     """
#     returning a tuple of(phase, absolute R)
#     the impedance at the potrntial and frequency can be led from
#     abs_R*sin(phase) + 
#     """
#     #数値初期化
#     period_time = 1/frequency

#     SIMULATED_PERIOD_NUMBER = 5
#     #TRANSMISSION_LINE_MODEL = 10
#     AMPLITUDE = E_amplituide

#     MINIMUM_PERIOD_NUMBER = 720
#     MAXIMUM_PERIOD_NUMBER = 10000

#     #E_center = -1.12

#     FRT = F/R/T
#     #conc_substrate = 7 #mol/L


#     def k1(E: float): #[1/s]
#         #return (10**log_k0_1)*conc_proton_source*np.exp(FRT*alpha1*-1*(E-E0_1))
#         return conc_proton_source*Butler_Volmer_cath(
#             k0=10**log_k0_1,
#             alpha=alpha1,
#             E0 = E0_1,
#             applied_E= E
#             )

#     def k2(E: float):#[1/s]
#         #return (10**log_k0_2)*conc_proton_source*np.exp(FRT*alpha2*-1*(E-E0_2))
#         return conc_proton_source*Butler_Volmer_cath(
#             k0=10**log_k0_2,
#             alpha=alpha2,
#             E0 = E0_2,
#             applied_E= E
#             )

#     def k1r(E: float):#[1/s]
#         #return (10**log_k0_1)*conc_hydroxide*np.exp(FRT*(1-alpha1)*(E-E0_1))
#         return conc_hydroxide*Butler_Volmer_ano(
#             k0=10**log_k0_1,
#             alpha=1-alpha1,
#             E0 = E0_1,
#             applied_E= E
#             )

#     def k2r(E: float):#[1/s]
#         #return (10**log_k0_2)*conc_hydroxide*conc_hydrogen*np.exp(FRT*(1-alpha2)*(E-E0_2))
#         return conc_hydroxide*conc_hydrogen*Butler_Volmer_ano(
#             k0=10**log_k0_2,
#             alpha=1-alpha2,
#             E0 = E0_2,
#             applied_E= E
#             )



#     #print("{}, {}, {}, {}".format(k1(E_center), k2(E_center), k1r(E_center), k2r(E_center)))

#     faraday_kinetic_scale:float = np.array([k1(E_center),k2r(E_center),k1r(E_center),k2(E_center)]).max() #[1/sec]
#     max_faraday_time_scale: float = 1/np.array([k1(E_center),k2r(E_center),k1r(E_center),k2(E_center)]).min()
#     min_faraday_time_scale: float = 1/np.array([k1(E_center),k2r(E_center),k1r(E_center),k2(E_center)]).max()
#     #print("faraday time scale: {}".format(1/faraday_kinetic_scale))

#     #あまりにも時定数が遅い場合、かなり変位が大きくなるような値を返して抜ける
#     #if 1/faraday_kinetic_scale > 100:
#     #    return (360.0, 10E-8) 

#     E_one_point_later = E_center+AMPLITUDE*np.sin(2*np.pi/MINIMUM_PERIOD_NUMBER)
#     faraday_kinetic_scale_one_point_later = np.array([k1(E_one_point_later),k2r(E_one_point_later),k1r(E_one_point_later),k2(E_one_point_later)]).max() #[1/sec]
#     faraday_current_scale_difference = F * np.abs(faraday_kinetic_scale - faraday_kinetic_scale_one_point_later)
#     faraday_registance_scale = (E_one_point_later - E_center) / faraday_current_scale_difference
#     #print("R_F: {}".format(faraday_registance_scale))

#     intermediate_generation_kinetic_scale = k1(E_center) + k2r(E_center)
#     intermediate_consumption_kinetic_scale = k1r(E_center) + k2(E_center)

#     if min_faraday_time_scale < period_time/MINIMUM_PERIOD_NUMBER/50:
#         #反応のタイムスケールよりも十分に一点間の反応が長い場合、定常状態になるのを待つ。
#         calculation_method = CaliculationMethod.WAIT_STEADY_STATE
#         time_scale = period_time/MINIMUM_PERIOD_NUMBER
#         print("calculate S.S.")
#         #print("time scale = {}".format(time_scale))
    
#     # elif intermediate_generation_kinetic_scale > intermediate_consumption_kinetic_scale*1000:
#     #     # 中間体被覆率θがずっと1という仮定
#     #     # 中間体消費速度がそのまま電流値になる
#     #     calculation_method = CaliculationMethod.CONSTANT_THETA_1
#     #     time_scale = period_time/MINIMUM_PERIOD_NUMBER
#     #     print("Assumption: theta_int = 1")

#     # elif intermediate_consumption_kinetic_scale > intermediate_generation_kinetic_scale*1000:
#     #     #中間体被覆率θがずっと0という仮定
#     #     #中間体生成速度がそのまま電流値になる
#     #     calculation_method = CaliculationMethod.CONSTANT_THETA_0
#     #     time_scale = period_time/MINIMUM_PERIOD_NUMBER
#     #     print("Assumption: theta_int = 0")

#     else:
#         #上記のどれにも当てはまらない場合に一点ずつ計算する
#         calculation_method = CaliculationMethod.SIMPLE_SEQUENCE
#         time_scale = 1/faraday_kinetic_scale
#         print("calculate sequentially")
#         #print("time scale = {}".format(time_scale))

#     points_per_period = int((MINIMUM_PERIOD_NUMBER/100)*period_time/time_scale) if period_time/time_scale>100 else MINIMUM_PERIOD_NUMBER
#     print("caliculation point per period: {}".format(points_per_period))
#     t = np.linspace(
#         0,
#         SIMULATED_PERIOD_NUMBER, 
#         SIMULATED_PERIOD_NUMBER*points_per_period
#         ) #[period]
#     point_time = period_time/points_per_period

#     #時間方向
#     E_diffs = AMPLITUDE*np.sin(t*np.pi*2)
#     E_list= E_center + E_diffs 
    
#     #空間方向
#     #print(E_list)
#     theta_int_init = (k1(E_center)+k2r(E_center))/(k1(E_center)+k2r(E_center)+k1r(E_center)+k2(E_center))

#     theta_int = theta_int_init
#     theta_star = 1 - theta_int

#     theta_avr_list = np.array([])

#     I=0
#     I_total_list = np.array([])
    
#     #print("---")
#     #print(period_time)
#     #print(points_per_period)
#     #print(point_time)

#     match(calculation_method):
#         # #中間体被覆率が変化しないという仮定のとき、phaseは0、RはEが最大値のときと最小値の時の差からすぐに算出可
#         # case CaliculationMethod.CONSTANT_THETA_0:
#         #     #電流値は中間体生成過程が支配
#         #     I_max = (k1(E_center + E_amplituide) - k2r(E_center+E_amplituide))*F* SURFACE_AREA
#         #     I_min = (k1(E_center - E_amplituide) - k2r(E_center-E_amplituide))*F* SURFACE_AREA
#         #     print("{}, {}".format(I_max, I_min))
#         #     I_diff = I_max - I_min 
#         #     resistance = abs(E_amplituide*2 / I_diff)
            
#         #     return (0.0, resistance)
        
#         # case CaliculationMethod.CONSTANT_THETA_1:
#         #     #電流値は中間体消費過程が支配
#         #     I_max = (k2(E_center + E_amplituide) - k1r(E_center+E_amplituide)) * F* SURFACE_AREA
#         #     I_min = (k2(E_center - E_amplituide) - k1r(E_center-E_amplituide)) * F* SURFACE_AREA
#         #     I_diff = I_max - I_min    
#         #     resistance = abs(E_amplituide*2 / I_diff)
            
#         #     return (0.0, resistance)
        
#         case CaliculationMethod.SIMPLE_SEQUENCE:
#             #time_scale = total_time_scale

#             for j, E_applied in enumerate(E_list):
        
#                 #-----Faraday current---------
#                 delta_theta_int = \
#                     (theta_star*k1(E_applied) - theta_int*k2(E_applied) \
#                     + theta_star * k2r(E_applied) - theta_int * k1r(E_applied)) * \
#                     point_time
                
#                 #theta_list = np.append(theta_list, delta_theta_int)
#                 theta_int += delta_theta_int
#                 theta_star -= delta_theta_int
                
#                 if theta_int > 1:
#                     theta_int = 1
#                     theta_star = 0

#                 if theta_int < 0:
#                     theta_int = 0
#                     theta_star = 1
                

#                 I = -(theta_int * k2(E_applied) - (1-theta_int)*k2r(E_applied) \
#                     +(1-theta_int) * k1(E_applied) - theta_int*k1r(E_applied)) * F * SURFACE_AREA

#                 I_total_list = np.append(I_total_list, I)
#                 theta_avr_list = np.append(theta_avr_list, theta_int)
            
                    
#         case CaliculationMethod.WAIT_STEADY_STATE:

#             for E_applied in E_list:
#                 #作用極-電解液間電位

                
#                 faraday_current = 0.0
#                 faraday_current_previous_loop = 0.0

#                 loop_time = 1/faraday_kinetic_scale/100 #[sec]
#                 repeat_time = 0
#                 convergenced = False

#                 #定常になるまで待つループ
#                 while convergenced == False:
#                     #伝送線のループ
                        
#                     #Faraday charge
#                     delta_theta_int = \
#                         (theta_star*k1(E_applied) - theta_int*k2(E_applied) \
#                         + theta_star * k2r(E_applied) - theta_int * k1r(E_applied)) * \
#                         point_time
                    
#                     theta_int += delta_theta_int
#                     theta_star -= delta_theta_int
                    
#                     if theta_int > 1:
#                         theta_int = 1
#                         theta_star = 0

#                     if theta_int < 0:
#                         theta_int = 0
#                         theta_star = 1

                                        
#                     #総ファラデー電流計算・更新
#                     faraday_current_previous_loop = faraday_current
#                     faraday_current = -(theta_int * k2(E_applied) - (1-theta_int)*k2r(E_applied) \
#                         + (1-theta_int) * k1(E_applied) - theta_int*k1r(E_applied)) * F * SURFACE_AREA
#                     repeat_time += 1

#                     #ファラデー電流とキャパシタンスの電荷が収束したか否か
#                     if abs(faraday_current-faraday_current_previous_loop)<abs(faraday_current)*0.01:
#                         convergenced = True

                
#                 #伝送戦モデルでの物理量の平均・総和
#                 """theta_avr_list = np.append(theta_avr_list, theta_int.mean())
#                 I_total_list = np.append(I_total_list, i_cell_faraday.sum() + double_layer_charge/point_time)
#                 """
#                 theta_avr_list = np.append(theta_avr_list, theta_int)
#                 I = faraday_current
#                 I_total_list = np.append(I_total_list, I)
                

#     #phase取得
#     I_last_period = I_total_list[-points_per_period:]
#     index_I_max = np.where(I_last_period == I_last_period.max())[0][0]
#     index_V_max = np.where(E_diffs[-points_per_period:] == E_diffs[-points_per_period:].max())[0][0]
#     I_phase = (index_I_max-index_V_max)/points_per_period*360 #rad to degree 
#     #絶対値180deg以上の時、反対周りの角度に変換
#     if I_phase > 180:
#         I_phase = I_phase-360
#     if I_phase < -180:
#         I_phase = 360 - I_phase
#     #R絶対値
#     i_width = np.abs(max(I_last_period)-min(I_last_period))/2
#     R_abs = np.abs(AMPLITUDE/i_width)
    
    
#     """print("frequency: {}".format(frequency))
#     print("phase: {}".format(I_phase))
#     print("|R|: {}".format(R_abs))
#     print("--")"""
    
#     #print(E_electrode)
#     if plot_flag ==True:
        
#         divd_value = 2

#         fig, ax = cmn.create_standard_matplt_canvas()
#         #ax.set_xlim(6, 10)
#         #ax.vlines([6.5, 8.5], [np.min(E_list), np.min(E_list)], [np.max(E_list), np.max(E_list)], linestyles=":", lw =1)
#         ax_right = ax.twinx()
#         ax.plot(t[-len(t)//divd_value:], E_list[-len(t)//divd_value:], c = "#FF5555")
#         ax_right.plot(t[-len(t)//divd_value:], theta_avr_list[-len(t)//divd_value:], c = "#5555FF", ls = ":")

#         fig, ax = cmn.create_standard_matplt_canvas()
#         #ax.set_xlim(6, 10)
#         #ax.vlines([6.5, 8.5], [np.min(E_list), np.min(E_list)], [np.max(E_list), np.max(E_list)], linestyles=":", lw =1)
#         ax_right = ax.twinx()
#         ax.plot(t[-len(t)//divd_value:], E_list[-len(t)//divd_value:], c = "#FF5555")
#         ax_right.plot(t[-len(t)//divd_value:], I_total_list[-len(t)//divd_value:], c = "#5555FF", ls = ":")

#     return (I_phase, R_abs)

# def total_impedance_simulation(
#     #farraday impedace parameters
#     frequency: float,
#     E_center: float,
#     E_amplituide: float,
#     conc_proton_source: float,
#     conc_hydrogen: float,
#     conc_hydroxide: float,
    
#     #reaction1
#     log_k0_1: float,
#     alpha1: float,
#     E0_1: float,

#     #reacton2
#     log_k0_2: float,
#     alpha2: float,
#     E0_2: float,

#     #other impedances
#     solution_resistanse: float,
#     double_layer_capasitance: float,
#     pore_resistance: float,

#     #other conditons
#     lambda_fac: float,
#     SURFACE_AREA = 0.15*0.15*np.pi, #default: RDE disk

    
# ):
    
#     def Z_F(R, phase):
#         return R*np.exp(phase*1j)
    
#     phase, R = faraday_impedance_simulation(
#         frequency=frequency,
#         E_center=E_center,
#         E_amplituide=E_amplituide,
#         conc_proton_source=conc_proton_source,
#         conc_hydrogen=conc_hydrogen,
#         conc_hydroxide=conc_hydroxide,

#         log_k0_1=log_k0_1,
#         alpha1=alpha1,
#         E0_1 = E0_1,
#         log_k0_2=log_k0_2,
#         alpha2=alpha2,
#         E0_2=E0_2,

#         SURFACE_AREA=SURFACE_AREA
#     )
#     Z_B= (1j*Z_F(R, phase))/(1j-double_layer_capasitance*frequency*Z_F(R, phase))
#     electrode_impedance = cmath.sqrt(pore_resistance * Z_B)*(1/cmath.tanh(cmath.sqrt(pore_resistance/Z_B)*lambda_fac))
#     total_impedance = electrode_impedance + solution_resistanse

#     return total_impedance


# def impedance_simulation(
#     #input parameters
#     frequency: float,
#     E_center: float,
#     E_amplituide: float,
#     conc_substrate: float,
#     conc_product: float,
    
#     #reaction1
#     log_k0_1: float,
#     alpha1: float,
#     E0_1: float,

#     #reacton2
#     log_k0_2: float,
#     alpha2: float,
#     E0_2: float,

#     #other conditons
#     solution_registance: float=1, #[Ohm]
#     double_layer_capacitance:float = 10E-9, #[F]
#     pore_registance: float = 10E-4,

#     TRANSMISSON_LINE_MODEL_LENGTH: int = 10,
#     SURFACE_AREA = 0.15*0.15*np.pi, #default: RDE disk
#     plot_flag: bool = False
# ):
#     #数値初期化
#     period_time = 1/frequency

#     SIMULATED_PERIOD_NUMBER = 3
#     #TRANSMISSION_LINE_MODEL = 10
#     AMPLITUDE = E_amplituide

#     MINIMUM_PERIOD_NUMBER = 360
#     MAXIMUM_PERIOD_NUMBER = 10000

#     #E_center = -1.12

#     FRT = F/R/T
#     #conc_substrate = 7 #mol/L

#     """#reaction1 * + sub -> *int
#     alpha1 = 0.8 #[-]
#     k0_1 = 2*10E0#[1/s]
#     E0_1= -1.09 #[V vs RE]

#     #reaction2 *int + sub-> * + product
#     alpha2 = 0.3
#     k0_2 = 2*10E-1
#     E0_2 = -1.10"""

#     def k1(E: float): #[1/s]
#         return (10**log_k0_1)*conc_substrate*np.exp(FRT*alpha1*-1*(E-E0_1))

#     def k2(E: float):#[1/s]
#         return (10**log_k0_2)*conc_substrate*np.exp(FRT*alpha2*-1*(E-E0_2))

#     def k1r(E: float):#[1/s]
#         return (10**log_k0_1)*conc_product*np.exp(FRT*(1-alpha1)*(E-E0_1))

#     def k2r(E: float):#[1/s]
#         return (10**log_k0_2)*conc_product*np.exp(FRT*(1-alpha2)*(E-E0_2))

#     faraday_kinetic_scale:float = np.array([k1(E_center),k2r(E_center),k1r(E_center),k2(E_center)]).max() #[1/sec]
#     #print("faraday time scale: {}".format(1/faraday_kinetic_scale))

#     #あまりにも時定数が遅い場合、かなり変位が大きくなるような値を返して抜ける
#     #if 1/faraday_kinetic_scale > 100:
#     #    return (360.0, 10E-8) 

#     E_one_point_later = E_center+AMPLITUDE*np.sin(2*np.pi/MINIMUM_PERIOD_NUMBER)
#     faraday_kinetic_scale_one_point_later = np.array([k1(E_one_point_later),k2r(E_one_point_later),k1r(E_one_point_later),k2(E_one_point_later)]).max() #[1/sec]
#     faraday_current_scale_difference = F * np.abs(faraday_kinetic_scale - faraday_kinetic_scale_one_point_later)
#     faraday_registance_scale = (E_one_point_later - E_center) / faraday_current_scale_difference
#     #print("R_F: {}".format(faraday_registance_scale))
#     double_layer_time_scale = double_layer_capacitance * faraday_registance_scale
#     #print("dl time scale: {}".format(double_layer_time_scale))
#     total_time_scale = max([1/faraday_kinetic_scale, double_layer_time_scale])

#     if total_time_scale*100 < period_time/MINIMUM_PERIOD_NUMBER:
#         #キャパシタンス及び反応のタイムスケールよりも十分に一点間の反応が長い場合、定常状態になるのを待つ。伝送線モデルを使わない
#         calculation_method = CaliculationMethod.WAIT_STEADY_STATE
#         time_scale = period_time/MINIMUM_PERIOD_NUMBER
#         #print("calculate S.S.")
#         #print("time scale = {}".format(time_scale))

#     else:
#         #そうでない場合、単純に一点ずつ計算する。伝送線モデルを使う。
#         calculation_method = CaliculationMethod.SIMPLE_SEQUENCE
#         time_scale = total_time_scale
#         #print("calculate sequentially")
#         #print("time scale = {}".format(time_scale))

#     points_per_period = int((MINIMUM_PERIOD_NUMBER/100)*period_time/time_scale) if period_time/time_scale>100 else MINIMUM_PERIOD_NUMBER
#     #print("caliculation point per period: {}".format(points_per_period))
#     t = np.linspace(
#         0,
#         SIMULATED_PERIOD_NUMBER, 
#         SIMULATED_PERIOD_NUMBER*points_per_period
#         ) #[period]

#     #時間方向
#     E_list=AMPLITUDE*np.sin(t*np.pi*2)+E_center 
    
#     #空間方向
#     E_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, E_center)
#     #print(E_list)
#     theta_int_init = (k1(E_center)+k2r(E_center))/(k1(E_center)+k2r(E_center)+k1r(E_center)+k2(E_center))

#     theta_int = np.full(TRANSMISSON_LINE_MODEL_LENGTH, theta_int_init)
#     theta_star = np.full(TRANSMISSON_LINE_MODEL_LENGTH, 1 - theta_int)

#     theta_avr_list = np.array([])

#     I=E_amplituide/solution_registance/100

#     cdl_voltage = np.full(TRANSMISSON_LINE_MODEL_LENGTH, E_center)
#     delta_charge_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, 0)

#     i_cell_faraday = np.full(TRANSMISSON_LINE_MODEL_LENGTH, I/TRANSMISSON_LINE_MODEL_LENGTH)
#     i_cell_double_layer = np.full(TRANSMISSON_LINE_MODEL_LENGTH, I/TRANSMISSON_LINE_MODEL_LENGTH)
#     i_cell = np.zeros(TRANSMISSON_LINE_MODEL_LENGTH)

#     I_total_list = np.array([])

#     point_time = period_time/points_per_period
    
#     #print("delta E : {}".format(E_one_point_later - E_center))
#     #print("faraday kinetic scale change: {} -> {}".format(faraday_kinetic_scale, faraday_kinetic_scale_one_point_later))
#     #print("furaday current scale : {}".format(faraday_current_scale_difference))
#     #print("faraday registance scale : {}".format(faraday_registance_scale))
    
#     #print("total time scale : {}".format(total_time_scale))
#     #raise(SyntaxError)


#     match(calculation_method):
#         case CaliculationMethod.SIMPLE_SEQUENCE:
#             #time_scale = total_time_scale
#             """points_per_period = int((MINIMUM_PERIOD_NUMBER/100)*period_time*time_scale) if period_time*time_scale>100 else MINIMUM_PERIOD_NUMBER
#             t = np.linspace(
#                 0,
#                 SIMULATED_PERIOD_NUMBER, 
#                 SIMULATED_PERIOD_NUMBER*points_per_period
#                 ) #[period]

#             #時間方向
#             E_list=AMPLITUDE*np.sin(t*np.pi*2)+E_center 
            
#             #空間方向
#             E_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, E_center)

#             theta_int_init = (k1(E_center)+k2r(E_center))/(k1(E_center)+k2r(E_center)+k1r(E_center)+k2(E_center))

#             theta_int = np.full(TRANSMISSON_LINE_MODEL_LENGTH, theta_int_init)
#             theta_star = np.full(TRANSMISSON_LINE_MODEL_LENGTH, 1 - theta_int)

#             theta_avr_list = np.array([])

#             I=E_center/solution_registance/1000

#             cdl_voltage = np.full(TRANSMISSON_LINE_MODEL_LENGTH, E_center)
#             delta_charge_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, 0)

#             i_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, I/TRANSMISSON_LINE_MODEL_LENGTH)
#             I_total_list = np.array([])"""

            
#             convergenced = False
#             repeat_time = 0

#             while convergenced == False and repeat_time<10:
#                 for E_applied in E_list:
#                     #作用極-電解液間電位
#                     E_electrode = E_applied - I*solution_registance
                    
#                     if np.abs(E_electrode-E_applied) > 1:
#                         raise ValueError(
#                             "Too large iR drop.\nI: {}\nE_applied: {}".format(I, E_applied)
#                         )
#                     #point_time = period_time / points_per_period
                    
#                     #伝送戦モデル
#                     for i in range(TRANSMISSON_LINE_MODEL_LENGTH):
#                         if repeat_time == 0:
#                             if i == 0:
#                                 E_cell[0] = E_electrode
#                             else:
#                                 E_cell[i]= E_cell[i-1] - i_cell[i-1] * pore_registance
                        
#                         #-----Faraday current---------
#                         delta_theta_int = \
#                             (theta_star[i]*k1(E_electrode) - theta_int[i]*k2(E_electrode) \
#                             + theta_star[i] * k2r(E_electrode) - theta_int[i] * k1r(E_electrode)) * \
#                             point_time
                        
#                         #theta_list = np.append(theta_list, delta_theta_int)
#                         theta_int[i] += delta_theta_int
#                         theta_star[i] -= delta_theta_int
                        
#                         if theta_int[i] > 1:
#                             theta_int[i] = 1
#                             theta_star[i] = 0

#                         if theta_int[i] < 0:
#                             theta_int[i] = 0
#                             theta_star[i] = 1

#                         #------capacitance current-------
#                         delta_E_cap = E_cell[i] - cdl_voltage[i] #[V]
#                         #print(delta_E_cap)
#                         cdl_voltage[i] = E_cell[i] #[V]

#                         delta_charge_cell = delta_E_cap * double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH #[C]
                        
#                         i_cell_faraday[i] = (theta_int[i] * k2(E_electrode) - (1-theta_int[i])*k2r(E_electrode) + \
#                             (1-theta_int[i]) * k1(E_electrode) - theta_int[i]*k2(E_electrode)) * F * SURFACE_AREA /TRANSMISSON_LINE_MODEL_LENGTH
                            
#                         i_cell_double_layer[i] = delta_charge_cell/point_time/TRANSMISSON_LINE_MODEL_LENGTH
#                         i_cell[i] = i_cell_faraday[i] + i_cell_double_layer[i]
#                         print(i_cell_faraday)
#                         print(i_cell_double_layer)

#                         #伝送線モデル計算が収束していたら抜ける
#                         if i > 20:
#                             sep = i//2
#                             i_total_initial = i_cell[:i][0:sep].mean()
#                             i_total_last = i_cell[:i][sep:].mean()
#                             if np.abs(i_total_last - i_total_initial) < np.abs(i_total_last)*0.001:
#                                 #print("converged")
#                                 #print(i_total_initial)
#                                 #print(i_total_last)
#                                 break
#                         #ここまで伝送戦モデルの計算
                    
#                     #伝送戦モデルでの物理量の平均・総和
#                     """theta_avr_list = np.append(theta_avr_list, theta_int.mean())
#                     I_total_list = np.append(I_total_list, i_cell.sum())"""
                    
#                     theta_avr_list = np.append(theta_avr_list, theta_int[:i].mean())
#                     magnifier = TRANSMISSON_LINE_MODEL_LENGTH/i
#                     #print(magnifier)
#                     print(I)
#                     I = i_cell[:i].sum()*magnifier
#                     I_total_list = np.append(I_total_list, I)
                    

#                     #電位点毎のループここまで

#                 if theta_avr_list[-points_per_period:].mean() - theta_avr_list[-2*points_per_period:].mean() < theta_avr_list.mean()*0.05:
#                     convergenced = True
                
#                 else:
#                     repeat_time+=1
#                     if repeat_time>10:
#                         break
#                     print("not convergenced")
#                     print(theta_avr_list)
#                     print("{}, {}".format(theta_avr_list[-points_per_period:].mean(), theta_avr_list[-2*points_per_period:].mean()))
                    
#                     #被覆率、電流値の時間依存をいったんリセット
#                     theta_avr_list = np.array([])
#                     I_total_list = np.array([])

#                     #他のセル電流値などは持ち越して再度ループ
#             #while抜け
#             if repeat_time >10:
#                 I_phase = np.inf
#                 R_abs = np.inf
                
                    
#         case CaliculationMethod.WAIT_STEADY_STATE:
#             """time_scale = period_time/MINIMUM_PERIOD_NUMBER

#             t = np.linspace(
#                 0,
#                 SIMULATED_PERIOD_NUMBER, 
#                 SIMULATED_PERIOD_NUMBER*points_per_period
#                 ) #[period]"""
#             for E_applied in E_list:
#                 #作用極-電解液間電位
#                 E_electrode:np.ndarray = E_applied - I*solution_registance
                
#                 if np.abs(E_electrode-E_applied) > 1:
#                     raise ValueError(
#                         "Too large iR drop.\nI: {}\nE_applied: {}".format(I, E_applied)
#                     )
#                 #point_time = period_time / points_per_period
                
#                 convergenced = False
#                 repeat_time = 0
                
#                 faraday_current = 0.0
#                 faraday_current_previous_loop = 0.0
#                 double_layer_charge = 0.0
#                 double_layer_charge_previous_loop = 0.0

#                 loop_time = total_time_scale/100 #[sec]

#                 #定常になるまで待つループ
#                 while convergenced == False:
#                     #伝送線のループ
#                     for i in range(TRANSMISSON_LINE_MODEL_LENGTH):
#                         if i == 0:
#                             E_cell[0] = E_electrode
#                         else:
#                             E_cell[i]= E_cell[i-1] - i_cell[i-1] * pore_registance
                        
#                         #Faraday charge
#                         delta_theta_int = \
#                             (theta_star[i]*k1(E_electrode) - theta_int[i]*k2(E_electrode) \
#                             + theta_star[i] * k2r(E_electrode) - theta_int[i] * k1r(E_electrode)) * \
#                             point_time
                        
#                         theta_int[i] += delta_theta_int
#                         theta_star[i] -= delta_theta_int
                        
#                         if theta_int[i] > 1:
#                             theta_int[i] = 1
#                             theta_star[i] = 0

#                         if theta_int[i] < 0:
#                             theta_int[i] = 0
#                             theta_star[i] = 1

#                         i_cell_faraday[i] = (theta_int[i] * k2(E_electrode) - (1-theta_int[i])*k2r(E_electrode) + \
#                             (1-theta_int[i]) * k1(E_electrode) - theta_int[i]*k2(E_electrode)) * F * SURFACE_AREA /TRANSMISSON_LINE_MODEL_LENGTH
                        

#                         #double layer capacitance
#                         delta_E_cap = E_cell[i] - cdl_voltage[i] #[V]
#                         #print(delta_E_cap)
#                         cdl_voltage[i] = E_cell[i] #[V]

#                         delta_charge_cell = delta_E_cap * double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH #[C]
                        
#                         i_cell_double_layer[i] = delta_charge_cell/point_time/TRANSMISSON_LINE_MODEL_LENGTH
                    
#                     #総ファラデー電流計算・更新
#                     faraday_current_previous_loop = faraday_current
#                     faraday_current = i_cell_faraday.sum()

#                     #電気二重層キャパシタンス電荷計算・更新
#                     double_layer_charge_previous_loop = double_layer_charge
#                     double_layer_charge += i_cell_double_layer.sum()*loop_time #電流×時間を積算

#                     repeat_time += 1
#                     #ファラデー電流とキャパシタンスの電荷が収束したか否か
#                     if abs(faraday_current-faraday_current_previous_loop)<abs(faraday_current)*0.01\
#                         and abs(double_layer_charge - double_layer_charge_previous_loop) < abs(double_layer_charge)*0.01:
#                         convergenced = True

#                     #伝送線モデル計算が収束していたら抜ける
#                     if i > 20:
#                         sep = i//2
#                         i_total_initial = i_cell[:i][0:sep].mean()
#                         i_total_last = i_cell[:i][sep:].mean()
#                         if np.abs(i_total_last - i_total_initial) < np.abs(i_total_last)*0.001:
#                             #print("converged")
#                             #print(i_total_initial)
#                             #print(i_total_last)
#                             break

                
#                 #伝送戦モデルでの物理量の平均・総和
#                 """theta_avr_list = np.append(theta_avr_list, theta_int.mean())
#                 I_total_list = np.append(I_total_list, i_cell_faraday.sum() + double_layer_charge/point_time)
#                 """
#                 theta_avr_list = np.append(theta_avr_list, theta_int[:i].mean())
#                 magnifier = TRANSMISSON_LINE_MODEL_LENGTH/i
#                 I = i_cell[:i].sum()*magnifier
#                 I_total_list = np.append(I_total_list, I)
                

#     #phase取得
#     I_last_period = I_total_list[-points_per_period:]
#     index_I_max = np.where(I_last_period == I_last_period.max())[0][0]
#     I_phase = (index_I_max/points_per_period*2 - 1/2)*2*90 #rad to degree 
#     #R絶対値
#     i_width = np.abs(max(I_last_period)-min(I_last_period))
#     R_abs = np.abs(AMPLITUDE/i_width)
    
    
#     print("frequency: {}".format(frequency))
#     print("phase: {}".format(I_phase))
#     print("|R|: {}".format(R_abs))
#     print("--")
    
#     #print(E_electrode)
#     if plot_flag ==True:
        
#         divd_value = 3

#         fig, ax = cmn.create_standard_matplt_canvas()
#         #ax.set_xlim(6, 10)
#         #ax.vlines([6.5, 8.5], [np.min(E_list), np.min(E_list)], [np.max(E_list), np.max(E_list)], linestyles=":", lw =1)
#         ax_right = ax.twinx()
#         ax.plot(t[-len(t)//divd_value:], E_list[-len(t)//divd_value:], c = "#FF5555")
#         ax_right.plot(t[-len(t)//divd_value:], theta_avr_list[-len(t)//divd_value:], c = "#5555FF", ls = ":")

#         fig, ax = cmn.create_standard_matplt_canvas()
#         #ax.set_xlim(6, 10)
#         #ax.vlines([6.5, 8.5], [np.min(E_list), np.min(E_list)], [np.max(E_list), np.max(E_list)], linestyles=":", lw =1)
#         ax_right = ax.twinx()
#         ax.plot(t[-len(t)//divd_value:], E_list[-len(t)//divd_value:], c = "#FF5555")
#         ax_right.plot(t[-len(t)//divd_value:], I_total_list[-len(t)//divd_value:], c = "#5555FF", ls = ":")

#     return (I_phase, R_abs)

@dataclass
class impedance_element:
    abs_resistance: float
    phase: float
    def to_complex(self)->complex:
        return self.abs_resistance*np.cos(self.phase) + 1j*self.abs_resistance*np.sin(self.phase)

    def __repr__(self):
        return self.to_complex().__repr__()
        

def impedance_calculater_from_elements(
    #farraday impedace parameters
    frequency: float,
    
    #Faraday impedance
    pseudo_capacitance: float,
    R_HER: float,
    R_int: float,

    #non-Faraday impedances
    solution_resistanse: float,
    double_layer_capasitance: float,
    pore_resistance: float,

    #other conditons
    lambda_fac: float,

):
    
    def Z_F(frequency, pseudo_capacitance, R_HER, R_int):
        Z_Cint = 1/(1j * frequency*2*np.pi * pseudo_capacitance)
        C_int_R_HER = 1/(1/Z_Cint + 1/R_HER)
        return C_int_R_HER + R_int
    
    Z_cdl = 1/(1j * frequency * 2*np.pi * double_layer_capasitance)
    
    #Z_B= (1j*Z_F(frequency, pseudo_capacitance, R_HER, R_int))\
    #    /(1j-double_layer_capasitance*frequency*Z_F(frequency, pseudo_capacitance, R_HER, R_int))
    Z_B = 1/(1/Z_F(frequency, pseudo_capacitance, R_HER, R_int) + 1/Z_cdl)

    electrode_impedance = np.sqrt(pore_resistance * Z_B)*(1/np.tanh(np.sqrt(pore_resistance/Z_B)*lambda_fac))
    total_impedance = electrode_impedance + solution_resistanse

    return total_impedance

def impedance_calculater_from_elements_no_intermediate(
        #farraday impedace parameters
    frequency: float,
    
    #Faraday impedance
    R_ct: float,

    #non-Faraday impedances
    solution_resistanse: float,
    double_layer_capasitance: float,
    pore_resistance: float,

    #other conditons
    lambda_fac: float,

):
    Z_B= (1j*R_ct)/(1j-double_layer_capasitance*frequency*R_ct)

    electrode_impedance = np.sqrt(pore_resistance * Z_B)*(1/np.tanh(np.sqrt(pore_resistance/Z_B)*lambda_fac))
    total_impedance = electrode_impedance + solution_resistanse

    return total_impedance