"""
伝送線モデル込みで逐次電流値計算をすることで、周期電位入力に対する電流応答をシミュレートし、位相と抵抗を返す
時定数が周期に対して十分早いとき、各電位での定常状態を計算する。そうでないときは非定常状態の計算をする。
伝送線モデルのセルを一つずつ計算して、収束した段階で計算を打ち止め、収束値から全体値を逆算する。
異常に時定数が長いとき（100秒以上）は即座に値を返す（360 deg, 10^-7 Ohm）。
"""
import numpy as np

from enum import Enum, auto
from copy import deepcopy as copy

import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec
from BFC_libs.physical_constants import F, R, T

class CaliculationMethod(Enum):
    WAIT_STEADY_STATE = auto #各電位で定常になるのを待ち、総電荷と時間から電流値を測定。時定数に対して周期が長いとき
    SIMPLE_SEQUENCE = auto #時間ピリオド毎に単純計算。時定数にたいして周期が短いとき。


def impedance_simulation(
    #input parameters
    frequency: float,
    E_center: float,
    E_amplituide: float,
    conc_substrate: float,
    conc_product: float,
    
    #reaction1
    log_k0_1: float,
    alpha1: float,
    E0_1: float,

    #reacton2
    log_k0_2: float,
    alpha2: float,
    E0_2: float,

    #other conditons
    solution_registance: float=1, #[Ohm]
    double_layer_capacitance:float = 10E-9, #[F]
    pore_registance: float = 10E-4,

    TRANSMISSON_LINE_MODEL_LENGTH: int = 10,
    SURFACE_AREA = 0.15*0.15*np.pi, #default: RDE disk
    plot_flag: bool = False
):
    #数値初期化
    period_time = 1/frequency
    cdl_each_cell = double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH

    SIMULATED_PERIOD_NUMBER = 2
    #TRANSMISSION_LINE_MODEL = 10
    AMPLITUDE = E_amplituide

    MINIMUM_PERIOD_NUMBER = 720
    MAXIMUM_PERIOD_NUMBER = 10000

    #E_center = -1.12

    FRT = F/R/T
    #conc_substrate = 7 #mol/L

    """#reaction1 * + sub -> *int
    alpha1 = 0.8 #[-]
    k0_1 = 2*10E0#[1/s]
    E0_1= -1.09 #[V vs RE]

    #reaction2 *int + sub-> * + product
    alpha2 = 0.3
    k0_2 = 2*10E-1
    E0_2 = -1.10"""

    def k1(E: float): #[1/s]
        return (10**log_k0_1)*conc_substrate*np.exp(FRT*alpha1*-1*(E-E0_1))

    def k2(E: float):#[1/s]
        return (10**log_k0_2)*conc_substrate*np.exp(FRT*alpha2*-1*(E-E0_2))

    def k1r(E: float):#[1/s]
        return (10**log_k0_1)*conc_product*np.exp(FRT*(1-alpha1)*(E-E0_1))

    def k2r(E: float):#[1/s]
        return (10**log_k0_2)*conc_product*np.exp(FRT*(1-alpha2)*(E-E0_2))

    faraday_kinetic_scale:float = np.array([k1(E_center),k2r(E_center),k1r(E_center),k2(E_center)]).max() #[1/sec]
    #print("faraday time scale: {}".format(1/faraday_kinetic_scale))

    #あまりにも時定数が遅い場合、かなり変位が大きくなるような値を返して抜ける
    #if 1/faraday_kinetic_scale > 100:
    #    return (360.0, 10E-8) 

    E_one_point_later = E_center+AMPLITUDE*np.sin(2*np.pi/MINIMUM_PERIOD_NUMBER)
    faraday_kinetic_scale_one_point_later = np.array([k1(E_one_point_later),k2r(E_one_point_later),k1r(E_one_point_later),k2(E_one_point_later)]).max() #[1/sec]
    faraday_current_scale_difference = F * np.abs(faraday_kinetic_scale - faraday_kinetic_scale_one_point_later)
    faraday_registance_scale = (E_one_point_later - E_center) / faraday_current_scale_difference
    #print("R_F: {}".format(faraday_registance_scale))
    double_layer_time_scale = double_layer_capacitance * faraday_registance_scale
    #print("dl time scale: {}".format(double_layer_time_scale))
    total_time_scale = max([1/faraday_kinetic_scale, double_layer_time_scale])

    if total_time_scale*100 < period_time/MINIMUM_PERIOD_NUMBER:
        #キャパシタンス及び反応のタイムスケールよりも十分に一点間の反応が長い場合、定常状態になるのを待つ。伝送線モデルを使わない
        calculation_method = CaliculationMethod.WAIT_STEADY_STATE
        time_scale = period_time/MINIMUM_PERIOD_NUMBER
        print("calculate S.S.")
        #print("time scale = {}".format(time_scale))

    else:
        #そうでない場合、単純に一点ずつ計算する。伝送線モデルを使う。
        calculation_method = CaliculationMethod.SIMPLE_SEQUENCE
        time_scale = total_time_scale
        print("calculate sequentially")
        #print("time scale = {}".format(time_scale))

    points_per_period = int((MINIMUM_PERIOD_NUMBER/100)*period_time/time_scale) if period_time/time_scale>100 else MINIMUM_PERIOD_NUMBER
    print("caliculation point per period: {}".format(points_per_period))
    t = np.linspace(
        0,
        SIMULATED_PERIOD_NUMBER, 
        SIMULATED_PERIOD_NUMBER*points_per_period
        )[1:] #[period]
    point_time = period_time/points_per_period

    #時間方向
    E_diffs = AMPLITUDE*np.cos(t*np.pi*2)
    E_list= E_center + E_diffs 
    
    #空間方向
    E_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, E_center)
    #print(E_list)
    theta_int_init = (k1(E_center)+k2r(E_center))/(k1(E_center)+k2r(E_center)+k1r(E_center)+k2(E_center))

    theta_int = np.full(TRANSMISSON_LINE_MODEL_LENGTH, theta_int_init)
    theta_star = np.full(TRANSMISSON_LINE_MODEL_LENGTH, 1 - theta_int)

    theta_avr_list = np.array([])

    I=0

    cdl_voltage = np.full(TRANSMISSON_LINE_MODEL_LENGTH, AMPLITUDE)
    delta_charge_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, 0.0)
    charge_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH,AMPLITUDE*double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH)
    
    i_cell_faraday = np.full(TRANSMISSON_LINE_MODEL_LENGTH, I/TRANSMISSON_LINE_MODEL_LENGTH)
    i_cell_double_layer = np.full(TRANSMISSON_LINE_MODEL_LENGTH, (AMPLITUDE)*(double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH/100))
    i_cell = np.zeros(TRANSMISSON_LINE_MODEL_LENGTH)

    I_total_list = np.array([])

    
    print("---")
    print(period_time)
    print(points_per_period)
    print(point_time)
    #print("delta E : {}".format(E_one_point_later - E_center))
    #print("faraday kinetic scale change: {} -> {}".format(faraday_kinetic_scale, faraday_kinetic_scale_one_point_later))
    #print("furaday current scale : {}".format(faraday_current_scale_difference))
    #print("faraday registance scale : {}".format(faraday_registance_scale))
    
    #print("total time scale : {}".format(total_time_scale))
    #raise(SyntaxError)


    match(calculation_method):
        case CaliculationMethod.SIMPLE_SEQUENCE:
            #time_scale = total_time_scale
            """points_per_period = int((MINIMUM_PERIOD_NUMBER/100)*period_time*time_scale) if period_time*time_scale>100 else MINIMUM_PERIOD_NUMBER
            t = np.linspace(
                0,
                SIMULATED_PERIOD_NUMBER, 
                SIMULATED_PERIOD_NUMBER*points_per_period
                ) #[period]

            #時間方向
            E_list=AMPLITUDE*np.sin(t*np.pi*2)+E_center 
            
            #空間方向
            E_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, E_center)

            theta_int_init = (k1(E_center)+k2r(E_center))/(k1(E_center)+k2r(E_center)+k1r(E_center)+k2(E_center))

            theta_int = np.full(TRANSMISSON_LINE_MODEL_LENGTH, theta_int_init)
            theta_star = np.full(TRANSMISSON_LINE_MODEL_LENGTH, 1 - theta_int)

            theta_avr_list = np.array([])

            I=E_center/solution_registance/1000

            cdl_voltage = np.full(TRANSMISSON_LINE_MODEL_LENGTH, E_center)
            delta_charge_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, 0)

            i_cell = np.full(TRANSMISSON_LINE_MODEL_LENGTH, I/TRANSMISSON_LINE_MODEL_LENGTH)
            I_total_list = np.array([])"""

            
            convergenced = False
            repeat_time = 0

            while convergenced == False and repeat_time<10:
                for j, E_diff in enumerate(E_diffs):
                    
                    current_convergence = False
                    
                    while(current_convergence == False):
                        
                        #作用極-電解液間電位
                        E_electrode = E_diff - I*solution_registance
                        
                        
                        
                        """if E_electrode * E_diff < 0:
                            E_electrode = 0"""
                        """    if abs(E_diff) < 1E-4:
                                E_electrode = 0
                            else:
                                I *=0.95
                                continue
                        """
                        """if np.abs(I)>abs(AMPLITUDE)/solution_registance:
                            I *=0.95
                            continue"""
                        #print("--")
                        #print("E_diff: {}".format(E_diff))
                        #print("I: {}".format(I))
                        #print("E_electrode: {}".format(E_electrode))
                        if np.abs(I*solution_registance) > 1:
                            print(I_total_list)
                            raise ValueError(
                                "Too large iR drop.\nI: {}\nE_electrode: {}".format(I, E_electrode)

                            )
                        #point_time = period_time / points_per_period
                        
                        #伝送戦モデル
                        i_cell_double_layer = np.zeros(TRANSMISSON_LINE_MODEL_LENGTH)
                        i0_dl = np.zeros(TRANSMISSON_LINE_MODEL_LENGTH)

                        for i in range(TRANSMISSON_LINE_MODEL_LENGTH):
                            if repeat_time == 0:
                                if i == 0:
                                    E_cell[0] = E_electrode
                                else:
                                    E_cell[i]= E_cell[i-1] - i_cell[i-1] * pore_registance

                                    """if E_cell[i] * E_cell[i-1]<0 or E_cell[i-1] ==0:
                                        E_cell[i]=0"""

                            
                            #------capacitance current-------
                            
                            #delta_E_cap = E_cell[i] - cdl_voltage[i] #[V]
                            #print(delta_E_cap)
                            #cdl_voltage[i] = E_cell[i] #[V]

                            #delta_charge_cell[i] = delta_E_cap * double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH #[C]
                            #i_cell_double_layer[i] = delta_charge_cell[i]/point_time



                            #R_total = solution_registance + pore_registance * i
                            """if i == 0:
                            
                            else:
                                i_cell_capasitance_initial_current = (E_cell[i] - cdl_voltage[i])/pore_registance
                                tau = pore_registance * double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH"""
                            
                            #print("E_cell - Vdl: {}".format(E_cell[i] - cdl_voltage[i]))
                            #i_cell_capasitance_initial_current = (E_diff - cdl_voltage[i])/R_total
                            #i_cell_capasitance_initial_current = (E_cell[i] - cdl_voltage[i])/pore_registance
                            """if i != TRANSMISSON_LINE_MODEL_LENGTH-1:
                                i_cell_capasitance_initial_current = (cdl_voltage[i+1] - cdl_voltage[i] - I*pore_registance)
                            else:
                                i_cell_capasitance_initial_current = (E_cell[i] - cdl_voltage[i])/R_total"""
                            #     (I + i_n)   (I + i_n - i_1)       i_n
                            #         →          →                  →
                            #  -----|二二|------|二二|-------- ||---|二二|--
                            #  |      R      |   r    |             r    |
                            #  |           -----    -----              -----
                            #  | E_diff    |   |    |   |              |   |  
                            # ----        Z_F  C   Z_F  C             Z_F  C
                            #  --          |   |    |   |              |   |
                            #  |           -----    -----              -----
                            #  --------------|--------|------||----------|
                            #                                       ← i_n
                            tau = (solution_registance+pore_registance*TRANSMISSON_LINE_MODEL_LENGTH/2) * double_layer_capacitance
                            Idl_0 = (E_diff - cdl_voltage[0])/solution_registance
                            if i ==0:
                                i0_dl[0] = cdl_each_cell*solution_registance/tau*Idl_0
                            elif i==1:
                                i0_dl[1] = cdl_each_cell*Idl_0/tau*(solution_registance + pore_registance - solution_registance*pore_registance*cdl_each_cell/tau)
                            else:
                                i0_dl[i] = (2 - cdl_each_cell*pore_registance/tau)*i0_dl[i-1] - i0_dl[i-2]
                            
                            delta_charge_cell[i] = -1*i0_dl[i] * tau*(np.exp(-point_time/tau) - 1)
                            i_cell_double_layer[i] = delta_charge_cell[i]/point_time
                             
                            
                            """R_over_r = pore_registance/solution_registance
                            sigma_R_over_r = (R_over_r**i - 1) /(R_over_r - 1)
                            i_cell_capasitance_initial_current = (E_diff - cdl_voltage[i])/solution_registance*(1-sigma_R_over_r)
                            tau = solution_registance * double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH
                            delta_charge_cell[i] = -1*i_cell_capasitance_initial_current * tau*(np.exp(-point_time/tau) - 1)
                            i_cell_double_layer[i] = delta_charge_cell[i]/point_time"""
                            

                            """if np.abs(I - i_cell[:i-1].sum()) < np.abs(I)*0.01:
                                break"""
                            """for k in range(10):
                                if k ==0:
                                    if i == 0:
                                        i_cell_capasitance_initial_current = (E_diff - cdl_voltage[i])/solution_registance
                                    else:
                                        if True:
                                            total_iR_drop = (i_cell_double_layer[:i-1].sum())*solution_registance
                                            for l in range(i-1):
                                                total_iR_drop += (i_cell_double_layer[:i-1].sum() - i_cell_double_layer[:l].sum()) * pore_registance
                                        
                                        else:
                                            total_iR_drop = I*solution_registance
                                            for l in range(i-1):
                                                total_iR_drop += (I - i_cell[:l].sum()) * pore_registance
                                        
                                        i_cell_capasitance_initial_current = (E_diff - total_iR_drop - cdl_voltage[i])/R_total
                                else:
                                    if i == 0:
                                        i_cell_capasitance_initial_current = (E_diff - cdl_voltage[i])/solution_registance
                                    else:
                                        total_iR_drop = (i_cell_double_layer[:i-1].sum())*solution_registance
                                        for l in range(i-1):
                                            total_iR_drop += (i_cell_double_layer[:i-1].sum() - i_cell_double_layer[:l].sum()) * pore_registance
                                

                                tau = R_total * double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH
                                #i_cell_capasitance_final_current = i_cell_capasitance_initial_current*np.exp(-1*point_time/tau)
                                #print(i_cell_capasitance_final_current - i_cell_capasitance_initial_current)
                                #print("i_init: {}".format(i_cell_capasitance_initial_current))
                                #delta_charge_cell[i] = i_cell_capasitance_initial_current*R_total*(double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH)*(i_cell_capasitance_final_current-i_cell_capasitance_initial_current)
                                #i_cell_capasitance_avr = (i_cell_capasitance_initial_current + i_cell_capasitance_final_current)/2
                                #delta_charge_cell[i] = i_cell_capasitance_avr * point_time
                                delta_charge_cell[i] = -1*i_cell_capasitance_initial_current * tau*(np.exp(-point_time/tau) - 1)
                                cdl_voltage[i] = (charge_cell[i] + delta_charge_cell[i])/(double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH) 
                                i_cell_double_layer[i] = delta_charge_cell[i]/point_time"""
                            #print("delta charge: {}".format(delta_charge_cell[i]))
                            #この時点ではCdlの電荷更新をしない
                            #収束してから更新
                            #print(delta_charge_cell)




                            #-----Faraday current---------
                            potential_for_faraday = E_cell[i]+E_center - cdl_voltage[i]
                            delta_theta_int = \
                                (theta_star[i]*k1(potential_for_faraday) - theta_int[i]*k2(potential_for_faraday) \
                                + theta_star[i] * k2r(potential_for_faraday) - theta_int[i] * k1r(potential_for_faraday)) * \
                                point_time
                            
                            #theta_list = np.append(theta_list, delta_theta_int)
                            theta_int[i] += delta_theta_int
                            theta_star[i] -= delta_theta_int
                            
                            if theta_int[i] > 1:
                                theta_int[i] = 1
                                theta_star[i] = 0

                            if theta_int[i] < 0:
                                theta_int[i] = 0
                                theta_star[i] = 1
                            

                            i_cell_faraday[i] = (theta_int[i] * k2(potential_for_faraday) - (1-theta_int[i])*k2r(potential_for_faraday) + \
                                (1-theta_int[i]) * k1(potential_for_faraday) - theta_int[i]*k2(potential_for_faraday)) * F * SURFACE_AREA /TRANSMISSON_LINE_MODEL_LENGTH
                            
                            
                            #伝送線セルの電流値計算
                            #i_cell[i] = i_cell_faraday[i] + i_cell_double_layer[i]
                            i_cell[i] = i_cell_double_layer[i]

                            #伝送線モデル計算が収束していたら抜ける
                            """if i > 20:
                                sep = 10
                                i_total_initial = i_cell[:i][-2*sep:-sep].mean()
                                i_total_last = i_cell[:i][-sep:-1].mean()
                                if np.abs(i_total_last - i_total_initial) < np.abs(i_total_last)*0.001:
                                    #print("converged")
                                    #print(i_total_initial)
                                    #print(i_total_last)
                                    break"""
                            #ここまで伝送戦モデルの計算
                        #print(i_cell_double_layer)
                        #print(i_cell_faraday)
                        I_tmp = I
                        I = i_cell_double_layer.sum()# + i_cell_faraday.sum()
                        #print(i0_dl)
                        current_convergence = True
                        cdl_voltage += delta_charge_cell /cdl_each_cell
                        #print("charge: {}".format(charge_cell))
                        #charge_cell = charge_cell + delta_charge_cell
                        """if np.abs(I - I_tmp) < max([abs(I), abs(I_tmp)]) * 0.01 or abs(I)<1E-9:
                            current_convergence = True
                            #Cdlの電荷更新
                            for i, _ in enumerate(charge_cell):
                                #charge_cell[i] += delta_charge_cell[i]
                                cdl_voltage[i] = E_cell[i]"""
                        """else:
                            if np.abs(I - I_tmp) > min([abs(I), abs(I_tmp)]) * 2:
                                I*=0.5
                            if I * solution_registance > 1:
                                I*= 0.1"""
                    #print("charge: {}".format(charge_cell))
                    #伝送戦モデルでの物理量の平均・総和
                    """theta_avr_list = np.append(theta_avr_list, theta_int.mean())
                    I_total_list = np.append(I_total_list, i_cell.sum())"""
                    
                    theta_avr_list = np.append(theta_avr_list, theta_int[:i].mean())
                    #magnifier = TRANSMISSON_LINE_MODEL_LENGTH/i
                    #print(magnifier)
                    #print(i_cell_double_layer)
                    #I = i_cell[:i].sum()*magnifier
                    I = i_cell.sum()
                    I_total_list = np.append(I_total_list, I)
                    

                    #電位点毎のループここまで

                convergenced = True
                """if theta_avr_list[-points_per_period:].mean() - theta_avr_list[-2*points_per_period:].mean() < theta_avr_list.mean()*0.05:
                    convergenced = True
                    repeat_time=0
                else:
                    repeat_time+=1
                    if repeat_time>10:
                        break
                    print("not convergenced")
                    print(theta_avr_list)
                    print("{}, {}".format(theta_avr_list[-points_per_period:].mean(), theta_avr_list[-2*points_per_period:].mean()))
                    
                    #被覆率、電流値の時間依存をいったんリセット
                    theta_avr_list = np.array([])
                    I_total_list = np.array([])"""

                    #他のセル電流値などは持ち越して再度ループ
            #while抜け
            if repeat_time >10:
                I_phase = np.inf
                R_abs = np.inf
                
                    
        case CaliculationMethod.WAIT_STEADY_STATE:
            """time_scale = period_time/MINIMUM_PERIOD_NUMBER

            t = np.linspace(
                0,
                SIMULATED_PERIOD_NUMBER, 
                SIMULATED_PERIOD_NUMBER*points_per_period
                ) #[period]"""
            for E_applied in E_list:
                #作用極-電解液間電位
                E_electrode:np.ndarray = E_applied - I*solution_registance
                
                
                if np.abs(E_electrode-E_applied) > 1:
                    raise ValueError(
                        "Too large iR drop.\nI: {}\nE_applied: {}".format(I, E_applied)
                    )
                #point_time = period_time / points_per_period
                
                convergenced = False
                repeat_time = 0
                
                faraday_current = 0.0
                faraday_current_previous_loop = 0.0
                double_layer_charge = 0.0
                double_layer_charge_previous_loop = 0.0

                loop_time = total_time_scale/100 #[sec]

                #定常になるまで待つループ
                while convergenced == False:
                    #伝送線のループ
                    for i in range(TRANSMISSON_LINE_MODEL_LENGTH):
                        if i == 0:
                            E_cell[0] = E_electrode
                        else:
                            E_cell[i]= E_cell[i-1] - i_cell[i-1] * pore_registance
                        
                        #Faraday charge
                        delta_theta_int = \
                            (theta_star[i]*k1(E_electrode) - theta_int[i]*k2(E_electrode) \
                            + theta_star[i] * k2r(E_electrode) - theta_int[i] * k1r(E_electrode)) * \
                            point_time
                        
                        theta_int[i] += delta_theta_int
                        theta_star[i] -= delta_theta_int
                        
                        if theta_int[i] > 1:
                            theta_int[i] = 1
                            theta_star[i] = 0

                        if theta_int[i] < 0:
                            theta_int[i] = 0
                            theta_star[i] = 1

                        i_cell_faraday[i] = (theta_int[i] * k2(E_electrode) - (1-theta_int[i])*k2r(E_electrode) + \
                            (1-theta_int[i]) * k1(E_electrode) - theta_int[i]*k2(E_electrode)) * F * SURFACE_AREA /TRANSMISSON_LINE_MODEL_LENGTH
                        

                        #double layer capacitance
                        delta_E_cap = E_cell[i] - cdl_voltage[i] #[V]
                        #print(delta_E_cap)
                        cdl_voltage[i] = E_cell[i] #[V]

                        delta_charge_cell = delta_E_cap * double_layer_capacitance/TRANSMISSON_LINE_MODEL_LENGTH #[C]
                        
                        i_cell_double_layer[i] = delta_charge_cell/point_time/TRANSMISSON_LINE_MODEL_LENGTH
                    
                    #総ファラデー電流計算・更新
                    faraday_current_previous_loop = faraday_current
                    faraday_current = i_cell_faraday.sum()

                    #電気二重層キャパシタンス電荷計算・更新
                    double_layer_charge_previous_loop = double_layer_charge
                    double_layer_charge += i_cell_double_layer.sum()*loop_time #電流×時間を積算

                    repeat_time += 1
                    #ファラデー電流とキャパシタンスの電荷が収束したか否か
                    if abs(faraday_current-faraday_current_previous_loop)<abs(faraday_current)*0.01\
                        and abs(double_layer_charge - double_layer_charge_previous_loop) < abs(double_layer_charge)*0.01:
                        convergenced = True

                    #伝送線モデル計算が収束していたら抜ける
                    if i > 20:
                        sep = i//2
                        i_total_initial = i_cell[:i][0:sep].mean()
                        i_total_last = i_cell[:i][sep:].mean()
                        if np.abs(i_total_last - i_total_initial) < np.abs(i_total_last)*0.001:
                            #print("converged")
                            #print(i_total_initial)
                            #print(i_total_last)
                            break

                
                #伝送戦モデルでの物理量の平均・総和
                """theta_avr_list = np.append(theta_avr_list, theta_int.mean())
                I_total_list = np.append(I_total_list, i_cell_faraday.sum() + double_layer_charge/point_time)
                """
                theta_avr_list = np.append(theta_avr_list, theta_int[:i].mean())
                magnifier = TRANSMISSON_LINE_MODEL_LENGTH/i
                I = i_cell[:i].sum()*magnifier
                I_total_list = np.append(I_total_list, I)
                

    #phase取得
    I_last_period = I_total_list[-points_per_period:]
    index_I_max = np.where(I_last_period == I_last_period.max())[0][0]
    index_V_max = np.where(E_diffs[-points_per_period:] == E_diffs[-points_per_period:].max())[0][0]
    I_phase = (index_I_max-index_V_max)/points_per_period*360 #rad to degree 
    #絶対値180deg以上の時、反対周りの角度に変換
    if I_phase > 180:
        I_phase = I_phase-360
    if I_phase < -180:
        I_phase = 360 - I_phase
    #R絶対値
    i_width = np.abs(max(I_last_period)-min(I_last_period))/2
    R_abs = np.abs(AMPLITUDE/i_width)
    
    
    print("frequency: {}".format(frequency))
    print("phase: {}".format(I_phase))
    print("|R|: {}".format(R_abs))
    print("--")
    
    #print(E_electrode)
    if plot_flag ==True:
        
        divd_value = 1

        fig, ax = cmn.create_standard_matplt_canvas()
        #ax.set_xlim(6, 10)
        #ax.vlines([6.5, 8.5], [np.min(E_list), np.min(E_list)], [np.max(E_list), np.max(E_list)], linestyles=":", lw =1)
        ax_right = ax.twinx()
        ax.plot(t[-len(t)//divd_value:], E_list[-len(t)//divd_value:], c = "#FF5555")
        ax_right.plot(t[-len(t)//divd_value:], theta_avr_list[-len(t)//divd_value:], c = "#5555FF", ls = ":")

        fig, ax = cmn.create_standard_matplt_canvas()
        #ax.set_xlim(6, 10)
        #ax.vlines([6.5, 8.5], [np.min(E_list), np.min(E_list)], [np.max(E_list), np.max(E_list)], linestyles=":", lw =1)
        ax_right = ax.twinx()
        ax.plot(t[-len(t)//divd_value:], E_list[-len(t)//divd_value:], c = "#FF5555")
        ax_right.plot(t[-len(t)//divd_value:], I_total_list[-len(t)//divd_value:], c = "#5555FF", ls = ":")

    return (I_phase, R_abs)

def faraday_impedance_simulation(
    #input parameters
    frequency: float,
    E_center: float,
    E_amplituide: float,
    conc_substrate: float,
    conc_product: float,
    
    #reaction1
    log_k0_1: float,
    alpha1: float,
    E0_1: float,

    #reacton2
    log_k0_2: float,
    alpha2: float,
    E0_2: float,

    #other conditons
    SURFACE_AREA = 0.15*0.15*np.pi, #default: RDE disk
    plot_flag: bool = False
):
    """
    returning a tuple of(phase, absolute R)
    the impedance at the potrntial and frequency can be led from
    abs_R*sin(phase) + 
    """
    #数値初期化
    period_time = 1/frequency

    SIMULATED_PERIOD_NUMBER = 5
    #TRANSMISSION_LINE_MODEL = 10
    AMPLITUDE = E_amplituide

    MINIMUM_PERIOD_NUMBER = 720
    MAXIMUM_PERIOD_NUMBER = 10000

    #E_center = -1.12

    FRT = F/R/T
    #conc_substrate = 7 #mol/L


    def k1(E: float): #[1/s]
        return (10**log_k0_1)*conc_substrate*np.exp(FRT*alpha1*-1*(E-E0_1))

    def k2(E: float):#[1/s]
        return (10**log_k0_2)*conc_substrate*np.exp(FRT*alpha2*-1*(E-E0_2))

    def k1r(E: float):#[1/s]
        return (10**log_k0_1)*conc_product*np.exp(FRT*(1-alpha1)*(E-E0_1))

    def k2r(E: float):#[1/s]
        return (10**log_k0_2)*conc_product*np.exp(FRT*(1-alpha2)*(E-E0_2))

    faraday_kinetic_scale:float = np.array([k1(E_center),k2r(E_center),k1r(E_center),k2(E_center)]).max() #[1/sec]
    #print("faraday time scale: {}".format(1/faraday_kinetic_scale))

    #あまりにも時定数が遅い場合、かなり変位が大きくなるような値を返して抜ける
    #if 1/faraday_kinetic_scale > 100:
    #    return (360.0, 10E-8) 

    E_one_point_later = E_center+AMPLITUDE*np.sin(2*np.pi/MINIMUM_PERIOD_NUMBER)
    faraday_kinetic_scale_one_point_later = np.array([k1(E_one_point_later),k2r(E_one_point_later),k1r(E_one_point_later),k2(E_one_point_later)]).max() #[1/sec]
    faraday_current_scale_difference = F * np.abs(faraday_kinetic_scale - faraday_kinetic_scale_one_point_later)
    faraday_registance_scale = (E_one_point_later - E_center) / faraday_current_scale_difference
    #print("R_F: {}".format(faraday_registance_scale))

    if 1/faraday_kinetic_scale*100 < period_time/MINIMUM_PERIOD_NUMBER:
        #キャパシタンス及び反応のタイムスケールよりも十分に一点間の反応が長い場合、定常状態になるのを待つ。伝送線モデルを使わない
        calculation_method = CaliculationMethod.WAIT_STEADY_STATE
        time_scale = period_time/MINIMUM_PERIOD_NUMBER
        print("calculate S.S.")
        #print("time scale = {}".format(time_scale))

    else:
        #そうでない場合、単純に一点ずつ計算する。伝送線モデルを使う。
        calculation_method = CaliculationMethod.SIMPLE_SEQUENCE
        time_scale = 1/faraday_kinetic_scale
        print("calculate sequentially")
        #print("time scale = {}".format(time_scale))

    points_per_period = int((MINIMUM_PERIOD_NUMBER/100)*period_time/time_scale) if period_time/time_scale>100 else MINIMUM_PERIOD_NUMBER
    print("caliculation point per period: {}".format(points_per_period))
    t = np.linspace(
        0,
        SIMULATED_PERIOD_NUMBER, 
        SIMULATED_PERIOD_NUMBER*points_per_period
        ) #[period]
    point_time = period_time/points_per_period

    #時間方向
    E_diffs = AMPLITUDE*np.sin(t*np.pi*2)
    E_list= E_center + E_diffs 
    
    #空間方向
    #print(E_list)
    theta_int_init = (k1(E_center)+k2r(E_center))/(k1(E_center)+k2r(E_center)+k1r(E_center)+k2(E_center))

    theta_int = theta_int_init
    theta_star = 1 - theta_int

    theta_avr_list = np.array([])

    I=0
    I_total_list = np.array([])
    
    print("---")
    #print(period_time)
    #print(points_per_period)
    #print(point_time)

    match(calculation_method):
        case CaliculationMethod.SIMPLE_SEQUENCE:
            #time_scale = total_time_scale

            for j, E_applied in enumerate(E_list):
        
                #-----Faraday current---------
                delta_theta_int = \
                    (theta_star*k1(E_applied) - theta_int*k2(E_applied) \
                    + theta_star * k2r(E_applied) - theta_int * k1r(E_applied)) * \
                    point_time
                
                #theta_list = np.append(theta_list, delta_theta_int)
                theta_int += delta_theta_int
                theta_star -= delta_theta_int
                
                if theta_int > 1:
                    theta_int = 1
                    theta_star = 0

                if theta_int < 0:
                    theta_int = 0
                    theta_star = 1
                

                I = -(theta_int * k2(E_applied) - (1-theta_int)*k2r(E_applied) \
                    +(1-theta_int) * k1(E_applied) - theta_int*k1r(E_applied)) * F * SURFACE_AREA

                I_total_list = np.append(I_total_list, I)
                theta_avr_list = np.append(theta_avr_list, theta_int)
                    
        case CaliculationMethod.WAIT_STEADY_STATE:

            for E_applied in E_list:
                #作用極-電解液間電位

                
                faraday_current = 0.0
                faraday_current_previous_loop = 0.0

                loop_time = 1/faraday_kinetic_scale/100 #[sec]
                repeat_time = 0
                convergenced = False

                #定常になるまで待つループ
                while convergenced == False:
                    #伝送線のループ
                        
                    #Faraday charge
                    delta_theta_int = \
                        (theta_star*k1(E_applied) - theta_int*k2(E_applied) \
                        + theta_star * k2r(E_applied) - theta_int * k1r(E_applied)) * \
                        point_time
                    
                    theta_int += delta_theta_int
                    theta_star -= delta_theta_int
                    
                    if theta_int > 1:
                        theta_int = 1
                        theta_star = 0

                    if theta_int < 0:
                        theta_int = 0
                        theta_star = 1

                                        
                    #総ファラデー電流計算・更新
                    faraday_current_previous_loop = faraday_current
                    faraday_current = -(theta_int * k2(E_applied) - (1-theta_int)*k2r(E_applied) \
                        + (1-theta_int) * k1(E_applied) - theta_int*k1r(E_applied)) * F * SURFACE_AREA
                    repeat_time += 1

                    #ファラデー電流とキャパシタンスの電荷が収束したか否か
                    if abs(faraday_current-faraday_current_previous_loop)<abs(faraday_current)*0.01:
                        convergenced = True

                
                #伝送戦モデルでの物理量の平均・総和
                """theta_avr_list = np.append(theta_avr_list, theta_int.mean())
                I_total_list = np.append(I_total_list, i_cell_faraday.sum() + double_layer_charge/point_time)
                """
                theta_avr_list = np.append(theta_avr_list, theta_int)
                I = faraday_current
                I_total_list = np.append(I_total_list, I)
                

    #phase取得
    I_last_period = I_total_list[-points_per_period:]
    index_I_max = np.where(I_last_period == I_last_period.max())[0][0]
    index_V_max = np.where(E_diffs[-points_per_period:] == E_diffs[-points_per_period:].max())[0][0]
    I_phase = (index_I_max-index_V_max)/points_per_period*360 #rad to degree 
    #絶対値180deg以上の時、反対周りの角度に変換
    if I_phase > 180:
        I_phase = I_phase-360
    if I_phase < -180:
        I_phase = 360 - I_phase
    #R絶対値
    i_width = np.abs(max(I_last_period)-min(I_last_period))/2
    R_abs = np.abs(AMPLITUDE/i_width)
    
    
    print("frequency: {}".format(frequency))
    print("phase: {}".format(I_phase))
    print("|R|: {}".format(R_abs))
    print("--")
    
    #print(E_electrode)
    if plot_flag ==True:
        
        divd_value = 2

        fig, ax = cmn.create_standard_matplt_canvas()
        #ax.set_xlim(6, 10)
        #ax.vlines([6.5, 8.5], [np.min(E_list), np.min(E_list)], [np.max(E_list), np.max(E_list)], linestyles=":", lw =1)
        ax_right = ax.twinx()
        ax.plot(t[-len(t)//divd_value:], E_list[-len(t)//divd_value:], c = "#FF5555")
        ax_right.plot(t[-len(t)//divd_value:], theta_avr_list[-len(t)//divd_value:], c = "#5555FF", ls = ":")

        fig, ax = cmn.create_standard_matplt_canvas()
        #ax.set_xlim(6, 10)
        #ax.vlines([6.5, 8.5], [np.min(E_list), np.min(E_list)], [np.max(E_list), np.max(E_list)], linestyles=":", lw =1)
        ax_right = ax.twinx()
        ax.plot(t[-len(t)//divd_value:], E_list[-len(t)//divd_value:], c = "#FF5555")
        ax_right.plot(t[-len(t)//divd_value:], I_total_list[-len(t)//divd_value:], c = "#5555FF", ls = ":")

    return (I_phase, R_abs)

