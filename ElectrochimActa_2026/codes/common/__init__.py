import pandas as pd
import numpy as np
from typing import NewType
from dataclasses import dataclass

import BFC_libs.electrochemistry as ec
import BFC_libs.electrochemistry.biologic as bl

#----------------------------constants---------------------------------
SURFACE_AREA = 0.15 * 0.15 * np.pi  # cm2
HgHgO_TO_RHE = ec.Potential(+0.905) #at 7M KOH
TIME_EVOLUTION_OUTPUT_DIR = "./time_evolution/"
TIME_EVOLUTION_FILE_FORMAT = "te_impedance_E1_{E1}V_Ec_{potential_center}V_Ea{potential_amp}mV.csv"
FIGURE_EXPORT_DIR = "../fig/"

#--------------------keys for dicts and DataFrames---------------------

#--------------------EIS columns----------------------
KEY_FREQUENCY = "frequency"
KEY_REAL_Z = "real Z"
KEY_MINUS_IMAGINARY_Z = "-imaginary Z"
KEY_POTENTIAL = "potential"

#--------------------EIS fitting--------------------
KEY_THETA_AMP = "theta_amp"
KEY_THETA_CENTER = "theta_center"
KEY_K_DECAY = "k_decay"
KEY_K_RDS_AMP = "k_rds_amp"
KEY_C_DL = "C_dl"
KEY_R_PORE = "R_pore"
KEY_LAMBDA_FAC = "lambda_fac"
KEY_R_SOL = "R_sol"
KEY_R_CONTACT = "R_contact"
KEY_C_CONTACT = "C_contact"
KEY_R_CT = "R_ct"
KEY_THETA_POISONED = "theta_poisoned"
KEY_THETA_POISONED_AMP = "theta_poisoned_amp"
KEY_K_DECAY_POISONED = "k_decay_poisoned"

#------------------elementary step paramters-----------------
KEY_ALPHA_1 = "alpha1"
KEY_K0_1 = "k0_1"
KEY_E0_1 = "E1"
KEY_LOG_K0_1 = "log(k0_1)"

KEY_ALPHA_2 = "alpha2"
KEY_K0_2 = "k0_2"
KEY_E0_2 = "E2"
KEY_LOG_K0_2 = "log(k0_2)"

KEY_ALPHA_POISON = "alpha_poison"
KEY_K0_POISON = "k0_poison"
KEY_E0_POISON = "E_poison"

#---------------------others------------------------------
KEY_CURRENT = "current"
KEY_ACTIVE_SITE_DENSITY = "active_site_density"
KEY_LOG_ACTIVE_SITE_DENSITY = "log(Gamma)"

#--------------------------types----------------------------
ElectronTransferCoefficient = NewType("ElectronTransferCoefficient", float)
Alpha = ElectronTransferCoefficient
ReactionRateCoefficient = NewType("ReactionRateCoefficient", float)
k0 = ReactionRateCoefficient

@dataclass
class FitParams():
    name: str
    min: float
    max: float
    initial: float
    def dict(self):
        return {
            "name" : self.name,
            "value": self.initial,
            "min": self.min,
            "max": self.max
    }

def get_fit_data(index_potential: float, fit_data: pd.DataFrame, parameter_names: list[str])->list[float]:
    row = fit_data.loc[index_potential]
    returned_paras_list = []
    for para in parameter_names:
        returned_paras_list.append(row[para+"_fit"])
    return returned_paras_list

def make_current_summary_from_EIS_raw_file(
    EIS_mpt_data: bl.BiologicEISData
)->pd.DataFrame:
    data_array = EIS_mpt_data.data

    eis_potentials:list[float] = []
    steady_currents:list[float] = []
    
    for data_each_potential in data_array:
        eis_potentials.append(data_each_potential.other_data[bl.KEY_POTENTIAL][0])
        
        CA_index = np.where(data_each_potential.frequency == ec.Frequency(0))[0][-1]
        steady_currents.append(data_each_potential.other_data[bl.KEY_CURRENT][0:CA_index//2].mean()/1000) # mA -> A
    #print(eis_potentials)
    potenial_series = pd.Series(eis_potentials, name=KEY_POTENTIAL)
    current_series = pd.Series(steady_currents, name=KEY_CURRENT)
    
    potential_current_df = pd.concat([potenial_series, current_series], axis=1)

    return potential_current_df

    
def get_nearest_potential_from_data(
    target_potential: float,
    obs_data: pd.DataFrame
)->ec.Potential:
    applied_potential = target_potential
    nearest_potential_index = (obs_data["potential"]-applied_potential).abs().idxmin()
    nearest_potential = obs_data["potential"][nearest_potential_index]

    if nearest_potential == np.nan or nearest_potential == None:
        raise ValueError("The potential cannot be found./nInput potential = {}".format(target_potential))
    return ec.Potential(nearest_potential)

def finalize_BO_result_file(
    file_path: str,
    object_column_names: list[str]
):
    ## 作成中
    original_file_name = file_path

    result_file = pd.read_csv(original_file_name)

    sum_array = result_file[object_column_names[0]].values

    for obj_column in object_column_names[1:]:
        sum_array = sum_array + result_file[obj_column].values

    result_sum = pd.Series(
        sum_array,
        index=result_file.index,
        name = "sum"
    )

    processed_dataframe = result_file.assign(sum=result_sum)

    processed_dataframe.to_csv(original_file_name[:-4]+"_prcd.csv", index=False)

def make_complex_array(data_frame: pd.DataFrame):
    complexes = data_frame["real Z"].values + -1 *1J *data_frame["-imaginary Z"].values
    return complexes