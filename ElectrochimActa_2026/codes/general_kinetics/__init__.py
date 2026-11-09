
import numpy as np

from enum import Enum, auto
from dataclasses import dataclass

#import BFC_libs.common as cmn
import BFC_libs.electrochemistry as ec
from BFC_libs.physical_constants import F, R, T

import common

FRT = F/R/T

def Butler_Volmer_cath(
    k0: common.ReactionRateCoefficient,
    alpha: common.ElectronTransferCoefficient, 
    E0: ec.Potential|ec.PotentialArray,
    applied_E: ec.Potential,
)->float|np.ndarray:
    delta_E_tmp = (applied_E-E0)
    
    
    if type(delta_E_tmp) is ec.Potential:
        delta_E = delta_E_tmp.value
    elif type(delta_E_tmp) is ec.PotentialArray:
        delta_E = delta_E_tmp.float_array()
    elif isinstance(delta_E_tmp, np.ndarray):
        delta_E = delta_E_tmp
    elif isinstance(delta_E_tmp, np.float64):
        delta_E = delta_E_tmp
    else:  
        raise TypeError("applied_E must be ec.Potential or ec.PotentialArray")
    return k0*np.exp(-1*FRT*alpha*delta_E)

def Butler_Volmer_ano(
    k0: common.ReactionRateCoefficient,
    alpha: common.ElectronTransferCoefficient, 
    E0: ec.Potential,
    applied_E: ec.Potential|ec.PotentialArray,
)->float|np.ndarray:
    delta_E_tmp = (applied_E-E0)

    if type(delta_E_tmp) is ec.Potential:
        delta_E = delta_E_tmp.value
    elif type(delta_E_tmp) is ec.PotentialArray:
        delta_E = delta_E_tmp.float_array()
    elif isinstance(delta_E_tmp, np.ndarray):
        delta_E = delta_E_tmp
    elif isinstance(delta_E_tmp, np.float64):
        delta_E = delta_E_tmp
    else:  
        raise TypeError("applied_E must be ec.Potential or ec.PotentialArray\n"
                        "Used type is {}".format(type(delta_E_tmp)))

    return k0*np.exp(FRT*alpha*delta_E)

class Direction(Enum):
    ANODIC = 1
    CATHODIC = 2

@dataclass
class ElementaryStep_ET:
    alpha: common.ElectronTransferCoefficient
    k0: common.ReactionRateCoefficient
    E0: ec.Potential
    forward_direction: Direction
    """
    assumptions:
    k0 includes Nernstian concentration terms.
    So k0 is an apperant value.
    Rates follow Butller-Volmer eq.
    The sum of alpha_forward and alpha_backward is 1.
    The member of this class "alpha" means alpha of the forward reaction.
    """
    def _rate(
        self,
        applied_potential: ec.Potential|ec.PotentialArray,
        forward_flag: bool
    ):
        if forward_flag == True:
            match self.forward_direction:
                case Direction.ANODIC:
                    func = Butler_Volmer_ano
                case Direction.CATHODIC:
                    func = Butler_Volmer_cath
        else:
            match self.forward_direction:
                case Direction.ANODIC:
                    func = Butler_Volmer_cath
                case Direction.CATHODIC:
                    func = Butler_Volmer_ano
        
        return func(
            k0 = self.k0,
            alpha = self.alpha if forward_flag else common.ElectronTransferCoefficient(1-self.alpha),
            E0 = self.E0,
            applied_E= applied_potential
        )
    
    def _overpotential(
        self,
        applied_potential: ec.Potential|ec.PotentialArray
    ) -> ec.Potential|ec.PotentialArray:
        return (
            applied_potential - self.E0 
            if self.forward_direction == Direction.ANODIC 
            else -(applied_potential - self.E0)
        )
    
    # def rate_anodic(
    #     self,
    #     applied_potential: ec.Potential|ec.PotentialArray
    # ):
    #     if self.forward_direction == Direction.ANODIC:
    #         return self._rate(applied_potential, True)
    #     else:
    #         return self._rate(applied_potential, False)
    
    # def rate_cathodic(
    #     self,
    #     applied_potential: ec.Potential|ec.PotentialArray
    # ):
    # rate_cathodic: 分岐が逆で、常にアノード側の速度定数を返します(general_kinetics:120-127)。どの図のコードからも呼ばれていません。
    #     if self.forward_direction == Direction.CATHODIC:
    #         return self._rate(applied_potential, False)
    #     else:
    #         return self._rate(applied_potential, True)

    def accelaration_fw(
        self,
        applied_potential: ec.Potential|ec.PotentialArray
    ):
        """
        Calculate the acceleration for the forward reaction rate.
        
        Args:
            applied_potential: Applied potential or potential array
            
        Returns:
            Accelaration of the forward reaction rate
        """
        rate = self._rate(applied_potential, True)

        if isinstance(applied_potential, ec.Potential):
            overpotential:float = self._overpotential(applied_potential).value
        else:
            overpotential:np.ndarray = self._overpotential(applied_potential).float_array()
        
        prefactor:float|np.ndarray = self.alpha * F / R / T
        
        return prefactor * rate

    def accelaration_bw(
        self,
        applied_potential: ec.Potential|ec.PotentialArray
    ):
        """
        Calculate the acceleration for the backward reaction rate.
        
        Args:
            applied_potential: Applied potential or potential array
            
        Returns:
            Accelaration of the backward reaction rate
        """
        rate = self._rate(applied_potential, False)
        
        if isinstance(applied_potential, ec.Potential):
            overpotential:float = -1*self._overpotential(applied_potential).value
        else:
            overpotential:np.ndarray = -1*self._overpotential(applied_potential).float_array()
        
        prefactor:float|np.ndarray = -1*(1 - self.alpha) * F/ R / T
        
        return prefactor * rate


    def rate_fw(
        self,
        applied_potential: ec.Potential|ec.PotentialArray
    ):
        return self._rate(applied_potential, True)
    
    def rate_bw(
        self,
        applied_potential: ec.Potential|ec.PotentialArray
    ):
        return self._rate(applied_potential, False)


