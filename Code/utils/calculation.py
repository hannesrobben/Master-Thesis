import numpy as np
    
class DataScaler:
    def __init__(self, data, new_min, new_max):
        self.old_min = min(data)
        self.old_max = max(data)
        self.new_min = new_min
        self.new_max = new_max
        self.data    = data

    def scale_number(self, old_number):
        scaled_number = (old_number - self.old_min) / (self.old_max - self.old_min) * (self.new_max - self.new_min) + self.new_min
        return scaled_number
    
def calibrate_Setup(Gauss, Beamwalk, FPI, n_actions = 1000, lam = 1550e-9, min_val=-1, max_val=10):
    '''
    Calculate the Optimal Intensity which the Photodiode could detect
    
    Not to sure if the minimal Intensity is better to be calculated using the 
    different q-Parameter or using the MISSMATCH calculation for every single step
    Parameters

    Parameters
    ----------
    Gauss : TYPE
        DESCRIPTION.
    Beamwalk : TYPE
        DESCRIPTION.
    FPI : TYPE
        DESCRIPTION.
    dl_range : TYPE, optional
        DESCRIPTION. The default is 0.1.
    n_actions : TYPE, optional
        DESCRIPTION. The default is 1000.
    lam : TYPE, optional
        DESCRIPTION. The default is 1550e-9.
    min_val : TYPE, optional
        DESCRIPTION. The default is -1.
    max_val : TYPE, optional
        DESCRIPTION. The default is 10.

    Returns
    -------
    Int_Data : TYPE
        DESCRIPTION.

    '''
    q_list, I_list  = [], []
    delta_l_list = np.linspace(0, Beamwalk.length_motorstage, n_actions)
    for i in range(len(delta_l_list)):
        delta_l = delta_l_list[i]
        q = Beamwalk.calc_q(delta_l)
        RoC_err, A_err, mismatch = FPI.calc_mismatch(q)
        q_list.append(q)
        I_err = 1- abs(mismatch)
        I_list.append(I_err)
    Int_Data = DataScaler(I_list, min_val, max_val)
    
    return Int_Data
