import numpy as np 
import math
import scipy
from scipy import constants as const
import scipy.special


class PdhSignalModel:
    '''
    PdhSignalModel simulates a Fabry-Perot interferometer signal.
    
    '''

    def __init__(self, E_0=1, E_1=1, Omega=200e6):
        '''
        Initialize the PdhSignalModel class.
        '''
        # CONSTANTS
        self.c = 299_792_458  # [m/s] velocity of light

        # Electric Field -- can be complex 
        self.E_0 = E_0  # Electric field for the beginning of the Cavity
        self.E_1 = E_1
        
        
        # CAVITY PARAMETERS
        self.L = 0.06  # [m] length of cavity
        self.R = 0.995  # Reflectivity of the mirrors of the FPI
        self.r = np.sqrt(self.R)  # amplitude refelctive coefficient
        self.FSR = self.c / (2 * self.L) # change to FPI simulation parameters

        # MODULATING PARAMETERS
        self.Omega = Omega  # [Hz] Modulationsfrequenz 
        beta = 0.25118864315095796                      # Modulation depth -- defined by EOM (-12dB) should be small b < 1

        self.J_0_beta = scipy.special.j0(beta)  # Bessel Function Order 0
        self.J_1_beta = scipy.special.j1(beta)  # Bessel Function Order 1
            # https://ctan.math.illinois.edu/macros/latex/contrib/hybrid-latex/examples/example-04.pdf 

        # CALCULATION PARAMETERS
        self.f = np.arange(-0.4 * self.FSR, 0.4 * self.FSR, self.FSR / 1e4)
        self.w = 2 * np.pi * self.f


    #### Fabry-Perot Interferometer Calculations ####

    def E_incident(self, t, w=None):
        '''
        Calculate the electric field of the incident beam.

        Parameters:
        - t: Time parameter.
        - w: Angular frequency parameter.

        Returns:
        - E_inc: Electric field of the incident beam.
        '''
        if w is None:
            w = self.w
        E_inc = self.E_0 * np.exp(1j * w * t)
        return E_inc

    def E_reflected(self, t, w=None):
        '''
        Calculate the electric field of the reflected beam.

        Parameters:
        - t: Time parameter.
        - w: Angular frequency parameter.

        Returns:
        - E_ref: Electric field of the reflected beam.
        '''
        if w is None:
            w = self.w
        E_ref = self.E_1 * np.exp(1j * w * t)
        return E_ref

    def calculate_F_w(self, w=None, f = None):
        '''
        Calculate the reflective coefficient.

        Parameters:
        - w: Angular frequency parameter.

        Returns:
        - F_w: Reflective coefficient.
        '''
        if f is None:
            if w is None:
                w = self.w
        else:
            w = 2* np.pi * f
            
        FSR = self.get_FSR()
        
        F_w = (self.r * (np.exp(1j * w / FSR) - 1)) / \
              (1 - self.r**2 * np.exp(1j * w / FSR))
        return F_w

    def get_FSR(self):
        return self.FSR

    #### Modulated Sidebands ####

    def calc_E_inc_mod(self, t, w=None):
        '''
        Calculate the modulated incident beam.

        Parameters:
        - t: Time parameter.
        - w: Angular frequency parameter.

        Returns:
        - E_inc_mod: Modulated incident beam.
        '''
        if w is None:
            w = self.w
        wave_C = self.J_0_beta * np.exp(1j * w * t)
        wave_S1 = self.J_1_beta * np.exp(1j * (w + self.Omega) * t)
        wave_S2 = self.J_1_beta * np.exp(1j * (w - self.Omega) * t)
        E_inc_mod = self.E_0 * (wave_C + wave_S1 - wave_S2)
        return E_inc_mod

    def calc_E_ref_mod(self, t, w=None):
        '''
        Calculate the modulated reflected beam.

        Parameters:
        - t: Time parameter.
        - w: Angular frequency parameter.

        Returns:
        - E_ref_mod: Modulated reflected beam.
        '''
        if w is None:
            w = self.w
        wave_C = self.J_0_beta * np.exp(1j * w * t)
        wave_S1 = self.J_1_beta * np.exp(1j * (w + self.Omega) * t)
        wave_S2 = self.J_1_beta * np.exp(1j * (w - self.Omega) * t)

        F_w = self.calculate_F_w(w)
        F_w_plus = self.calculate_F_w(w=w + 2 * np.pi * self.Omega)
        F_w_minus = self.calculate_F_w(w=w - 2 * np.pi * self.Omega)

        E_ref_mod = self.E_0 * (F_w * wave_C +
                                F_w_plus * wave_S1 -
                                F_w_minus * wave_S2)
        return E_ref_mod

    # Total Power 
    def calc_P_0(self):
        '''
        Calculate the total power.

        Returns:
        - P_0: Total power.
        '''
        P_0 = np.abs(self.E_0)**2
        return P_0

    # Power Carrier-Frequency
    def calc_P_C(self):
        '''
        Calculate the power at the carrier frequency.

        Returns:
        - P_C: Power at the carrier frequency.
        '''
        P_C = self.J_0_beta**2 * self.P_0()
        return P_C

    # Power Sidebands
    def calc_P_S(self):
        '''
        Calculate the power at the sidebands.

        Returns:
        - P_S: Power at the sidebands.
        '''
        P_S = self.J_1_beta**2 * self.P_0()
        return P_S

    def calculate_error(self):
        '''
        Calculate the error.

        Returns:
        - error: Error value.
        '''
        w_m_positive = self.w + (2 * np.pi * self.Omega)
        w_m_negative = self.w - (2 * np.pi * self.Omega)

        F_w = self.calculate_F_w()
        F_w_positive = self.calculate_F_w(w=w_m_positive)
        F_w_negative = self.calculate_F_w(w=w_m_negative)

        error = F_w * np.conjugate(F_w_positive) - \
            np.conjugate(F_w) * F_w_negative

        return error

    def calc_PD_lock(self):
        pass
    
    def calc_PD_trans(self):
        pass

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    
    signal = PdhSignalModel()
    error = signal.calculate_error()
    
    plt.plot(signal.f, np.imag(error))
    # plt.plot(signal.f, np.real(error))
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    