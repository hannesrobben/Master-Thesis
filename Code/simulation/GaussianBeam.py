import numpy as np
try:
    from scipy import integrate
except:
    raise Exception('No import of scipy.integrate possible!')


class GaussianBeam():
    
    def __init__(self, Power = 0.2):
        self.P_total = Power
        
    
    def calc_I_total(self, w, r=None, P_total=None):
        '''
        Intensity Calculation

        Parameters
        ----------
        w : Beam Waist of the GaussianBeam
        r : optional - Radius(Array) of the Beam
        P_total : optional Total Power - Set to Init Power

        Returns
        -------
        I_r : Intensity

        '''
        if r == None:    
            r = np.linspace(-2*w, 2*w, 100)
        if P_total == None: P_total = self.P_total
        I_r = 2*P_total/(np.pi*w**2) * np.exp(-2*r**2/w**2)
        I = integrate.simpson(I_r,r)
        return I
    
    def calc_I(self, E, r):
        
        I = integrate.simpson(E*np.conj(E),r)
        return I
        
    def calc_E(self, q, r= None, at_waist=False, z=0, lam = 1550e-9, E0=1):
        '''
        w0 = None -> q-Parameter should be at waist

        Parameters
        ----------
        q : COMPLEX q PARAMETER
        r : RADIAL BOUNDARY CONDITIONS
        z : TYPE, optional
            PROPAGATION DISTANCE. The default is 0.
        lam : TYPE, optional
            WAVELENGTH. The default is 1550e-9.
        E0 : TYPE, optional
            ELECTRICAL FIELD. The default is 1.

        Returns
        -------
        E

        '''
        qi = 1/q
        if np.real(qi) == 0: at_waist=True
        z_R = np.imag(q)
        w_0 = np.sqrt(z_R * lam / np.pi)
        if type(r) is not float:
            if r.any() == None: r= np.linspace(0, 2*w_0, 100)
        
        k= 2*np.pi / lam
        w = np.sqrt(-lam/(np.pi*np.imag(qi)))
        
        if at_waist: # -> RoC inf -> term with RoC = 0
            z=0 # waist
            E = E0 * (w_0 / w) * np.exp(-r**2 / w**2) * np.exp(1j * (k * z - np.arctan(z / z_R)))
        else:
            if z == 0: z=np.real(q)
            RoC = 1/np.real(qi)
            E = E0 * (w_0 / w) * np.exp(-r**2 / w**2) * np.exp(1j * (k * z - np.arctan(z / z_R) + (k * r**2) / (2 * RoC)))
        # E = E0 * (w0 / w) * np.exp(-r**2 / w**2) * np.exp(1j * (k * z - np.atan(z / zR) + (k * r**2) / (2 * RoC)))
        return E
