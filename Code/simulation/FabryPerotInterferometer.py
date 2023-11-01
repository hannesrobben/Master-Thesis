import cmath
import numpy as np
import cmath
import matplotlib.pyplot as plt
# from GaussianBeam import GaussianBeam
from .GaussianBeam import GaussianBeam
from scipy import integrate

class FPISim():
    '''
    
    Parameters
    ----------
    l : length of the cavity in m                 
    r : Reflectiviy (Intensity) of the Mirrors                 
    f : Focal length of the Mirror (f=r/2)
    hemispherical : BOOL, optional, The default is True.
    r_equal : TYPE, optional The default is True. Reflectivity of the Mirors differs : TYPE
        
    DESCRIPTION:
    Simulation of a Fabry-Perot Intereferometer with the calculation of the Mode 
    Mismatch between an given q-Parameter and the geometrics of the cavity.
    '''
    
    
    def __init__(self, 
                 l, # length of the cavity in m
                 r, # Reflectiviy (Intensity) of the Mirrors
                 f, # Focal length of the Mirror
                 hemispherical = True, 
                 r_equal = True, # Reflectivity of the Mirors differs
                 Adding_Noise= True, # if True Noise is getting added to the length
                 lam = 1550e-9
                 ):
        if hemispherical == False:
            raise Exception('No Implementation for non-hemispherical Cavities!')
        if r_equal == False:
            self.r_2 = float(input('Please provide Reflectiviy of the second Mirror: ') )
            pass
        self.start_length = l
        self.length = self.start_length
        self.noise_active = Adding_Noise
        self.R = r
        self.focal_length = f
        self.RoC = 2* self.focal_length
        self.length_eff = self.length/(1-self.R**2)**(1/2)
        self.n_roundtrip = 2*self.length_eff/lam
        self.add_Noise()
        
        self.fsr = self.calc_FSR()
        self.Finesse = self.calc_Finesse()
        self.Gauss = GaussianBeam()
        
    def calc_Finesse(self): 
        return (np.pi * np.sqrt(self.R)) / (1 - self.R)
    
    def calc_FSR(self):
        c= 299792458
        return c/(self.length*2)
        
    def calc_FWHM(self):
        pass
    
    def add_Noise(self, noise_factor=None):
        # based on l1=l0*(1+a*dt)
        if self.noise_active != True: 
            return
        if noise_factor==None:
            a= 1e-6 # wärmeausdehungskoeffizient
            dT= 0.1 # Temperaturunterschied 0.1 K
            amp1=1  # Verstärkung
            noise_factor  = amp1*(a*dT)
        self.length = self.start_length * (1 + np.random.normal(loc = 0.0, scale = noise_factor))
        self.length_eff = self.length/(1-self.R**2)**(1/2)
    
    def calc_mismatch_old(self, q_in, lam=1550e-9, z=None):
        '''
        Calculation of the Beam Divergence and Area Error of an incident beam 
        and the fundamental mode of the FPI.
        
        Parameters
        ----------
        q_in : complex q-parameter of incident beam
        lam : wavelength [m]  default=1550e-9.

        Returns
        -------
        RoC_err : Beam Divergence Error.
        A_err : Area Error of Beam
        '''
        if z==None: z=self.length
        
        # calc of fundamental mode q-Parameter
        self.calc_q_optim() 
        w = self.calc_w0(self.q_optim)
        r = np.linspace(0, 2*w, 100)
        phi = np.linspace(0,2*np.pi, 100)
        
        E_cav = self.Gauss.calc_E(self.q_optim, r = r)
        E_beam = self.Gauss.calc_E(q_in, r=r)
        
        f1 = E_beam * np.conj(E_cav)
        
        a=integrate.simpson(abs(E_beam)**2,r)
        # print(a)
        f2 =  2*np.pi* a *  2*np.pi*integrate.simpson(abs(E_cav)**2,r)
        MISSMATCH = abs(integrate.simpson(f1*r,r))**2 / f2
        # mm = abs(integrate.simpson(f1,r))**2
        
        # effective Reflectivity
        R_eff = self.R * MISSMATCH
        F=(4*MISSMATCH)/((1-MISSMATCH)**2)
        # Transmittet Intensity
        I = 1/ (1 +(F * (np.sin(2*np.pi*self.length/lam))**2))
        I_sicherung = 1/ (1 + (4*R_eff)/(1-R_eff)**2 * (np.sin(2*np.pi*self.length/lam))**2)
        
        # calc Radius of Curvature
        RoC_cavity=self.calc_RoC(self.q_optim, z=z)
        RoC_in=self.calc_RoC(q_in, z=z)
        
        # calc beamwaist 
        w_cavity= self.calc_w0(self.q_optim, lam)
        w_in= self.calc_w0(q_in, lam)
        
        # calc BeamDivergenceError and AreaError of Beam
        RoC_err=(RoC_in - RoC_cavity)/RoC_cavity
        A_err=((w_in**2-w_cavity**2)/w_cavity**2)
        # Error = 100% * (1 - exp(-2 * ((r_aperture / w_beam)^2)))

        return RoC_err, A_err, I, MISSMATCH
    
    def error_calc(self, q_in, lam=1550e-9, z=None):
        if z==None: z=self.length
        # calc beamwaist 
        w_cavity= self.calc_w0(self.q_optim, lam)
        w_in= self.calc_w0(q_in, lam)
        
        # calc Radius of Curvature
        RoC_cavity=self.calc_RoC(self.q_optim, z=z)
        RoC_in=self.calc_RoC(q_in, z=z)
        
        # calc BeamDivergenceError and AreaError of Beam
        RoC_err=(RoC_in - RoC_cavity)/RoC_cavity
        A_err=(((np.pi*w_in)**2-(np.pi*w_cavity)**2)/(np.pi*w_cavity)**2)
        
        return A_err, RoC_err
        
        
    def calc_mismatch_hard(self, q_in, lam=1550e-9, z=None):
        '''
        Explanation has to be adressed and changed
        
        Parameters
        ----------
        q_in : complex q-parameter of incident beam
        lam : wavelength [m]  default=1550e-9.

        Returns
        -------
        '''
        if z==None: z=self.length
        
        # calc of fundamental mode q-Parameter
        self.calc_q_optim() 
        w_cavity= self.calc_w0(self.q_optim, lam)
        A_err, RoC_err = self.error_calc(q_in)
        
        func = lambda r,phi: self.calc_E(q_in, r=r)*np.conj(self.calc_E(self.q_optim, r = r))*r
        
        E_cav =  lambda r,phi: (self.calc_E(self.q_optim, r = r))**2 *r
        E_beam = lambda r,phi: (self.calc_E(q_in, r=r))**2 *r
        
        funcint_loss,_ = integrate.dblquad(func,0, 2*np.pi, 0, 2*w_cavity )
        funcint_E_beam,_ = integrate.dblquad(E_beam,0, 2*np.pi, 0, 2*w_cavity )
        funcint_E_cav,_ = integrate.dblquad(E_cav,0, 2*np.pi, 0, 2*w_cavity )
        
        MISSMATCH = abs(funcint_loss)**2 / (funcint_E_beam*funcint_E_cav)
        
        
        MISSMATCH2 = abs((self.q_optim - q_in) / (self.q_optim - np.conj(q_in)))**2
        
        return RoC_err, A_err, MISSMATCH
    
    
    def calc_mismatch(self, q_in, lam=1550e-9, z=None):
        '''
        Explanation has to be adressed and changed
        
        Parameters
        ----------
        q_in : complex q-parameter of incident beam
        lam : wavelength [m]  default=1550e-9.

        Returns
        -------
        '''
        if z==None: z=self.length
        
        # calc of fundamental mode q-Parameter
        self.calc_q_optim() 
        w_cavity= self.calc_w0(self.q_optim, lam)
        A_err, RoC_err = self.error_calc(q_in)
        
        MISSMATCH = abs((self.q_optim - q_in) / (self.q_optim - np.conj(q_in)))**2
        
        return RoC_err, A_err, MISSMATCH
    
    
    
    def calc_q_optim(self):
        '''
        Calculation of fundamental mode q-Parameter.

        Returns
        -------
        q_optim : complex q-parameter of fundamental mode of the FPI
        '''
        M = self.calc_internal_Matrix()
        # a = C, b= (D-A), c = -B
        # root 1 = 
        root_pos, root_neg = self.quadratic_roots(M[1][0], (M[1][1]-M[0][0]) , - M[0][1] )
        # print("{} q-parameter 1\n".format(1/root1))
        # print("{} q-parameter 2\n".format(1/root2))
        self.q_optim = root_neg
        return root_neg
    
    def calc_internal_Matrix(self):
        '''
        Calculation of the internal Ray Transfer Marix for a hemispherical FPI. 
        Used for calculate the optimal q-paramter.

        Returns
        -------
        M : RayTransfer-(ABCD-)Matrix of one roundtrip in the FPI
        '''
        self.add_Noise()
        M1 = np.array([ [1, self.length] , [0, 1] ]) # ABCD MAtrix for Travel one time through cavity
        M2 = np.array([ [1, 0] , [-2/self.RoC, 1] ]) # Reflection of curved mirror
        M3 = M1                                      # ABCD MAtrix for Travel second time through cavity
        M4 = np.array([[1,0],[0,1]] )                # Reflection of flat mirror
        M = M4@M3@M2@M1                              # Matrixmultiplication
        return M
    
    def quadratic_roots(self, a, b, c):
        # Calculate the discriminant
        discriminant = cmath.sqrt(b**2 - 4*a*c)
        
        # Calculate the two roots
        root_pos = (-b + discriminant) / (2*a)
        root_neg = (-b - discriminant) / (2*a)
        
        return root_pos, root_neg

    
    def forward_Matrix(self, q_in, M):
        '''
        Calculation of the complex q-Parameter with an given Input-Parameter 
        and corresponding Ray-Transfer Matrix.
        '''
        q1 = (M[0][0] * q_in + M[0][1] )/ \
             (M[1][0] * q_in + M[1][1] )
        return q1
        
    def calc_RoC(self, q, z=None):
        #z = length of propagation in m
        if z== None:
            qi = 1/q
            R_z = 1/np.real(qi)
        else: 
            M = np.array([ [1, z] , [0, 1] ])
            q = self.forward_Matrix(q, M)
            qi = 1/q
            R_z = 1/np.real(qi)
        return R_z
    
    def calc_w0(self, q, lam=1550e-9, z=None):
        #z = length of propagation in m
        if z== None:
            qi = 1/q
            w0 = np.sqrt(-lam/(np.pi*np.imag(qi)))
        else: 
            M = np.array([ [1, z] , [0, 1] ])
            q = self.forward_Matrix(q, M)
            qi = 1/q
            w0 = np.sqrt(-lam/(np.pi*np.imag(qi)))
        return w0

    def calc_E(self, q, r= None, at_waist=False, z=0, lam = 1550e-9, E0=1):
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


if __name__ == "__main__":
    FPI = FPISim(0.06, 0.993, f=0.1)
    #M = FPI.calc_internal_Matrix()
    q= FPI.calc_q_optim()
    M = np.array([ [1, 0.06] , [0, 1] ])
    q =(M[0][0] * q + M[0][1] ) / \
       (M[1][0] * q + M[1][1] )
    print('M= ',M)
    print('q= ', q)
    
    q_vergleich = 0.06+0.0875515138991168j
    mm= FPI.calc_mismatch(q_vergleich)
    print('mismatch= ',mm)

    print(q) 
    wplot = FPI.calc_w0(q)
    print(wplot)
    Rplot = FPI.calc_RoC(q)
    w_vergleich = FPI.calc_w0(q_vergleich)
    Rvergleich = FPI.calc_RoC(q_vergleich)
    P_total =0.2
    r=np.linspace(-wplot, wplot, 100)
    I_r = 2*P_total/(np.pi*wplot**2) * np.exp(-2*r**2/wplot**2)
    I_r2 = 2*P_total/(np.pi*w_vergleich**2) * np.exp(-2*r**2/w_vergleich**2)
    
    delta_I = I_r - I_r2
    RoC_err, A_err, missmatch  = FPI.calc_mismatch(q_in=q_vergleich)
    
    I_r2_vgl = (1-np.abs(RoC_err))*I_r2
    I_vergleich = (1-np.abs(missmatch)) * I_r
    
    vergleich = I_r2-I_vergleich
    # print(vergleich)
    delta = delta_I-I_vergleich
    
    
    # plt.plot(r,vergleich)
    # plt.plot(r,I_vergleich, label='Vergleich')
    # plt.plot(r,I_r, label='Optimum')
    # plt.plot(r,I_r2, label='Q2')
    # plt.plot(r,I_r2_vgl, label='Q2 vergleich')
    # plt.axvline(1/np.e**2)
    # plt.axvline(-1/np.e**2)
    
    plt.legend(['Vergleich','Optimum','Q2', 'Q2 vergleich'])
    
    
    def plot_RoC_mismatch(w,R):
        
        tmax = w/R
        t = np.linspace(-tmax, tmax, 100)
        
        x= R * (np.cos(t)-1)
        y= R * (np.sin(t))
        
        return x, y
    
    
    x1,y1=plot_RoC_mismatch(wplot,Rplot)
    x2,y2=plot_RoC_mismatch(w_vergleich,Rvergleich)
    
    plt.plot(x1,y1)
    plt.plot(x2,y2)
    
    FPI.calc_q_optim()
    
