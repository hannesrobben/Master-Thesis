'''
Beamwalk get elements funktioniert nicht
'''
import cmath
import numpy as np
from .sim_args import get_args
# from sim_args import get_args

args=get_args()

class OPT_ELEMENT():
    def __init__(self, 
                 type = 'FreeSpace',
                 Position= 0.0, 
                 x= 0.0,
                 A= 1,
                 B= 0,
                 C= 0,
                 D= 1,
                 lam=1550E-9):
        # Initial Setup without Specification:
        #   Type as FreeSpacce but Without an Addition of any distance.
        self.pos = Position     # Propagation Direction
        self.pos_x = float(x)   # direction perpendicular to propagation
        self.lam = lam          # Wavelength
        self.RayTransfer = []   # ABCD Matrix
        
        self.Elementtype = type
        self.type_change(type, A,B,C,D)
        self.reset()
        
    def reset(self):
        pass
        
    def forward(self, q_in, w_in = None):
        '''
        Function for transform the incoming beam to the modified outgoing beam 
        usign the complex q-Parameter.
        Saving the Output q-Parameter as a Variable of the object.

        Parameters
        ----------
        q_in : COMPLEX Q-PARAMTER INPUT-BEAM
        w_in: BEAM WAIST
            IF OUTCOUPLER==TRUE: CALCULATION OF COMPLEX Q-PARAMETER 

        Returns
        -------
        q : COMPLEX Q-PARAMETER OF OUTPUT-BEAM

        '''
        if self.Elementtype == 'Outcoupler':
            if w_in == None: raise Exception('No Beam Waist w_0 given!')
            self.transform(0, w_in)
            q=self.q
        else:
            self.q =(self.RayTransfer[0][0] * q_in + self.RayTransfer[0][1] )/ \
                    (self.RayTransfer[1][0] * q_in + self.RayTransfer[1][1] )
            q=self.q
        return q
    
    def transform(self, R_in, w_in, lam=None):
        '''
        Function for transform the incoming beam to the modified outgoing beam
        with an Input Radius of Curvature and Input beam waist.
        Saving the complex q-Parameter of the output-beam as an Variable of the
        object.
        
        Parameters
        ----------
        R_in : Input Radius of Curvature of the Gaussian-shape Beam
        w_in : Beam Waist (not diameter)
        lam = None: Option to give modified wavelength to the transformation
        
        Savings
        ----------
        self.q: Complex q-Parameter of the output-beam
        
        Returns
        -------
        R_out : Output Radius of Curvature of the Gaussian-shape Beam
        w_out : Output Beam Waist of the Gaussian-shape Beam
        '''
        if lam==None:
            lam=self.lam
            
        if R_in == 0:
            q_0 = (np.pi * w_in**2 * 1j)  / (self.lam) # no minus because of i above /
        else:
            q_0 = 1 / ( (1/R_in) - (self.lam * 1j) / (np.pi * w_in**2)) 
            
        self.q =(self.RayTransfer[0][0] * q_0 + self.RayTransfer[0][1] ) / \
                (self.RayTransfer[1][0] * q_0 + self.RayTransfer[1][1] )

        [R_out, w_out] = [ 1/np.real(1/self.q) , 
                          np.sqrt( - lam / (np.pi * np.imag(1/self.q)) ) ]
        return R_out, w_out
    
    def type_change(self, type, A,B,C,D):
        # https://www.optowiki.info/glossary/abcd-matrix/
        if type == 'FreeSpace':
            self.RayTransfer = [[1,B],[0,1]] 
        elif type == 'ThinLens':
            # C= - 1/f focal_length for convex/converging lens
            self.RayTransfer = [[1,0],[C,1]] 
        elif type == 'ThickLens':
            # for future
            # https://www.optowiki.info/glossary/abcd-matrix/
            raise Exception('Not yet implemented!')
        elif type == 'Plate':
            # B = d/n 
            raise Warning('B-Parameter not yet defined!')
            self.RayTransfer = [[1, B],[0,1]] 
        elif type == 'FlatMirror':
            self.RayTransfer = [[1,0],[0,1]] 
        elif type == 'CurvedMirror':
            # C = -2 / R_e  -> horizontal: R_e = R * cos(theta) or vertical: R_e = R / cos(theta)
            self.RayTransfer = [[1,0],[C,1]] 
        elif type == 'Outcoupler':
            # 2 * win = waist diameter (=1.21 mm)
            # B =  waist distance (=6.13 mm)
            self.RayTransfer = [[1, -6.13E-3],[0,1]] 
            self.transform(0, 1.21E-3/2)
        else:
            raise Exception('Choose a valid type! type = FreeSpace, ThinLens, ThickLens, Plate, FlatMirror, CurvedMirror')
        
    def get_ABCD(self):
        try:
            # print(self.RayTransfer)
            return self.RayTransfer
        except:
            raise Exception('No ABCD Matrix available!')
            
    def set_ABCD(self, *args, **kwargs):
        for key, val in kwargs.items():
            if key == 'A' or key == 'a':
                self.RayTransfer[0][0] = val
            elif key == 'B' or key == 'b':
                self.RayTransfer[0][1] = val
            if key == 'C' or key == 'c':
                self.RayTransfer[1][0] = val
            elif key == 'D' or key == 'd':
                self.RayTransfer[1][1] = val
                
    def get_pos_x(self): return self.pos_x


class Beamwalk():
    def __init__(self, Version = "Version 0",
                 Adding_Noise = True,
                 noise_factor=0.01,
                 lam = 1550E-9,
                 x1= None, x2= None,
                 d1=None, d2=None ,d3=None, d4=None, f=None):
        self.noise_active = Adding_Noise
        self.noise_factor = noise_factor
        self.reset(lam)
        if Version == "Version 0":
            self.load_Version0()
        elif Version == "Version 1":
            if d1 != None and d2!=None and d3!= None and d4!= None and f!=None:
                self.load_Version1(d1=d1, d2=d2, d3=d3, d4=d4 ,f=f)
            else: 
                raise Warning('Need more information. See Beamwalk Version 1')
        elif Version == "Version L2L":
            if x1 == None and x2== None:
                self.load_Version_l2l(x1=0.015, x2=0)
            elif x1!= None and x2!= None: 
                self.load_Version_l2l(x1, x2)
            else: 
                raise Warning('Need more information. See Beamwalk Version Lens to Lens!')
        elif Version == "Version 2Lens":
            if x1 == None:
                self.load_Version_2lens(x1=0.015)
            else: 
                self.load_Version_2lens(x1)
            # 0.0075 because f2f distance for f=100mm
        else:
            raise Exception("No Version: ", Version, " implemented")
        self.add_Noise()
            
    def reset(self, lam):
        self.list_elements = []
        self.hidden_list = []
        self.q_begin = 0
        self.q_calculation = False
        
        self.length_motorstage = args.length_stage #[m] length of motorstage - 100 mm
        # stage is 185 mm in totals -> equal offset -> offset = 42.5
        self.offset = args.length_offset  #42.5e-3
        
        # 2,5 ghz fsr -> nächster resonanzfrequenz wäre 1549.979965565 ,1550.020034952 
        # scale richtet sich nach Laser noise
        # aktuell 0.1 % fsr ist 
        self.lam = lam + 0.00000000002 * np.random.normal(loc=0.0, scale=0.001)
        
            
    def get_hidden(self):
        '''
        List of the distances
    

        Returns
        -------
        LIST OF N-times d1,d2 Distances
            d1 = OFFSET_1 + Delta_L(Movement Of Motorstage)
            d2 = self.length_motorstage - delta_l + self.offset2
            
            DESCRIPTION.

        '''
        return self.hidden_list
        
    def calc_q(self, delta_l, w_in=None):
        '''
        Calculation of the complex beam-Parameter q of the Optical Path declared
        with the Version.

        Parameters
        ----------
        delta_l : Distance of Stagetravel in m. 
        w_in : Beam-waist w_0, if necessary, else use change_Outcoupler

        Returns
        -------
        q : Complex Beam Parameter at the end of the Beamwalk.
        '''
        if self.q_calculation==True:
            q=self.q_begin
        else:
            q=0
            if w_in is not None:
                q=self.Outcoupler.forward(q, w_in)
            else: raise Exception("No Beam-waist given!")
        
        # Manipulation of the Optical-Elements influences by the Variation delta_l - Version-depending
        self.add_Noise(noise_factor=1e-8)
        self.manipulate_Elements(delta_l)
        
        # Complex Beam Parameter from Outcoupler to Begin FPI
        for i in range(len(self.list_elements)):
            q = self.list_elements[i].forward(q)
            # print(self.list_elements[i].Elementtype)
            # print(q)
        return q
    
    def get_elements(self):
        '''to be deleted?!'''
        list_attribute = dir(self)
        help_list = []
        Position_list = []
        for eachArg in list_attribute:
            if isinstance(eachArg, OPT_ELEMENT):
                 help_list.append(eachArg)
                 Position_list.append(eachArg.pos)
                 
        # use zip() to combine the two lists      
        combined_list = zip(Position_list, help_list)
        # sort the combined list by the float values
        sorted_list = sorted(combined_list)
        # extract the sorted matrices list from the sorted list
        Setup_list = [Element for _, Element in sorted_list]
        return Setup_list    
    
    
    def get_RoC(self, q):
        qi = 1/q
        R_z = 1/np.real(qi)
        return R_z
    
    def get_w0(self, q, lam=1_550e-9):
        qi = 1/q
        return np.sqrt(-lam / (np.pi * np.imag(qi)))
    
    def change_Outcoupler(self, w_in, R_in):
        '''
        w_in = beam waist diameter
        R_in = Waist Distance of Gaussian-shaped Beam
        
        THORLABS
        -------
        waist-diameter = 2*w_in
        waist-distance = R_in
        '''
        self.Outcoupler.set_ABCD(A=1, B=-R_in, C=0, D=1)
        self.Outcoupler.transform(0, w_in/2)
        self.q_begin = self.Outcoupler.q
        if self.list_elements[0]==self.Outcoupler:
            self.list_elements.pop(0)
            self.q_calculation=True
        
    def change_Lens(self, focal_length, n_lens = None):
        '''
        focal_length = focal length of the lens on the motorstage(Version 0)
        
        OPTIONAL: n_lens = refers to the n Lens in direction of the light
                            path. Not usable for Version 0
        '''
        if self.Version == "Version 0":
            self.Lens1.set_ABCD(C=-1/focal_length)
        elif self.Version == "Version L2L":
            if n_lens==0: self.Lens1.set_ABCD(C=-1/focal_length)
            if n_lens==1: self.Lens2.set_ABCD(C=-1/focal_length)
            if n_lens==2: self.Lens3.set_ABCD(C=-1/focal_length)
        elif self.Version == "Version 2Lens":
            if n_lens==0: self.Lens2.set_ABCD(C=-1/focal_length)
            if n_lens==1: self.Lens3.set_ABCD(C=-1/focal_length)
            
        
    def manipulate_Elements(self, delta_l):
        '''
        Manipulating the influenced Distance-Elements of the Beamwalk. 
        delta l in m
        Versionprotocol
        ----------
        Version 0: Manipulating 2 Distances
            Whole Way of Beamtravel (offset+del_l+len_motor-del_l+offset2)

        Parameters
        ----------
        delta_l : length (in m)
        '''
        if  self.Version == 'Version 0' or self.Version == 'Version L2L' or \
            self.Version == 'Version 1' or self.Version == 'Version 2Lens':
            # offset+del_l+len_motor-del_l+offset2
            d1 = self.offset + delta_l
            d2 = self.length_motorstage - delta_l + self.offset
            self.hidden_list.append(d1)
            self.hidden_list.append(d2)
            self.Dist_Motor1.set_ABCD(A=1, B=d1, C=0, D=1)
            self.Dist_Motor2.set_ABCD(A=1, B=d2, C=0, D=1)
        else:
            raise Exception('No Implentation for {}'.format(self.Version))
            
    def add_Noise(self, noise_factor= None):
        
        if self.noise_active != True: 
            return
        if noise_factor == None:
            try:
                noise_factor = self.noise_factor
            except: 
                raise Exception("No noise factor given!")
        if self.Version == 'Version 0':
            D1_Noise=args.dist1 * (1 + np.random.normal(loc = 0.0, scale = noise_factor))
            D2_Noise=args.dist2 * (1 + np.random.normal(loc = 0.0, scale = noise_factor))
            D3_Noise=args.dist3 * (1 + np.random.normal(loc = 0.0, scale = noise_factor))
            D4_Noise=args.dist4 * (1 + np.random.normal(loc = 0.0, scale = noise_factor))
            
            self.Dist_1.set_ABCD(B=D1_Noise)
            self.Dist_2.set_ABCD(B=D2_Noise)
            self.Dist_3.set_ABCD(B=D3_Noise)
            self.Dist_4.set_ABCD(B=D4_Noise)
        elif self.Version == 'Version L2L' or self.Version == 'Version 2Lens' :
            D1_Noise=args.dist1 * (1 + np.random.normal(loc = 0.0, scale = noise_factor))
            D2_Noise=args.dist2 * (1 + np.random.normal(loc = 0.0, scale = noise_factor))
            D3_Noise=args.dist3 * (1 + np.random.normal(loc = 0.0, scale = noise_factor))
            D4_Noise=args.dist4 * (1 + np.random.normal(loc = 0.0, scale = noise_factor))
            
            self.Dist_1.set_ABCD(B=D1_Noise)
            self.Dist_2.set_ABCD(B=D2_Noise)
            self.Dist_3.set_ABCD(B=D3_Noise)
            self.Dist_4.set_ABCD(B=D4_Noise)
        else:
            raise Warning('No Noise for {} available!'.format(self.Version))
    
        
    def load_Version0(self):
        '''
        Version with a Lens on a Motorstage. 
            Range of Motion = 0.1m
            focal_length = 700e-3
            Outcoupler F280APC-1550
            
        SETUP
        -------
        OUTCOUPLER
        MOTORSTAGE incl. Lens
        MIRROR(2x)
        DISTANCE_4 -> END
            dist1: Outcoupler and Begin Motorstage
            dist2: Motorstage and first Mirror 
            dist3: Mirror to Mirror
            dist4: Mirror to Cavity
            
        Savings
        -------
        Each Optical-Element will be Part of the Class
        list_elements: list of elements of the Beam Path
        '''
        self.Version='Version 0'
        
        focal_length = 0.7 # focal length of the Lens f = 400mm
        lam = self.lam
        
        self.Dist_Motor1 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1, lam=lam)        # Begin Motorstage - Lens
        self.Dist_Motor2 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1, lam=lam)        # Lens - End Motorstage
        
        # waist diameter Outcoupler 2*w_in = 3 mm or 1.6mm
        self.Outcoupler = OPT_ELEMENT(type = 'Outcoupler',   Position=0, lam=lam)             # Outcoupler
        self.Lens1 =    OPT_ELEMENT(type = 'ThinLens', C=-1/focal_length, Position=1, lam=lam)        # Lens for Modematching
        self.Mirror1 = OPT_ELEMENT(type = 'FlatMirror',      Position=2, lam=lam)             # BeamWalk Mirror 1
        self.Mirror2 = OPT_ELEMENT(type = 'FlatMirror',      Position=3, lam=lam)             # BeamWalk Mirror 2
        self.Dist_1 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=0.5, lam=lam)           # Distance between Outcoupler and Begin Motorstage
        self.Dist_2 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1.5, lam=lam)           # Distance between Motorstage and first Mirror 
        self.Dist_3 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=2.5, lam=lam)           # Mirror to Mirror
        self.Dist_4 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=3.5, lam=lam)           # Mirror to Cavity
        
        self.list_elements.append(self.Outcoupler)
        self.list_elements.append(self.Dist_1)
        self.list_elements.append(self.Dist_Motor1)
        self.list_elements.append(self.Lens1)
        self.list_elements.append(self.Dist_Motor2)
        self.list_elements.append(self.Dist_2)
        self.list_elements.append(self.Mirror1)
        self.list_elements.append(self.Dist_3)
        self.list_elements.append(self.Mirror2)
        self.list_elements.append(self.Dist_4)
        # self.list_elements.append(Dist_2, Mirror1, Dist_3, Mirror2, Dist_4)
        
        # F240APC-1550
        # self.change_Outcoupler(w_in=1.6e-3/2,R_in=7.26e-3)
        #F280APC-1550 - aktuell verbaut
        self.change_Outcoupler(w_in=3.6e-3/2, R_in=18.07e-3 )  
        # print(self.Outcoupler.get_ABCD())
        self.change_Lens(focal_length)
        self.Dist_1.set_ABCD(B=args.dist1)
        self.Dist_2.set_ABCD(B=args.dist2)
        self.Dist_3.set_ABCD(B=args.dist3)
        self.Dist_4.set_ABCD(B=args.dist4)
    
    
    def load_Version1(self, d1, d2, d3, d4, f):
        '''
        Version with a Lens on a Motorstage - based on Version 1.
            Range of Motion = 0.1m
            focal_length = 400e-3
            Outcoupler F280APC-1550
            
            d1 to d4 : Distances
            f: focal length of Lens
            
        SETUP
        -------
        OUTCOUPLER
        MOTORSTAGE incl. Lens
        MIRROR(2x)
        DISTANCE_4 -> END
            dist1: Outcoupler and Begin Motorstage
            dist2: Motorstage and first Mirror 
            dist3: Mirror to Mirror
            dist4: Mirror to Cavity
            
        Savings
        -------
        Each Optical-Element will be Part of the Class
        list_elements: list of elements of the Beam Path
        '''
        self.Version='Version 1'
        self.load_Version0()
        # F280APC-1550 - aktuell verbaut
        self.change_Outcoupler(w_in=3.6e-3/2, R_in=18.07e-3 )  
        # print(self.Outcoupler.get_ABCD())
        self.change_Lens(f)
        self.Dist_1.set_ABCD(B=d1)
        self.Dist_2.set_ABCD(B=d2)
        self.Dist_3.set_ABCD(B=d3)
        self.Dist_4.set_ABCD(B=d4)
        
    def change_Version1(self, d1, d2, d3, d4, f):
        if self.Version != 'Version 1': return
        self.change_Lens(f)
        self.Dist_1.set_ABCD(B=d1)
        self.Dist_2.set_ABCD(B=d2)
        self.Dist_3.set_ABCD(B=d3)
        self.Dist_4.set_ABCD(B=d4)
        
    def load_Version_l2l(self, x1, x2):
        '''
        Version Lens to Lens.
            One Lens before Motorstage. 
            One Lens on Motorstage. 
            One Lens behind Motorstage
            Range of Motion = 0.1m
            f has to be defined yet
            focal_length = 25e-3
            x1 - Offset for the distance between lens1 and begin motorstage
            x2 - Offset for the distance between end motorstage and lens3
            
        SETUP
        -------
        OUTCOUPLER
        
        Lens TO Lens SETUP incl. MOTORSTAGE - DISTANCES CONTROLLED BY X1 and X2
        
        MIRROR(2x)
        4 DISTANCE, 2 DISTANCE AT MOTORSTAGE, 2 DISTANCE FOR LENS
        
        Savings
        -------
        Each Optical-Element will be Part of the Class
        list_elements: list of elements of the Beam Path
        '''
        self.Version='Version L2L'
        
        focal_length1 = 25e-3 # focal length of the Lens 
        focal_length2 = 25e-3 # focal length of the Lens 
        focal_length3 = 25e-3 # focal length of the Lens 
        
        self.Dist_Motor1 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1)        # Begin Motorstage - Lens
        self.Dist_Motor2 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1)        # Lens - End Motorstage
        
        self.dist_lens_1 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=0.8)      # Distance Lens 1 - Begin Motorstage
        self.dist_lens_2 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1.2)      # Distance End Motorstage - Lens 3
        
        # waist diameter Outcoupler 2*w_in = 3 mm or 1.6mm
        self.Outcoupler = OPT_ELEMENT(type = 'Outcoupler',   Position=0)             # Outcoupler
        self.Lens1 =    OPT_ELEMENT(type = 'ThinLens', C=-1/focal_length1, Position=0.7)        # Lens for Modematching
        self.Lens2 =    OPT_ELEMENT(type = 'ThinLens', C=-1/focal_length2, Position=1)        # Lens for Modematching
        self.Lens3 =    OPT_ELEMENT(type = 'ThinLens', C=-1/focal_length3, Position=1.3)        # Lens for Modematching
        self.Mirror1 = OPT_ELEMENT(type = 'FlatMirror',      Position=2)             # BeamWalk Mirror 1
        self.Mirror2 = OPT_ELEMENT(type = 'FlatMirror',      Position=3)             # BeamWalk Mirror 2
        
        self.Dist_1 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=0.5)           # Distance between Outcoupler and Lens 1
        self.Dist_2 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1.5)           # Distance 
        self.Dist_3 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=2.5)           # Mirror to Mirror
        self.Dist_4 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=3.5)           # Mirror to Cavity
        
        self.list_elements.append(self.Outcoupler)
        # Outcoupler to lens 1
        self.list_elements.append(self.Dist_1)
        self.list_elements.append(self.Lens1)
        self.list_elements.append(self.dist_lens_1)
        #Lens 1 to Begin Motorstage
        self.list_elements.append(self.Dist_Motor1)
        self.list_elements.append(self.Lens2)
        self.list_elements.append(self.Dist_Motor2)
        # end motorstage to lens 3
        self.list_elements.append(self.dist_lens_2)
        self.list_elements.append(self.Lens3)
        # lens 3 to mirror
        self.list_elements.append(self.Dist_2)
        self.list_elements.append(self.Mirror1)
        self.list_elements.append(self.Dist_3)
        self.list_elements.append(self.Mirror2)
        self.list_elements.append(self.Dist_4)
        # self.list_elements.append(Dist_2, Mirror1, Dist_3, Mirror2, Dist_4)
        
        '''muss alles noch aufs setup angepasst werden
        die Linsen, die realen distanzen etc. 
        Die distanzen zwischen Linse-Motorstage und Motorstge Linse werden im ersten Schritt festgesetzt 
        kann in Zukunft für Optimierungsverfahren variable gemacht werden
        ---
        DISTANZEN:
            LENS1-MOTORSTAGE = 5cm offset -> x1
            MOTORSTAGE-LENS3 = 5cm offset -> x2
        '''
        
        # F240APC-1550 
        self.change_Outcoupler(w_in=1.6e-3,R_in=7.26e-3)
        # print(self.Outcoupler.get_ABCD())
        self.change_Lens(focal_length1, n_lens=0)
        self.change_Lens(focal_length2, n_lens=1)
        self.change_Lens(focal_length3, n_lens=2)
        
        self.dist_lens_1.set_ABCD(B=x1)
        self.dist_lens_2.set_ABCD(B=x2)

        self.Dist_1.set_ABCD(B=args.dist1-x1)
        self.Dist_2.set_ABCD(B=args.dist2-x2)
        self.Dist_3.set_ABCD(B=args.dist3)
        self.Dist_4.set_ABCD(B=args.dist4)
        

    def load_Version_2lens(self, x1):
        '''
        Version with 2 Lens.
            One Lens on Motorstage. 
            One Lens behind Motorstage
            Range of Motion = 0.1m
            f has to be defined yet
            focal_length = 25e-3
            x1 - Offset for the distance between end motorstage and lens3
            
        SETUP
        -------
        OUTCOUPLER
        
        Lens TO Lens SETUP incl. MOTORSTAGE - DISTANCES CONTROLLED BY X1 and X2
        
        MIRROR(2x)
        4 DISTANCE, 2 DISTANCE AT MOTORSTAGE, 2 DISTANCE FOR LENS
        
        Savings
        -------
        Each Optical-Element will be Part of the Class
        list_elements: list of elements of the Beam Path
        '''
        self.Version='Version 2Lens'
        
        focal_length = 25e-3 # focal length of the Lens 
        
        
        self.Dist_Motor1 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1)        # Begin Motorstage - Lens
        self.Dist_Motor2 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1)        # Lens - End Motorstage
        
        self.dist_lens_2 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1.2)      # Distance End Motorstage - Lens 3
        
        # waist diameter Outcoupler 2*w_in = 3 mm or 1.6mm
        self.Outcoupler = OPT_ELEMENT(type = 'Outcoupler',   Position=0)             # Outcoupler
        self.Lens2 =    OPT_ELEMENT(type = 'ThinLens', C=-1/focal_length, Position=1)        # Lens for Modematching
        self.Lens3 =    OPT_ELEMENT(type = 'ThinLens', C=-1/focal_length, Position=1.3)        # Lens for Modematching
        self.Mirror1 = OPT_ELEMENT(type = 'FlatMirror',      Position=2)             # BeamWalk Mirror 1
        self.Mirror2 = OPT_ELEMENT(type = 'FlatMirror',      Position=3)             # BeamWalk Mirror 2
        
        self.Dist_1 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=0.5)           # Distance between Outcoupler and Lens 1
        self.Dist_2 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=1.5)           # Distance 
        self.Dist_3 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=2.5)           # Mirror to Mirror
        self.Dist_4 = OPT_ELEMENT(type = 'FreeSpace', B=1,   Position=3.5)           # Mirror to Cavity
        
        self.list_elements.append(self.Outcoupler)
        # Outcoupler to lens 1
        self.list_elements.append(self.Dist_1)
        #Lens 1 to Begin Motorstage
        self.list_elements.append(self.Dist_Motor1)
        self.list_elements.append(self.Lens2)
        self.list_elements.append(self.Dist_Motor2)
        # end motorstage to lens 3
        self.list_elements.append(self.dist_lens_2)
        self.list_elements.append(self.Lens3)
        # lens 3 to mirror
        self.list_elements.append(self.Dist_2)
        self.list_elements.append(self.Mirror1)
        self.list_elements.append(self.Dist_3)
        self.list_elements.append(self.Mirror2)
        self.list_elements.append(self.Dist_4)
        # self.list_elements.append(Dist_2, Mirror1, Dist_3, Mirror2, Dist_4)
        
        '''muss alles noch aufs setup angepasst werden
        die Linsen, die realen distanzen etc. 
        Die distanzen zwischen Linse-Motorstage und Motorstge Linse werden im ersten Schritt festgesetzt 
        kann in Zukunft für Optimierungsverfahren variable gemacht werden
        ---
        DISTANZEN:
            LENS1-MOTORSTAGE = 5cm offset -> x1
            MOTORSTAGE-LENS3 = 5cm offset -> x2
        '''
        
        # F240APC-1550 
        self.change_Outcoupler(w_in=1.6e-3,R_in=7.26e-3)
        # print(self.Outcoupler.get_ABCD())
        self.change_Lens(focal_length, n_lens=0)
        self.change_Lens(focal_length, n_lens=1)
        
        self.dist_lens_2.set_ABCD(B=x1)

        self.Dist_1.set_ABCD(B=args.dist1)
        self.Dist_2.set_ABCD(B=args.dist2-x1)
        self.Dist_3.set_ABCD(B=args.dist3)
        self.Dist_4.set_ABCD(B=args.dist4)
        
        
    def calc_w0(self, q, lam=1550e-9, z=None):
        #z = length of propagation in m
        if z== None:
            w0=self.get_w0(q)
        else: 
            M = np.array([ [1, z] , [0, 1] ])
            q = self.forward_Matrix(q, M)
            qi = 1/q
            w0 = cmath.sqrt(-lam/(np.pi*np.imag(qi)))
        return w0, q
    
    def forward_Matrix(self, q_in, M):
        '''
        Calculation of the complex q-Parameter with an given Input-Parameter 
        and corresponding Ray-Transfer Matrix.
        '''
        q1 = (M[0][0] * q_in + M[0][1] )/ \
             (M[1][0] * q_in + M[1][1] )
        return q1
    
        
if __name__ == "__main__":
    
    Beam = Beamwalk()
    Beam.change_Outcoupler(w_in=3.6e-3, R_in=18.07e-3 )
    q= Beam.q_begin
    w = Beam.get_w0(q)
    RayTransfer = [[1,0],[-1/0.4,1]] 
    list_omega = []
    distance = np.linspace(-0.1,0.1, 100)
    
    for i in range(len(distance)):
        
        w,q1=Beam.calc_w0(q,z=distance[i])
        q_lens = Beam.forward_Matrix(q1, RayTransfer)
        print(1/q1)
        print(1/q_lens)
        # print(np.real(1 / (q+distance[i]) ))
        list_omega.append(w)
    
    import matplotlib.pyplot as plt
    
    plt.plot(distance, list_omega)
        
        
    
    
    
    
    
