from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import numpy as np
import random
from collections import deque
import torch

from utils.calculation import calibrate_Setup

from simulation.sim_args import get_args
from simulation.Beamwalk import Beamwalk
from simulation.FabryPerotInterferometer import FPISim
from simulation.PoundDreverHall import PdhSignalModel
from simulation.GaussianBeam import GaussianBeam

simargs = get_args()

class ControlModelEnv(Env):
    def __init__(self, n_actions, n_obs_timesteps=5, version='Version 0', Adding_Noise=True, rewardtype=0, obs_low=10, obs_high=1000):
        '''
        Initialization of the Environment with the Argument of the amount of 
        possible actions.
        Returns
        -------
        None.

        '''
        # PEP 8: Add a docstring to describe the purpose of the function.
        # self.n_obs = 1 # Observations of Environment -> Photodiodes
        self.n_obs_timesteps = n_obs_timesteps  # amount of timesteps of signal_Photodiode
        self.env_version = version
        self.n_actions = n_actions
        self.obs_low = obs_low
        self.obs_high = obs_high

        self.action_space = Discrete(self.n_actions)
        self.observation_space = Box(low=np.float32(np.array([obs_low, obs_low, obs_low, obs_low, obs_low])),
                                     high=np.float32(np.array([obs_high, obs_high, obs_high, obs_high, obs_high])),
                                     dtype=np.float32)
        self.set_seed()
        self.noise_active = Adding_Noise
        self.rewardtype = rewardtype
        self.max_eval_env = 6
        self.reset()

    def reset(self, eval_num=None):
        '''
        Reset the whole Environment to defined start values. Initialize the 
        Simulation of the Fabry-Perot-Interferometer and the optical Beamwalk.
        -------
        STATE -- ARRAY(n,m) - n timesteps with m Signals
            State for the Environment, observable with Photodiodes
        '''
        # Observation State of Environment which is given to the agent to choose an action
        self.state = deque([], maxlen=self.n_obs_timesteps)
        for i in range(self.n_obs_timesteps):
            self.state.append(0)

        # startpoint of the Lens which can be moved
        self.start_pos = random.randint(0, self.n_actions)
        # 100 mm divided by n_actions -> per action, 100/n_actions mm movement is made
        self.pos_relative_lense = self.start_pos * simargs.length_stage / self.n_actions

        # for logging
        self.params_optical_setup = simargs

        # initialization of the Setup
        self.INIT_OPT_SETUP()

        self.eval_state = []
        # info = {}
        # metadata = {}

        self.done_cntr = 0

        return np.array(self.state, dtype=np.float32)  # converts (1,20,2) to (20,2) to get the right shape

    def step(self, action):
        '''
        Chosen Action changes the environment.

        Parameters
        ----------
        action: INT
            The Agent chose an action which is "given" to the Environment
            The Environment reacts on this

        Returns
        -------
        State : ARRAY(n,m) - n timesteps with m Signals 
            Observable state of Environment
        reward : FLOAT
            The Reward is the main Trigger if an action was good or not
        done : bool
            If True -- canceled learning
        info : dict
            Different Informationen of the Environment which has to be stored
            somewhere

        '''
        info_cntr, done = '', False
        next_state = 0
        # postprocessing the action-value
        delta_l = self.postprocessing_action(action)

        # Calculation of q-Parameter with the new Beamwalk-System
        q = self.Beam.calc_q(delta_l)
        # Calculation of the mode-mismatch in the FPI
        RoC_err, Area_err, mismatch = self.FPI.calc_mismatch(q)

        # new state of the Environment
        I_err = 1 - abs(mismatch)
        next_state = self.Int_Data.scale_number(I_err)
        self.state.append(next_state)

        # calculate reward
        reward = self.calc_reward(RoC_err, Area_err, mismatch)
        self.eval_state = [RoC_err, Area_err, I_err]

        if 0 <= RoC_err <= 0.001:
            self.done_cntr += 1
            info_cntr += 'RoC-'
            info = info_cntr
        elif 0 <= Area_err <= 0.001:
            self.done_cntr += 1
            info_cntr += 'Area-'
            info = info_cntr
        else:
            self.done_cntr = 0
            info_cntr = '0-'
            info = info_cntr

        if self.done_cntr >= 5:
            done = True

        return np.array(self.state), reward, done, info

    def calc_reward(self, R_err, A_err, mismatch):
        '''
        Reward-function to calculate the Reward in dependency of PD2-Signal
        if the calculation of reward is changed must also be changed in calculation

        Returns
        -------
        reward : FLOAT
            1 - np.sqrt(R_err**2 + A_err**2)
        '''
        if self.rewardtype == 0:
            reward = 1 - abs(mismatch)
            # print(reward)
            if A_err < 0.0001 and R_err < 0.001:
                reward = 1.5
            elif A_err > 50_000 and R_err > 5000:
                reward = 0
        elif self.rewardtype == 1:
            reward = 1 - 0.1 * np.sqrt(R_err**2 + A_err**2)
        elif self.rewardtype == 2:
            reward = ((R_err**2 + A_err**2) / (R_err + A_err)**2)
        elif self.rewardtype == 3:
            reward = 1 / (A_err + R_err)
        elif self.rewardtype == 4:
            reward = 10 - np.sqrt(R_err**2 + A_err**2)

        if reward < 0:
            reward = 0
        # print('reward: ', reward, '\tA Error: ', A_err, '\tR Error: ', R_err)
        return reward

    def render(self, mode='human'):
        '''
        To visualize with Pygame or something different
        '''
        pass

    def close(self):
        pass

    def get_eval(self, get_Imax=False):
        if not get_Imax:
            x = self.eval_state
        else:
            x = self.eval_state
            x.append(self.I_max)
        return x

    def reset_eval_env(self, eval_num):
        if eval_num > self.max_eval_env:
            raise Exception('Not enough evaluation Environments!')
        x = eval_num + self.max_eval_env  # eval number from 0 to max_eval_env, + max because training_env number = eval_env number
        self.load_Beamwalk(num=x)

    def INIT_OPT_SETUP(self):
        Laserpower = 200e-3  # 200 mW
        self.Gaussian = GaussianBeam(Power=Laserpower)
        self.FPI = FPISim(l=simargs.length_cav,  # length of the cavity
                          r=simargs.reflectivity_mirrors,  # Reflectivity of Mirrors (amplitude)
                          f=simargs.focal_length_cav,  # Focal length in m -> r=200mm
                          Adding_Noise=self.noise_active)
        if self.env_version == "Training":
            x = np.random.randint(0, self.max_eval_env)
            self.load_Beamwalk(num=x)
        elif self.env_version == "Evaluation":
            x = np.random.randint(self.max_eval_env, 2 * self.max_eval_env)
            self.load_Beamwalk(num=x)
        else:
            self.Beam = Beamwalk(Version=self.env_version,
                                 Adding_Noise=self.noise_active)
            if self.env_version == 'Version 0':
                self.Beam.change_Lens(focal_length=0.7)                     # f = 700mm
                self.Beam.change_Outcoupler(w_in=3.6e-3, R_in=18.07e-3 )  #F280APC-1550
            elif self.env_version == 'Version L2L':
                self.Beam.change_Lens(0.1, n_lens=0)
                self.Beam.change_Lens(0.075, n_lens=1)
                self.Beam.change_Lens(0.1, n_lens=2)
                # self.Beam.change_Outcoupler(w_in=3.6e-3, R_in=18.07e-3 )  #F280APC-1550
                self.Beam.change_Outcoupler(w_in=3.0e-3, R_in=15.44e-3)  #F260APC-1550
            elif self.env_version == 'Version 2Lens':
                self.Beam = Beamwalk(Version = 'Version 2Lens', 
                                     Adding_Noise=self.noise_active,
                                     x1 = 0.05)
                self.Beam.Dist_1.set_ABCD(B=0.05)
                self.Beam.Dist_2.set_ABCD(B=0.05)
                self.Beam.Dist_3.set_ABCD(B=0.1)
                self.Beam.Dist_4.set_ABCD(B=0.05)
                self.Beam.change_Lens(0.4, n_lens=0)
                self.Beam.change_Lens(0.125, n_lens=1)
                self.Beam.change_Outcoupler(w_in=2.15e-3, R_in=11.15e-3 )
                # Beam 3 F220
        
        self.Beam.manipulate_Elements(self.pos_relative_lense)
        self.PDH = PdhSignalModel()
        
        self.Int_Data = calibrate_Setup(Gauss = self.Gaussian, 
                                        Beamwalk = self.Beam, 
                                        FPI = self.FPI, 
                                        n_actions = self.n_actions,
                                        lam = 1550e-9,
                                        min_val=self.obs_low, max_val=self.obs_high)
         
        # maximale Intensität bei Setup-optimalem Modenmatching
        self.hidden_I_max = max(self.Int_Data.data)
        self.hidden_I_min = min(self.Int_Data.data)
        # Maximale / OPTIMUM der Intensität möglich mit dem Setup
        
    def postprocessing_action(self, action):
        # a method to calculate the new delta_l for die lense
        ratio=self.Beam.length_motorstage/self.n_actions
        delta_length = ratio*action
        return delta_length
        
    def set_seed(self, seed=0) -> None:
        '''
        To ensure reproducible runs we fix the seed for different libraries
        '''
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

        # Deterministic operations for CuDNN, it may impact performances
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

        self.action_space = Discrete(self.n_actions, seed = seed)
        self.observation_space = Box(low = np.float32(np.array([self.obs_low,self.obs_low,self.obs_low,self.obs_low,self.obs_low])),
                                     high= np.float32(np.array([self.obs_high,self.obs_high,self.obs_high,self.obs_high,self.obs_high])),
                                     dtype=np.float32,
                                     seed = seed)
        
    def load_Beamwalk(self, num):
        if type(num) is not int and type(num) is not float:  raise Exception('Number has to be an Integer!')
        if num == 0:
            self.Beam = Beamwalk(Version = 'Version 0', 
                                 Adding_Noise=self.noise_active)
            self.Beam.change_Lens(focal_length=0.75)                 # f = 750mm
            self.Beam.change_Outcoupler(w_in=3.0e-3, R_in=15.44e-3 ) #F260APC-1550
            # Setup w F260
            
        elif num == 1:
            self.Beam = Beamwalk(Version = 'Version 1', 
                                 Adding_Noise=self.noise_active,
                                 d1= 0.2  ,
                                 d2= 0.1  ,
                                 d3= 0.1  ,
                                 d4= 0.05 ,
                                 f=  0.4 )
            self.Beam.change_Outcoupler(w_in=1.6e-3, R_in=7.26e-3 )
            # Beam 4 F240            
        elif num == 2:
            self.Beam = Beamwalk(Version = 'Version 2Lens', 
                                 Adding_Noise=self.noise_active,
                                 x1 = 0.05)
            self.Beam.Dist_1.set_ABCD(B=0.05)
            self.Beam.Dist_2.set_ABCD(B=0.05)
            self.Beam.Dist_3.set_ABCD(B=0.1)
            self.Beam.Dist_4.set_ABCD(B=0.05)
            self.Beam.change_Lens(0.4, n_lens=0)
            self.Beam.change_Lens(0.125, n_lens=1)
            self.Beam.change_Outcoupler(w_in=2.15e-3, R_in=11.15e-3 )
            # Beam 3 F220
        elif num == 3:
            self.Beam = Beamwalk(Version = 'Version 2Lens', 
                                 Adding_Noise=self.noise_active,
                                 x1 = 0.05)
            self.Beam.Dist_1.set_ABCD(B=0.2)
            self.Beam.Dist_2.set_ABCD(B=0.1)
            self.Beam.Dist_3.set_ABCD(B=0.1)
            self.Beam.Dist_4.set_ABCD(B=0.05)
            self.Beam.change_Lens(0.1, n_lens=0)
            self.Beam.change_Lens(0.075, n_lens=1)
            self.Beam.change_Outcoupler(w_in=3.0e-3, R_in=15.44e-3 ) 
            # Beam 4 F260
        elif num == 4:
            self.Beam = Beamwalk(Version = 'Version L2L', 
                                 Adding_Noise=self.noise_active)
        elif num == 5:
            self.Beam = Beamwalk(Version = 'Version L2L', 
                                 Adding_Noise=self.noise_active)
            self.Beam.change_Lens(0.05, n_lens=0)
            self.Beam.change_Lens(0.125, n_lens=1)
            self.Beam.change_Lens(0.125, n_lens=2)
            self.Beam.change_Outcoupler(w_in=1.21e-3, R_in=6.13e-3)  #F110APC-1550
            # Beam L2L F110
        elif num == 6:     
            self.Beam = Beamwalk(Version = 'Version 1', 
                                 Adding_Noise=self.noise_active,
                                 d1= 0.1   ,
                                 d2= 0.1   ,
                                 d3= 0.3  ,
                                 d4= 0.2   ,
                                 f= 0.7    )
            self.Beam.change_Outcoupler(w_in=3.6e-3, R_in=18.07e-3 ) 
            # Beam 1 F280
        elif num == 7:
            self.Beam = Beamwalk(Version = 'Version 1', 
                                 Adding_Noise=self.noise_active,
                                 d1= 0.03   ,
                                 d2= 0.18   ,
                                 d3= 0.25   ,
                                 d4= 0.1   ,
                                 f= 0.5    )
            self.Beam.change_Outcoupler(w_in=2.15e-3, R_in=11.15e-3 )
            # Beam 2 F220
        elif num == 8:
            self.Beam = Beamwalk(Version = 'Version 2Lens', 
                                 Adding_Noise=self.noise_active,
                                 x1 = 0.05)
            self.Beam.Dist_1.set_ABCD(B=0.03)
            self.Beam.Dist_2.set_ABCD(B=0.18)
            self.Beam.Dist_3.set_ABCD(B=0.25)
            self.Beam.Dist_4.set_ABCD(B=0.1)
            self.Beam.change_Lens(0.05, n_lens=0)
            self.Beam.change_Lens(0.4, n_lens=1)
            self.Beam.change_Outcoupler(w_in=1.21e-3, R_in=6.13e-3)  #
            #Beam 2L F110APC-1550
        elif num == 9:
            self.Beam = Beamwalk(Version = 'Version 2Lens', 
                                 Adding_Noise=self.noise_active,
                                 x1 = 0.05)
            self.Beam.Dist_1.set_ABCD(B=0.1)
            self.Beam.Dist_2.set_ABCD(B=0.1)
            self.Beam.Dist_3.set_ABCD(B=0.2)
            self.Beam.Dist_4.set_ABCD(B=0.45)
            self.Beam.change_Lens(0.125, n_lens=0)
            self.Beam.change_Lens(0.5, n_lens=1)
            self.Beam.change_Outcoupler(w_in=3.0e-3, R_in=15.44e-3 ) 
            # Beam 2L F260
        elif num == 10:
            self.Beam = Beamwalk(Version = 'Version L2L', 
                                 Adding_Noise=self.noise_active)
            self.Beam.change_Lens(0.075, n_lens=0)
            self.Beam.change_Lens(0.3, n_lens=1)
            self.Beam.change_Lens(0.075, n_lens=2)
            self.Beam.change_Outcoupler(w_in=2.15e-3, R_in=11.15e-3 )  
            # Beam L2L F220.2
        elif num == 11:
            self.Beam = Beamwalk(Version = 'Version L2L', 
                                 Adding_Noise=self.noise_active)
            self.Beam.change_Lens(0.1, n_lens=0)
            self.Beam.change_Lens(0.125, n_lens=1)
            self.Beam.change_Lens(0.1, n_lens=2)
            self.Beam.change_Outcoupler(w_in=3.6e-3, R_in=18.07e-3 )   
            # Beam L2L F280
        else:
            raise Exception('No Beamwalk for Number ', num, ' implemented!')
        
        