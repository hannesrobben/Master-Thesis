import argparse
import torch
import torch.backends

def get_args(flag=None):
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('system parameters')
    group.add_argument('--cuda', type=int, default=0, 
                       help='ID of GPU to use, -1 for disabled')
    group.add_argument('--mps', type=int, default=0, 
                       help='???')
    group.add_argument('--log-path',type=str,default='logs/training',
                        help='Path for the logs to be saved to')
    group.add_argument('--save-interval', type=int, default=100,
                        help='Interval for data-saving')
    group.add_argument('--eval-interval', type=int, default=100,
                        help='Interval for evaluate the agent!')

    group = parser.add_argument_group('DQN parameters')
    group.add_argument('--total-steps',type=int,default=int(2000),
                        help='Total number of learning steps')
    group.add_argument('--decay-steps',type=int,default=int(100000),
                        help='Number of learning steps for epsilon decay')
    group.add_argument('--rollout-steps',type=int,default=4,
                        help='Number of learning steps for epsilon decay') 
    group.add_argument('--init-eps',type=float,default=1,
                        help='Initial epsilon value')
    group.add_argument('--final-eps',type=float,default=0.05,
                        help='Final epsilon value')
    group.add_argument('--eval-eps',type=float,default=0.05,
                        help='Final epsilon value')
    group.add_argument('--loss-freq',type=int,default=10000,
                        help='Number of steps between loss traking')
    group.add_argument('--target-net-freq',type=int,default=200,
                        help='Number of steps between target network updates')
    group.add_argument('--eval-freq',type=int,default=100000,
                        help='Nuber of steps between evaluations')
    group.add_argument('--plot-freq',type=int,default=100000,
                        help='Nuber of steps between evaluations')
    group.add_argument('--max-grad-norm',type=int,default=50,
                        help='Max gradients norm')
    group.add_argument('--batch-size',type=int,default=64,
                        help='Training batch size')
    group.add_argument('--mem-size',type=int,default=int(1e6),
                        help='Maximum ReplayMemory size')
    group.add_argument('--init-buff-size',type=int,default=int(5e4),
                        help='Initial replay buffer population')
    group.add_argument('--lr',type=float,default=2.5e-4,
                        help='Learning rate')
    group.add_argument('--gamma',type=float,default=.99,
                        help='DQN discount factor')
    group.add_argument('--tau',type=float,default=0.005,
                        help='TAU is the update rate of the target network')
    group.add_argument('--loss-type',type=str,default='Huber',choices=['MSE','Huber'],
                        help='DQN loss type')
    group.add_argument('--dqn-mode',type=str,default='Linear',
                       choices=['Conv_Duel','Duel','Linear'],
                        help='Changes the Network-Structure of the AI')
    group.add_argument('--double',action='store_true',default=False,
                        help='Double DQN')
    group.add_argument('--duel',action='store_true',default=False,
                        help='Dueling DQN')
    group.add_argument('--plot-show',action='store_true',default=False,
                        help='Option show plots or save')

    group = parser.add_argument_group('Environment parameters')
    group.add_argument('--env-name',type=str,default='FPI_PDH_Lock',
                        help='Simulation of an Fabry-Perót Interferometer with a Pound-Drever-Hall stabilization') 
    group.add_argument('--beta-dB',type=float,default=-12,
                        help='Modulation depth in dezibel') 
    group.add_argument('--beta',type=float,default=0.25118864315095796,
                        help='Modulation depth - EOM dependend') 
    group.add_argument('--n-actions',type=int,default=1000, 
                        help='Number of total possible actions - DC-Stage with 100mm length') 
    
    group.add_argument('--input-dims',type=list,default=[[],[],[]], 
                        help='Dimension of the Input for the Agent. Equal to observation_space of Environment') 
    group.add_argument('--input-len',type=int,default=5,
                        help='Length of the Input for the Agent. Equal to number of observation of the Environment') 

    group = parser.add_argument_group('Special DQNNetwork parameters')
    group.add_argument('--layer-1',type=int,default=128,
                        help='Size first hidden layer for linear DQN NeuralNetwork') 
    group.add_argument('--layer-2',type=int,default=256,
                        help='Size second hidden layer for linear DQN NeuralNetwork') 
    
    group.add_argument('--conv-layer-1',type=int,default=32,
                        help='Size first hidden layer for Convolutional Dueling DQN NeuralNetwork') 
    group.add_argument('--conv-layer-2',type=int,default=64,
                        help='Size second hidden layer for Convolutional Dueling DQN NeuralNetwork') 
    group.add_argument('--conv-layer-3',type=int,default=128,
                        help='Size third hidden layer for Convolutional Dueling DQN NeuralNetwork') 
    
    group.add_argument('--duel-layer-1',type=int,default=128,
                        help='Size first hidden layer for Dueling DQN NeuralNetwork') 
    group.add_argument('--duel-layer-2',type=int,default=128,
                        help='Size second hidden layer for Dueling DQN NeuralNetwork') 
    group.add_argument('--duel-layer-3',type=int,default=128,
                        help='Size third hidden layer for Dueling DQN NeuralNetwork') 
    
    # group.add_argument('--clip-rewards',action='store_true',default=True,
    #                     help='Clip rewards to +-1')

    # group.add_argument('--torch-idx',action='store_true',default=True,
    #                     help='Use torch-type indexing for batches')
    # group.add_argument('--scale',action='store_true',default=False,
    #                     help='Scales frames')

    group = parser.add_argument_group('Simulation parameters')
    
    group.add_argument('--length-cav',type=float,default=0.06,
                        help='Length of the cavity.') 
    group.add_argument('--reflectivity-mirrors',type=float,default=0.993,
                        help='Amplitude reflectivity R = r^2(field-reflectivity)') 
    group.add_argument('--focal-length-cav',type=float,default=0.1,
                        help='f=r/2, r=200mm for our mirrors') 
    
    group.add_argument('--dist1',type=float,default=0.1,
                        help='Distance Outcoupler - Modematching-Stage') 
    group.add_argument('--dist2',type=float,default=0.1,
                        help='Distance Modematching Stage - First Mirror') 
    group.add_argument('--dist3',type=float,default=0.2,
                        help='Distance First Mirror - Second Mirror') 
    group.add_argument('--dist4',type=float,default=0.45,
                        help='Distance Second Mirror - Cavity') 
    # group.add_argument('--',type=float,default=0.1,
    #                     help='') 
    
    group.add_argument("--hidden-layers", type=int, nargs="+", 
                        help='Definition of the hidden layers') 
    
    
    if flag=='nb':
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    if torch.cuda.is_available():
            args.device = torch.device('cuda', vars(args)['cuda'])
    elif torch.backends.mps.is_available():
            args.device = torch.device('mps', vars(args)['mps'])
    else:
        args.device = torch.device('cpu')
    # args.device = torch.device('cpu')
    
    
    return args