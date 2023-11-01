import argparse
import torch
import torch.backends

def get_args(flag=None):
    parser = argparse.ArgumentParser()

    group = parser.add_argument_group('Simulation parameters')
    
    group.add_argument('--length-cav',type=float,default=0.06,
                        help='Length of the cavity.') 
    group.add_argument('--reflectivity-mirrors',type=float,default=0.993,
                        help='Amplitude reflectivity R = r^2(field-reflectivity)') 
    group.add_argument('--focal-length-cav',type=float,default=0.1,
                        help='f=r/2, r=200mm for our mirrors') 
    group.add_argument('--length-stage',type=float,default=0.1,
                        help='Length of the motorized linear stage in m.') 
    group.add_argument('--length-offset',type=float,default=42.5e-3,
                        help='Length of offset for the motorized linear stage in m.') 
    
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
    
    
    group.add_argument('--env-name',type=str,default='FPI_PDH_Lock',
                        help='Simulation of an Fabry-Perót Interferometer with a Pound-Drever-Hall stabilization') 
    group.add_argument('--beta-dB',type=float,default=-12,
                        help='Modulation depth in dezibel') 
    group.add_argument('--beta',type=float,default=0.25118864315095796,
                        help='Modulation depth - EOM dependend') 
    group.add_argument('--n-actions',type=int,default=1000, 
                        help='Number of total possible actions - DC-Stage with 100mm length') 
    
    group.add_argument('--input-dims',type=list,default=[[],[],[]], 
                        help='Dimenstion of the Input for the Agent. Equal to observation_space of Environment') 

    # group.add_argument('--clip-rewards',action='store_true',default=True,
    #                     help='Clip rewards to +-1')

    # group.add_argument('--torch-idx',action='store_true',default=True,
    #                     help='Use torch-type indexing for batches')
    # group.add_argument('--scale',action='store_true',default=False,
    #                     help='Scales frames')

    if flag=='nb':
        args = parser.parse_args(args=[])
    else:
        args = parser.parse_args()

    
    return args