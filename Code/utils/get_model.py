from typing import Optional, List
from pdb import set_trace as stop

import torch.nn as nn


def get_model(
    input_dim: int,
    output_dim: int,
    hidden_layers: Optional[List[int]] = None,
    activation_function = "relu"
):
    """
    Feed-forward network, made of linear layers with ReLU activation functions
    The number of layers, and their size is given by `hidden_layers`.
    """
    # assert init_method in {'default', 'xavier'}

    if hidden_layers is None:
        # linear model
        model = nn.Sequential(nn.Linear(input_dim, output_dim))

    else:
        # neural network
        # there are hidden layers in this case.
        dims = [input_dim] + hidden_layers + [output_dim]
        modules = []
        # print(dims)
        for i, dim in enumerate(dims[:-2]):
            modules.append(nn.Linear(dims[i], dims[i + 1]))
            if activation_function == "relu":
                modules.append(nn.ReLU())
            elif activation_function == "leaky_relu":
                modules.append(nn.LeakyReLU())
            elif activation_function == "sigmoid":
                modules.append(nn.Sigmoid())

        modules.append(nn.Linear(dims[-2], dims[-1]))
        model = nn.Sequential(*modules)
        # stop()

    # n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # print(f'{n_parameters:,} parameters')

    return model


if __name__ == "__main__":
    import gymnasium as gym
    
    env = gym.make('CartPole-v1')
    print(env.observation_space.shape)
    nn_hidden_layers= [128, 256]            # args['nn_hidden_layers'],
    input_dims = env.observation_space.shape[0]
    n_actions = 1000 
    dqn_net = get_model(input_dim=input_dims,
                            output_dim=n_actions,
                            hidden_layers=nn_hidden_layers)