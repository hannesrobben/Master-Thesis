import os
import json
import numpy as np

import torch as T
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from utils.args import get_args
from utils.get_model import get_model

args = get_args()
device = args.device

class ReplayBuffer:
    def __init__(self, max_size, input_shape, n_actions):
        # Initialize memory buffer for experience replay
        self.memory_size = max_size
        self.memory_counter = 0
        self.state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.new_state_memory = np.zeros((self.memory_size, *input_shape), dtype=np.float32)
        self.action_memory = np.zeros(self.memory_size, dtype=np.int64)
        self.reward_memory = np.zeros(self.memory_size, dtype=np.float32)
        self.terminal_memory = np.zeros(self.memory_size, dtype=bool)

    # Store transitions in the replay memory
    def store_transition(self, state, action, reward, next_state, done):
        index = self.memory_counter % self.memory_size
        self.state_memory[index] = state
        self.action_memory[index] = action
        self.reward_memory[index] = reward
        self.new_state_memory[index] = next_state
        self.terminal_memory[index] = done
        self.memory_counter += 1

    # Sample a batch from the replay memory
    def sample_buffer(self, batch_size):
        max_mem = min(self.memory_counter, self.memory_size)
        batch = np.random.choice(max_mem, batch_size, replace=False)
        states = self.state_memory[batch]
        actions = self.action_memory[batch]
        rewards = self.reward_memory[batch]
        next_state = self.new_state_memory[batch]
        dones = self.terminal_memory[batch]
        return states, actions, rewards, next_state, dones

class DQN(nn.Module):
    # Define the structure of the Deep Q-Network
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(*input_dims, fc1_dims)
        self.layer2 = nn.Linear(fc1_dims, fc2_dims)
        self.layer3 = nn.Linear(fc2_dims, n_actions)

        # Initialize weights
        self.layer1.weight.data.normal_(0, 0.1)
        self.layer2.weight.data.normal_(0, 0.1)
        self.layer3.weight.data.normal_(0, 0.1)

    # Define the forward pass through the network
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return self.layer3(x)

class DQNAgent:
    """
    Simple DQNAgent that performs Q-value calculations from observations
    and selects actions based on Q-values.
    """
    def __init__(self, input_dims, dqn_mode="Linear", gamma=args.gamma, tau=args.tau,
                 n_actions=args.n_actions, eps_min=args.final_eps, replace=args.target_net_freq,
                 eps_decay=args.decay_steps, epsilon_start=args.init_eps, device=args.device,
                 mem_size=args.mem_size, lr: float = args.lr, activation_function="relu",
                 nn_hidden_layers=[250, 500], batch_size=args.batch_size, normalize_state=True,
                 n_steps_warm_up_memory=100, max_grad_norm=args.max_grad_norm, freq_steps_train=1,
                 replacement_kind='hard', log_dir=None):
        # Initialization of the DQNAgent
        self.n_steps_warm_up_memory = n_steps_warm_up_memory
        self.epsilon_start = epsilon_start
        self.eps_decay = eps_decay
        self.eps_min = eps_min
        self.device = device
        self.mem_size = mem_size
        self.gamma = gamma
        self.tau = tau
        self.input_dims = input_dims
        self.n_actions = n_actions
        self.replace_target_count = replace
        self.learning_rate = lr
        self.activation_func = activation_function
        self.nn_hidden_layers = nn_hidden_layers
        self.batch_size = batch_size
        self.normalize_state = normalize_state
        self.max_grad_norm = max_grad_norm
        self.dqn_mode = dqn_mode
        self.freq_steps_train = freq_steps_train
        self.replacement_kind = replacement_kind
        if log_dir is not None:
            self.log_dir = log_dir
        else:
            print("/nNo logging")
        self.reset()
        self.hyperparameters = {
            # Hyperparameters for the DQNAgent
        }

    def reset(self):
        self.action_space = [i for i in range(self.n_actions)]                  #
        self.learn_step_counter = 0                     # Zähler zum Nachvollziehen des Lernens?
        self.epsilon=self.epsilon_start
        self.writer = SummaryWriter(self.log_dir)
        
        self.memory = ReplayBuffer(max_size= self.mem_size, 
                                   input_shape= self.input_dims, 
                                   n_actions= self.n_actions)
        if self.dqn_mode == "Linear":
            self.policy_net = get_model(input_dim=self.input_dims[0],
                                    output_dim=self.n_actions,
                                    hidden_layers=self.nn_hidden_layer,
                                    activation_function = self.act_func)
            self.tgt_net = get_model(input_dim=self.input_dims[0],
                                    output_dim=self.n_actions,
                                    hidden_layers=self.nn_hidden_layer,
                                    activation_function = self.act_func)
        else:
            raise Exception('No Network parameters set! Please specify your desired Network')
        
        self.policy_net.to(device=self.device)
        self.tgt_net.to(device=self.device)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.loss = nn.MSELoss() # noch ändern zu args parameter
        self.log_loss = []

    def store_transition(self, state, action, reward, next_state, done):
        # Store the transition (state, action, reward, next_state, done) in the replay memory
        self.memory.store_transition(state, action, reward, next_state, done)
    
    def sample_memory(self):
        # Sample a batch from the replay memory to perform learning
        state, action, reward, new_states, done = self.memory.sample_buffer(self.batch_size)
        states = T.tensor(state)
        rewards = T.tensor(reward)
        dones = T.tensor(done)
        actions = T.tensor(action)
        next_states = T.tensor(new_states)
        return states, actions, rewards, next_states, dones
        
    def calc_loss(self):
        # Calculate the loss for the Q-network
        # Extract necessary components from the sampled memory
        states, actions, rewards, next_state, dones = self.sample_memory()
        states = self._state_processor(states)
        actions = self._state_processor(actions, dtype='int')
        next_state = self._state_processor(next_state)
        rewards = self._state_processor(rewards)
        dones = self._state_processor(dones)
        
        # Calculate expected Q-values using the policy net
        rewards = rewards.view(rewards.size(0), 1)
        q_pred = self.policy_net.forward(states).gather(1, actions.view(actions.size(0), 1))
        q_next = self.tgt_net.forward(next_state)
        
        # Compute the Q-target
        max_q_next = T.max(q_next, 1)[0]
        max_q_next = max_q_next.view(max_q_next.size(0), 1)
        q_target = rewards + self.gamma * max_q_next
        
        # Compute the loss using a specified loss function (MSE loss in this case)
        loss = self.loss(q_pred, q_target)
        self.log_loss.append(loss.item())
        
        return loss

    def learn(self):
        # The method responsible for the learning process
        if self.memory.memory_counter < self.batch_size:
            return  # If there's not enough data in the memory, skip learning
        
        self.learn_step_counter += 1
        
        if self.learn_step_counter < self.n_steps_warm_up_memory:
            return  # Skip learning until the warm-up period is over
        
        loss = self.calc_loss()
        
        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        self.optimizer.step()
        
        self.write_history(loss, grad_norm)
        self.replace_target_network(kind=self.replacementkind)  # Hard or soft network reset

    def normalize_gradient(self):
        # Normalize the gradient of the policy network
        grad_norm = nn.utils.clip_grad_norm_(self.policy_net.parameters(), self.max_grad_norm)
        return grad_norm

    def write_history(self, loss, grad_norm):
        # Write the loss and gradient information to the TensorBoard
        self.writer.add_scalars('step_train/agent_history', {'loss': loss.item(), 
                                                             'grad_norm': grad_norm}, 
                               self.learn_step_counter)

    def replace_target_network(self, kind='hard'):
        if kind == 'hard':
            if self.replace_target_count is not None and self.learn_step_counter % self.replace_target_count == 0:
                # Replace the target network with the policy network (hard replacement)
                self.tgt_net.load_state_dict(self.policy_net.state_dict())
        elif kind == 'soft':
            target_net_state_dict = self.tgt_net.state_dict()
            policy_net_state_dict = self.policy_net.state_dict()
            for key in policy_net_state_dict:
                # Update the target network using soft replacement
                target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (1 - self.tau)
            self.tgt_net.load_state_dict(target_net_state_dict)

    def get_number_of_actions(self):
        return self.policy_net.n_actions

    def get_q_values(self, states):
        """
        Calculates Q-values given a list of observations (states).
        """
        states = self._state_processor(states)
        q_values = self.policy_net.forward(states)
        return q_values.detach().cpu().numpy()

    def _state_processor(self, states, dtype='float'):
        """
        Converts a list of states into a torch tensor and copies it to the device.
        """
        if dtype == 'float':
            return T.tensor(states, dtype=T.float32).to(self.device)
        elif dtype == 'int':
            return T.tensor(states, dtype=T.int64).to(self.device)
        else:
            raise Exception('Error in state_processor in Agent!')

    def choose_action(self, observation, greedy=True):
        # Choose an action based on the Q-values from the observation

        q_values = self.get_q_values(observation)

        if not greedy:
            # Return the action with the highest Q-value
            return q_values.argmax()

        if np.random.random() > self.epsilon:
            # Choose the action with the highest Q-value
            action = q_values.argmax()
        else:
            # Choose a random action with exploration
            action = np.random.choice(self.action_space)

        self.epsilon = self.linear_decay(self.learn_step_counter)
        return action

    def linear_decay(self, cur_step):
        # Linearly decay the exploration epsilon from start to end
        if cur_step >= self.eps_decay:
            return self.eps_min
        return (self.epsilon_start * (self.eps_decay - cur_step) + self.eps_min * cur_step) / self.eps_decay

    def save(self, save_dir):
        # Save the policy network's parameters to the specified directory
        T.save(self.policy_net.state_dict(), save_dir)

    def save_to_disk(self, path):
        if not path.exists():
            os.makedirs(path)
        # Save hyperparameters in a JSON file
        with open(path / 'hparams.json', 'w') as f:
            json.dump(self.hyperparameters, f)
        # Save the target network to the specified path
        T.save(self.tgt_net, path / 'model')
