from pathlib import Path

import torch as T
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from Environment import ControlModelEnv

import os

from .args import get_args 
args = get_args()

global projectpath
global measurementpath
global metadata
projectpath = ''
measurementpath = ''
metadata = {}



class DataScaler:
    def __init__(self, data, new_min, new_max):
        # Find the current minimum and maximum values in the data
        self.old_min = min(data)
        self.old_max = max(data)
        self.new_min = new_min
        self.new_max = new_max

    def scale_number(self, x):
        scaled_number = (x - self.old_min) / (self.old_max - self.old_min) * (self.new_max - self.new_min) + self.new_min
        return scaled_number
    
    def normalize(self, data, new_min, new_max):
        # Normalize each value to the new range
        normalized_data = [(x - self.old_min) / (self.old_max - self.old_min) * (self.new_max - self.new_min) + self.new_min for x in data]
        return normalized_data

def get_agent_id(path= None) -> str:
    """
    Returns the highest id of the saved agents
    """
    if path == None:
        from utils.config import SAVED_AGENTS_DIR
        dir = Path(SAVED_AGENTS_DIR) 
        if not dir.exists():
            os.makedirs(dir)
    else:
        dir = Path
        if not dir.exists():
            os.makedirs(dir)
    # try:
    #     agent_id = max([int(id) for id in os.listdir(dir)]) + 1
    # except ValueError:
    #     agent_id = 0

    ids = []
    for id in os.listdir(dir):
        try:
            ids.append(int(id))
        except:
            pass
    if len(ids) > 0:
        agent_id = max(ids) + 1
    else:
        agent_id = 0
    # stop()
    return str(agent_id)

def save_dict_to_file(dic, name, path=None):
    if name[-4:]!='.txt':
        name = name + '.txt'
    if path!=None:
        if type(path) == str:
            name = path + name
        else:
            name=path / name
    f = open(name,'w')
    f.write(str(dic))
    f.close()

def load_dict_from_file(name, path=None):
    if name[-4:]!='.txt':
        name = name + '.txt'
    f = open(name,'r')
    data=f.read()
    f.close()
    return eval(data)

def save_data(name, 
              path,
              scores, 
              avg_scores, 
              save_interval, 
              info, 
              time, 
              eps_history=None):
    x = [i + 1 for i in range(len(scores))]
    if eps_history == None:
        dict_scores = {'Episode': x, 'Score': scores,
                       'Avg Score': avg_scores, 
                        'Zeit': time, 
                        'Speicherzyklus': save_interval, 
                        'Informationen': info}
    else:
        dict_scores = {'Episode': x, 'Score': scores, 
                       'Avg Score': avg_scores, 
                        'Epsilon': eps_history, 
                        'Zeit': time, 
                        'Speicherzyklus': save_interval, 
                        'Informationen': info}
    name = name + '.txt'
    save_dict_to_file(dict_scores, name, path)


def calc_avg(history):
    plt_data_mean, plt_data_dev = [], []
    for episode in history:
        episode = np.array(episode)
        mean = np.mean(episode)
        sigma = np.std(episode)
        plt_data_mean.append(mean)
        plt_data_dev.append(sigma)
    return plt_data_mean,plt_data_dev
    

    
def plot_training(path, ac_hstry, rw_hstry, loss_hrsty, rw_avg, eps):
# n_actions=None):
    '''
    Generation of Image -> Training
    Version 1: Action - Reward - Loss
    Version 2: Reward - Loss

    Parameters
    ----------
    path : Path for saving
    ac_hstry : history of actions
    rw_hstry : history of rewards
    rw_avg : average reward
    eps : epsilon
    '''
    
    try:
        plt.close('all')
        os.makedirs(path)
    except:
        pass
    fig_v1, (axx1,axx2,axx3) = plt.subplots(3,1, sharex='row')
    fig_v2, (axxx1, axxx2) = plt.subplots(2,1, sharex='row')
    
    fig1, ax1 = plt.subplots(1,1)
    fig2, ax2 = plt.subplots(1,1)
    fig3, ax3 = plt.subplots(1,1)
    
    
    # Generating Plot for Action Mapping
    # if n_actions != None:
    #     ratio=0.1/n_actions #0.1 m durch anzahl aktionen
    #     delta_length = ratio*np.array(ac_hstry)
    mean_data, sigma_data = calc_avg(ac_hstry)
    mean_data=np.array(mean_data)
    sigma_data=np.array(sigma_data)
    y1 = mean_data-sigma_data
    y2 = mean_data+sigma_data
    x = np.linspace(0, len(ac_hstry), len(ac_hstry))
    
    # Plotting Action-Mapping
    ax1.fill_between(x,y1,y2,interpolate=True,label='Standard Deviation Action',color='blue',alpha=0.10)
    ax1.plot(x,mean_data,color='blue',label='Mean Action',alpha=1.00)
    ax1.set_xlim([0,len(ac_hstry)])
    ax1.set_ylim([0,args.n_actions])
    ax1.set_xlabel('Episodes')
    ax1.set_ylabel('Action')
    ax1.legend()
    
    axx1.plot(x,mean_data,color='blue',label='Mean Action',alpha=1.00)
    axx1.set_xlim([0,len(ac_hstry)])
    axx1.set_ylim([0,args.n_actions])
    axx1.set_xlabel('Episodes')
    axx1.set_ylabel('Action')
    axx1.legend()
    
    ax1.set_title('Average-chosen Action over Episodes')
    if type(path) == str:
        fname =path +'ActionMapping.pdf'
    else:
        fname =path/'ActionMapping.pdf'
    fig1.savefig(fname, dpi=1200)
    if type(path) == str:
        fname =path +'ActionMapping.png'
    else:
        fname =path/'ActionMapping.png'
    fig1.savefig(fname, dpi=1200)
    
    # Generating Plot for Reward Mapping
    mean_data, sigma_data = calc_avg(rw_hstry)
    mean_data=np.array(mean_data)
    sigma_data=np.array(sigma_data)
    y1 = mean_data-sigma_data
    y2 = mean_data+sigma_data
    x = np.linspace(0, len(rw_hstry), len(rw_hstry))
    average=np.array(rw_avg)
    average=average/1000
    
    
    # Plotting Reward Mapping
    ax2.fill_between(x,y1,y2,interpolate=True,label='Standard Deviation Reward',color='blue',alpha=0.10)
    ax2.plot(x,mean_data,color='blue',label='Mean Reward',alpha=1.00)
    ax2.plot(x,average,color='red',label='Average Reward over 100 Episodes',alpha=1.00)
    ax2.plot(x,eps,color='green',label='Epsilon of the Agent',alpha=1.00)
    ax2.set_xlim([0,len(ac_hstry)])
    ax2.set_ylim([0,1])
    ax2.set_xlabel('Episodes')
    ax2.set_ylabel(' $\dfrac{Reward}{Maximum  Reward}$')
    ax2.legend()
    
    axx2.plot(x,mean_data,color='blue',label='Mean Reward',alpha=1.00)
    axx2.plot(x,average,color='red',label='Average Reward over 100 Episodes',alpha=1.00)
    # axx2.plot(x,eps,color='green',label='Epsilon of the Agent',alpha=1.00)
    axx2.set_xlim([0,len(ac_hstry)])
    axx2.set_ylim([0,1])
    axx2.set_xlabel('Episodes')
    axx2.set_ylabel(' $\dfrac{Reward}{Maximum  Reward}$')
    axx2.legend()
    
    axxx1.plot(x,mean_data,color='blue',label='Mean Reward',alpha=1.00)
    axxx1.plot(x,average,color='red',label='Average Reward over 100 Episodes',alpha=1.00)
    # axxx1.plot(x,eps,color='green',label='Epsilon of the Agent',alpha=1.00)
    axxx1.set_xlim([0,len(ac_hstry)])
    axxx1.set_ylim([0,1])
    axxx1.set_xlabel('Episodes')
    axxx1.set_ylabel(' $\dfrac{Reward}{Maximum  Reward}$')
    axxx1.legend()
    
    ax2.set_title('Reward over Episodes')
    fname = str(path)+'/RewardMapping.pdf'
    fig2.savefig(fname, dpi=1200)
    fname = str(path)+'/RewardMapping.png'
    fig2.savefig(fname, dpi=1200)
    
    
    # Generating Plot for Loss Mapping
    mean_data, sigma_data = calc_avg(loss_hrsty)
    mean_data=np.array(mean_data)
    sigma_data=np.array(sigma_data)
    y1 = mean_data-sigma_data
    y2 = mean_data+sigma_data
    x = np.linspace(0, len(rw_hstry), len(rw_hstry))
    average=np.array(rw_avg)
    average=average/1000
    
    # Plotting Reward Mapping
    ax3.fill_between(x,y1,y2,interpolate=True,label='Standard Deviation Loss',color='blue',alpha=0.10)
    ax3.plot(x,mean_data,color='blue',label='Mean Loss',alpha=1.00)
    ax3.set_xlim([0,len(ac_hstry)])
    # ax3.set_ylim([0,1])
    ax3.set_xlabel('Episodes')
    ax3.set_ylabel('Loss')
    ax3.legend()
    
    axx3.plot(x,mean_data,color='blue',label='Mean Loss',alpha=1.00)
    axx3.set_xlim([0,len(ac_hstry)])
    # axx3.set_ylim([0,1])
    axx3.set_xlabel('Episodes')
    axx3.set_ylabel('Loss')
    axx3.legend()
    
    axxx2.plot(x,mean_data,color='blue',label='Mean Loss',alpha=1.00)
    axxx2.set_xlim([0,len(ac_hstry)])
    # axxx2.set_ylim([0,1])
    axxx2.set_xlabel('Episodes')
    axxx2.set_ylabel('Loss')
    axxx2.legend()
    
    ax3.set_title('Loss over Episodes')
    fname = str(path)+'/LossMapping.pdf'
    fig3.savefig(fname, dpi=1200)
    fname = str(path)+'/LossMapping.png'
    fig3.savefig(fname, dpi=1200)
    
    axx1.set_title('Actions performed', loc='left', fontstyle='oblique', fontsize='medium')
    axx2.set_title('Rewards', loc='left', fontstyle='oblique', fontsize='medium')
    axx3.set_title('Loss of the Agent', loc='left', fontstyle='oblique', fontsize='medium')
    axxx1.set_title('Rewards', loc='left', fontstyle='oblique', fontsize='medium')
    axxx2.set_title('Loss of the Agent', loc='left', fontstyle='oblique', fontsize='medium')
    fig_v1.suptitle('Training over {} Episodes'.format(len(x)))
    fig_v2.suptitle('Training over {} Episodes'.format(len(x)))
    
    fname = str(path)+'/1_Version1.pdf'
    fig_v1.savefig(fname, dpi=1200)
    fname = str(path)+'/1_Version1.png'
    fig_v1.savefig(fname, dpi=1200)
    fname = str(path)+'/1_Version2.pdf'
    fig_v2.savefig(fname, dpi=1200)
    fname = str(path)+'/1_Version2.png'
    fig_v2.savefig(fname, dpi=1200)
    
    plt.close()
    
      

class DQN(nn.Module):
    def __init__(self, input_dims, n_actions, fc1_dims, fc2_dims):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(input_dims, fc1_dims)
        self.layer2 = nn.Linear(fc1_dims, fc2_dims)
        self.layer3 = nn.Linear(fc2_dims, n_actions)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        x = self.layer1(x)
        x = F.relu(x)
        x = self.layer2(x)
        x = F.relu(x)
        return self.layer3(x)


def eval_model_load(directory,
               input_dims = 5, 
               fc1_dims = 256, 
               fc2_dims = 128, 
               n_actions = 1000, 
               save_interval=100):
    '''
    Parameters
    ----------
    directory : TYPE
        DESCRIPTION.
    input_dims : TYPE, optional
        DESCRIPTION. The default is 5.
    fc1_dims : TYPE, optional
        DESCRIPTION. The default is 128.
    fc2_dims : TYPE, optional
        DESCRIPTION. The default is 256.
    n_actions : TYPE, optional
        DESCRIPTION. The default is 1000.
    save_interval : TYPE, optional
        DESCRIPTION. The default is 100.

    Returns
    -------
    None.

    '''
    list_dir = os.listdir(directory)
    n_net = len(list_dir)
    
    for i in range(n_net):
        done_cnt, step = 0, 0
        current_agent=(i+1)*save_interval
        file_dir = directory + '/agent_{}.pkl'.format(current_agent)
        
        action_hstry, RoC_hstry, A_hstry, I_hstry = [], [], [], []
    
        agent = DQN(input_dims, fc1_dims, fc2_dims, n_actions)
        agent.load_state_dict(T.load(file_dir))
        
        env = ControlModelEnv(n_actions)
        
        state = env.reset()
        # env.change()
        
        while True:
            # list of env 
            state = agent._state_processor(state)    
            actions = agent.forward(state.unsqueeze(dim=0))
            action = T.argmax(actions).item()
            
            state, _,done, _ = env.step(action)
            RoC_Err, A_Err, I_Err = env.get_eval()
            
            step+=1
            if step==2000:
                print('\nAgent with {} Episode training is not good!'.format(current_agent)) 
                break
            if done==True:
                done_cnt+=1
                if done_cnt>=10:
                    break
            else:
                done_cnt=0
            
            action_hstry.append(action)
            RoC_hstry.append(RoC_Err)
            A_hstry.append(A_Err)
            I_hstry.append(I_Err)
                
        plot_evaluation(action_hstry, RoC_hstry, A_hstry, I_hstry, 
                        path = directory, current= current_agent)
    
def plot_evaluation(actions, RoC_Err, A_err, I_err, 
                    path = None, 
                    enumerator = None,
                    current=None, ylimits=True):
    x = np.arange(0, len(actions))
    x1=np.arange(0, len(A_err))
    fig, (ax1,ax2, ax3) = plt.subplots(3, 1, sharex=True) #figsize=[12,10],
                                      # gridspec_kw={'height_ratios': [2,3,3]})
    # plt.subplots_adjust(right= 0.95, 
    #                     left= 0.05, 
    #                     bottom = 0.1,
    #                     hspace=0.8)
    ratio=100/1000 # to show it in mm
    action = ratio*np.array(actions)
    
    ax1.plot(x, action , 'k-' , label='Postion in mm')
    fig.tight_layout()
    ax2.plot(x, RoC_Err , 'b-')# , label='Divergence Error')
    ax02 = ax2.twinx()
    ax02.plot(x1, np.array(A_err),'r-')#, label='Area Error')# marker= '.',color=
    fig.tight_layout()
    ax3.plot(x, I_err , 'k-' , label='Normalized Intensity')
    fig.tight_layout()
    
    ax1.set_title('Position of the Lens on the Motorstage')
    # ax1.set_xlim([0,len(actions)+1])
    ax1.set_ylim([0,105])
    ax1.set_xlabel('Steps of AI')
    ax1.set_ylabel('Lensposition/mm')
    ax1.legend(loc='upper right')
    fig.tight_layout()
    
    ax2.set_title('Divergence and Area Error of the Modematching')
    fig.tight_layout()
    if ylimits==True:
        ax2.set_ylim([0,1.55])
    ax2.set_xlabel('Steps of AI')
    ax2.set_ylabel('Error')
    ax2.legend(['Divergence Error'], loc='upper left')
    ax02.legend(['Area Error'], loc='upper right')
    # ax2.legend(['Divergence Error', 'Area Error'], loc=0)
    fig.tight_layout()
    
    ax3.set_title('Normalized state of the Environment given to the Agent')
    # ax3.set_xlim([0,len(actions)+1])
    ax3.set_ylim([0,1.05])
    ax3.set_xlabel('Steps of AI')
    ax3.set_ylabel('State')
    ax3.legend(loc='upper right')

    # fig.tight_layout()
    
    if path != None:
        path = os.path.join(path, 'eval_img')
        try:
            os.mkdir(path)
        except:
            pass
        if enumerator != None:
            fname1 = path+'/Agent_{}_eval.pdf'.format(enumerator)
            fname2 = path+'/Agent_{}_eval.png'.format(enumerator)
        else:
            fname1 = path+'/Agent_{}_eval.pdf'.format(0)
            fname2 = path+'/Agent_{}_eval.png'.format(0)
        
        fig.savefig(fname1, dpi=1200)
        fig.savefig(fname2, dpi=1200)
        plt.close(fig)
        
def save_model(agent, directory):
    agent.save(directory)       
        

def eval_model(directory, 
               agent,
               env,
               eval_steps = 1,
               max_steps_per_episode = 25,
               greedy = False,
               seed = 0,
               info = "Keine Info",
               hparams = {},
               number_to_seperate = None,
               max_eval_env = 6 ):
    #set_seed(env, seed)
    
    reward_mean_per_episode =[]
    reward_per_episode, steps_per_episode, action_per_episode = [] ,[] ,[]
    RoC_per_episode, Area_per_episode, I_Err_per_episode = [], [], []
    if env.env_version == "Evaluation":
        for i in range(env.max_eval_env):
            # output metrics
            RoC_per_step, Area_per_step, I_State_per_step, action_per_step = [], [], [], []
        
            state = env.reset()
            env.reset_eval_env(i)
            rewards = 0
            steps = 0
            done = False
            while not done:
                action = agent.choose_action(state, greedy=False)
                next_state, reward, done, info = env.step(action)
                
                # collecting data
                RoC_Err, A_Err, I_State = env.get_eval()
                # print(RoC_Err,A_Err,I_State)
                RoC_per_step.append(RoC_Err)
                Area_per_step.append(A_Err)
                I_State_per_step.append(I_State)
                action_per_step.append(action)
                
                rewards += reward
                steps += 1
                if steps == max_steps_per_episode:
                    done = True
                state = next_state
            
            if number_to_seperate != None:
                enum = str(number_to_seperate) + '_' + str(i+1)
            else:
                enum = i+1
            plot_evaluation(action_per_step, RoC_per_step, Area_per_step, I_State_per_step, 
                            path = directory, enumerator = enum, ylimits=False)
            reward_mean_per_episode.append(rewards/steps)
            reward_per_episode.append(rewards)
            steps_per_episode.append(steps)
            RoC_per_episode.append(RoC_per_step)
            Area_per_episode.append(Area_per_step)
            I_Err_per_episode.append(I_State_per_step)
            action_per_episode.append(action_per_step)
        
    else:
        for i in range(0, eval_steps):
            # output metrics
            RoC_per_step, Area_per_step, I_Err_per_step, action_per_step = [], [], [], []
        
            state = env.reset()
            rewards = 0
            steps = 0
            done = False
            while not done:
                action = agent.choose_action(state, greedy=False)
                next_state, reward, done, info = env.step(action)
                
                # collecting data
                RoC_Err, A_Err, I_err = env.get_eval()
                
                RoC_per_step.append(RoC_Err)
                Area_per_step.append(A_Err)
                I_Err_per_step.append(I_err)
                action_per_step.append(action)
                
                rewards += reward
                steps += 1
                if steps == max_steps_per_episode:
                    done = True
                state = next_state
            
            if number_to_seperate != None:
                enum = str(number_to_seperate) + '_' + str(i+1)
            else:
                enum = i+1
            plot_evaluation(action_per_step, RoC_per_step, Area_per_step, I_Err_per_step, 
                            path = directory, enumerator = enum, ylimits=False)
            reward_mean_per_episode.append(rewards/steps)
            reward_per_episode.append(rewards)
            steps_per_episode.append(steps)
            RoC_per_episode.append(RoC_per_step)
            Area_per_episode.append(Area_per_step)
            I_Err_per_episode.append(I_Err_per_step)
            action_per_episode.append(action_per_step)
                
    dict_evaluation = {
        'Info' : info,
        'Mean Reward' : reward_mean_per_episode,
        'Reward' : reward_per_episode,
        'Steps' : steps_per_episode,
        'Error Radius' : RoC_per_episode,
        'Error Area' : Area_per_episode,
        'Error Intensity' : I_Err_per_episode,
        'Action' : action_per_episode,
        'Hyperparameter': hparams
        }
    save_dict_to_file(dict_evaluation, 'Evaluation_Dict', path = directory)

    agent.writer.add_scalar('eval/avg_reward', np.mean(reward_mean_per_episode), i)
    agent.writer.add_scalar('eval/avg_steps', np.mean(steps_per_episode), i)
    
    return reward_per_episode, steps_per_episode


