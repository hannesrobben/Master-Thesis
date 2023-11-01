from tqdm import tqdm
import numpy as np
from datetime import datetime
import os
cur_dir = os.getcwd()

from Environment import ControlModelEnv
from DQN_Agent import DQNAgent

from utils.utils import plot_training, save_model, eval_model, save_data
from utils.args import get_args as get_args_system

os.chdir(cur_dir)

args = get_args_system()

global I_logging 
I_logging = [] 


def training_DQN(filename, 
                 parent_dir,
                 max_episode = None,
                 env = None,
                 agent = None, 
                 n_actions = 1000,
                 save_interval = 100, #args.save_interval
                 eval_interval = 100, #args.eval_interval
                 rewardtype = 0,
                 env_version = "Training",
                 maxstep = 1000, 
                 Adding_Noise=True,
                 act_func="relu"
                ):
    now=datetime.now()
    date=now.strftime('%Y_%m_%d')
    test_name = filename +'_'+ date #test specification + training date
    
    BASE_DIR = parent_dir+'/TRAINING_DATA/'+test_name # base directory for test
    
    LOG_DIR = BASE_DIR+'/logs' # logging directory
    MODEL_DIR = BASE_DIR+'/model' # model saving directory
    IMG_DIR = BASE_DIR+'/img' # directory for standard generated images
    EVAL_DIR = BASE_DIR+'/eval' # directory for evaluating the agent
    try:
        os.makedirs(LOG_DIR)
    except:
        pass
    try:
        os.makedirs(MODEL_DIR)
    except:
        pass
    try:
        os.makedirs(IMG_DIR)
    except:
        pass
    try:
        os.makedirs(EVAL_DIR)
    except:
        pass
    
    current_episode = 0

    logging = False
    
    if max_episode == None: max_episode = args.total_steps
    if env == None: env = ControlModelEnv(n_actions, 
                                          rewardtype= rewardtype, 
                                          version = env_version,
                                          Adding_Noise=Adding_Noise)  
    if env_version == "Training":
        env_eval = ControlModelEnv(n_actions, 
                                rewardtype= rewardtype, 
                                version = "Evaluation", Adding_Noise=True)
    else:
        env_eval = ControlModelEnv(n_actions, 
                                rewardtype= rewardtype, 
                                version = env_version, Adding_Noise=True)
    if agent == None: agent = DQNAgent(input_dims=env.observation_space.shape, 
                                       log_dir= LOG_DIR, replace = 500,
                                       activation_function=act_func )     
    if LOG_DIR != None: logging = True
    
    
    
    # output metrics
    scores , avg_scores, eps_history = [], [], []
    reward_per_episode, steps_per_episode, action_per_episode = [] ,[] ,[]
    loss_per_episode = []
   
    for current_episode in tqdm(range(0, max_episode)):
    # while current_episode < max_episode:
        tic = datetime.now() 
        done = False
        observation = env.reset()
        
        step, score = 0, 0 
        reward_per_step, action_per_step, loss_per_step = [], [], []

        while True:
            # print(observation)
            action = agent.choose_action(observation)
            
            next_state, reward, done, info = env.step(action)
            
            # reward ist max. 1, also score max. 1*1000=1000
            agent.store_transition(observation, action, reward, next_state, int(done))
            agent.learn()
            if agent.log_loss != []:
                loss = agent.log_loss[-1]
                loss_per_step.append(loss)
            observation = next_state.astype(np.float32)
            score += reward
            step += 1
            reward_per_step.append(reward)
            action_per_step.append(action)
            I_logging.append(info)
            
            # done_info = info
            if step==maxstep: break
            if done:
                break
            
        score_mean = score / step
        # print(score)
        current_episode+=1
        toc = datetime.now()
        time = toc-tic
        
        if logging:
            # agent.writer.add_scalar('train/loss', agent.loss, current_episode)
            if agent.log_loss != []: agent.writer.add_scalar('train/log_loss', agent.log_loss[-1], current_episode)
            agent.writer.add_scalar('train/rewards_total_per_episode', score, current_episode)
            agent.writer.add_scalar('train/rewards_mean_durch_step', score_mean, current_episode)
            agent.writer.add_scalar('train/steps', step, current_episode)
            agent.writer.add_scalar('train/epsilon', agent.epsilon, current_episode)
            agent.writer.add_scalar('train/replay_memory_size', agent.mem_size, current_episode)
            agent.writer.add_scalar('train/trainingtime', time.seconds, current_episode)
    
        steps_per_episode.append(step)
        reward_per_episode.append(reward_per_step)
        action_per_episode.append(action_per_step) # action history
        loss_per_episode.append(loss_per_step)
        scores.append(score)
        avgscore = np.mean(scores[-100:])
        avg_scores.append(avgscore)
        eps_history.append(agent.epsilon)
        
        if current_episode%save_interval == 0:
            n=current_episode/save_interval
            model_save_dir = MODEL_DIR+'/agent_{}.pkl'.format(current_episode)
            save_model(agent, model_save_dir)
        if current_episode%eval_interval == 0:
            n=current_episode/eval_interval
            env_eval.reset()
            eval_model(EVAL_DIR, agent, env_eval, eval_steps = 1, number_to_seperate=n)
                
    toc = datetime.now()
    time = toc-tic
    save_data(filename, BASE_DIR, scores, avg_scores, save_interval, info, time)
    plot_training(IMG_DIR, action_per_episode, reward_per_episode, loss_per_episode, avg_scores, eps_history)
    # plot_loss_steps(IMG_DIR, loss_per_episode, steps_per_episode)
    
    
def plot_loss_steps(IMG_DIR,
              loss_history,
              step_per_episode_counter):
    try:
        import matplotlib.pyplot as plt
    except:
        pass
    fig1, ax1 = plt.subplots(1,1)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')

    loss_step = [item for sublist in loss_history for item in sublist]
    #Loss-Mapping
   
    ax1.plot(loss_step)
    
    ax1.set_title('Loss per Step')
    fname = IMG_DIR+'/Loss_Mapping_1'+'.pdf'
    fig1.savefig(fname, dpi=1200)
    fname = IMG_DIR+'/Loss_Mapping_1'+'.png'
    fig1.savefig(fname, dpi=1200)
    
def plot_loss(IMG_DIR,
              loss_history,
              step_per_episode_counter):
    try:
        import matplotlib.pyplot as plt
    except:
        pass
    
    # Plotting A
    fig1, ax1 = plt.subplots(1,1)
    ax1.set_xlabel('Step')
    ax1.set_ylabel('Loss')
    step=0
    counter=0
    
    #Loss-Mapping
    for episode in loss_history:
        step+=1
        if step%50==0:
            counter+=1
            ax1.set_title('50 Episodes of Loss. Nummer:{}'.format(counter))
            fname = IMG_DIR+'/Loss_Mapping_'+str(counter)+'.pdf'
            fig1.savefig(fname, dpi=1200)
            fname = IMG_DIR+'/Loss_Mapping_'+str(counter)+'.png'
            fig1.savefig(fname, dpi=1200)
            
            fig1, ax1 = plt.subplots(1,1)
            ax1.set_xlabel('Step')
            ax1.set_ylabel('Loss')
        ax1.plot(episode)
        step+=1
        
    
    counter+=1
    ax1.set_title('50 Episodes of Loss. Nummer:{}'.format(counter))
    fname = IMG_DIR+'/Loss_Mapping_'+str(counter)+'.pdf'
    fig1.savefig(fname, dpi=1200)
    fname = IMG_DIR+'/Loss_Mapping_'+str(counter)+'.png'
    fig1.savefig(fname, dpi=1200)


def training_hyperparams(max_episode = None,
                         env = None,
                         agent = None, 
                         LOG_DIR = None,
                         img_dir=None):
    current_episode = 0

    logging = False
    
    if max_episode == None: max_episode = args.total_steps
    if env == None: 
        n_actions = 1000
        env = ControlModelEnv(n_actions)       
    if agent == None: agent = DQNAgent(input_dims=env.observation_space.shape)
    if LOG_DIR != None: logging = True
    
    tic = datetime.now() 
    # output metrics
    scores , avg_scores, eps_history = [], [], []
    reward_per_episode, reward_per_step_episode = [], []
    steps_per_episode, action_per_episode, loss_per_episode = [] ,[] ,[]
   
    # print(max_episode)
    for current_episode in tqdm(range(0, max_episode)):
        done = False
        observation = env.reset()
        
        step, score = 0, 0 
        reward_per_step, action_per_step, loss_per_step = [], [], []
        
        while True:
            # print(observation)
            action = agent.choose_action(observation)
            
            next_state, reward, done, info = env.step(action)
            
            # reward ist max. 1, also score max. 1*1000=1000
            agent.store_transition(observation, action, reward, next_state, int(done))
            agent.learn()
            if agent.log_loss != []:
                loss = agent.log_loss[-1]
                loss_per_step.append(loss)
            
            observation = next_state.astype(np.float32)
            score += reward
            step += 1
            reward_per_step.append(reward)
            action_per_step.append(action)
            if step==1000: break
            # done_info = info
            if done:
                break
        
        current_episode+=1
        toc = datetime.now()
        time = toc-tic
        
        if logging:
            if agent.log_loss != []: agent.writer.add_scalar('train/loss', agent.log_loss[-1], current_episode)
            agent.writer.add_scalar('train/rewards_total_per_episode', score, current_episode)
            agent.writer.add_scalar('train/steps', step, current_episode)
            agent.writer.add_scalar('train/epsilon', agent.epsilon, current_episode)
            agent.writer.add_scalar('train/replay_memory_size', agent.mem_size, current_episode)
            agent.writer.add_scalar('train/trainingtime', time.seconds, current_episode)
    
        steps_per_episode.append(step)
        reward_per_step_episode.append(reward_per_step)
        reward_per_episode.append(score)
        action_per_episode.append(action_per_step) # action history
        loss_per_episode.append(loss_per_step)
        scores.append(score)
        avgscore = np.mean(scores[-100:])
        avg_scores.append(avgscore)
        eps_history.append(agent.epsilon)
                
    toc = datetime.now()
    time = toc-tic
    plot_training(img_dir, action_per_episode, reward_per_step_episode, loss_per_episode, avg_scores, eps_history)
    print('\n{} min needed for Training!'.format(time))
    
    