from datetime import datetime
import os
from routines.training_script import training_DQN

if __name__ == '__main__':   
    
    filename = 'Training'  # foldername
    parent_dir = os.getcwd()
    tic = datetime.now()

    training_DQN(filename, parent_dir, 
                  max_episode=1000, eval_interval= 100, rewardtype = 0, 
                  env_version = "Training",
                  maxstep = 1000, Adding_Noise=True,
                  act_func="relu")

    toc = datetime.now()
    time = toc-tic
    print('Total time used:', time)