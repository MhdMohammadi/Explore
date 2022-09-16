import habitat
import argparse
from models import QNet, ReplayMemory
from Environment import is_done, get_environment, get_action_by_id
from Agent import RandomAgent
import torch.optim as optim
import torch.nn as nn
import torch
import numpy as np
from tqdm import tqdm
from visual import get_unseen_map, put_mark_on_map
import matplotlib.pyplot as plt


net: QNet = None
rb: ReplayMemory = None
optimizer: optim.Adam = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def optimize_model(config):
    global net, rb, optimizer

    if len(rb) < config.batch_size:
        return
    
    optimizer.zero_grad()

    data = torch.tensor(rb.sample(config.batch_size, 1 - config.gamma)).to(device) # (states, actions, positive_states, negative_states)

    positive_results = net.forward(data[0], data[1], data[2])
    negative_results = net.forward(data[0], data[1], data[3])

    loss = torch.sum(torch.log(torch.sigmoid(positive_results)) - torch.log(1 - torch.sigmoid(negative_results)))

    loss.backward()

    optimizer.step()

def train(config):
    global net, rb, optimizer
    print('--- start creating models and utilities ---')
    net = QNet(config, device).to(device)
    optimizer = optim.Adam(net.parameters(), lr=config.lr)
    rb  = ReplayMemory(config)

    print('--- create the environment ---')
    env: habitat.Env = get_environment(config.sim, config.config_path)
    agent = RandomAgent(env, config)
    
    print('--- start the learning section ---')
    
    for episode in range(config.episode):
        print(f'--- epislon {episode} has been started ---')

        env.reset()
 
        map = get_unseen_map(env, config.resolution)
        
        initial_loc = env.sim.sample_navigable_point()
        goal_loc = env.sim.sample_navigable_point()

        agent.set_position(goal_loc)
        put_mark_on_map(map, env)
        goal_state = agent.get_full_obs()

        agent.set_position(initial_loc)
        put_mark_on_map(map, env)
        current_state = agent.get_full_obs()


        for step in tqdm(range(config.episode_len)):
        # for step in range(config.episode_len):
            put_mark_on_map(map, env)
            best_action_id = net.get_best_action(torch.tensor(current_state).to(device).unsqueeze(0), 
                                                 torch.tensor(goal_state).to(device).unsqueeze(0), device)
            best_action = get_action_by_id(best_action_id)

            agent.take_action(best_action)
            next_state = agent.get_full_obs()
            
            rb.push((current_state, best_action, next_state, episode, step))

            optimize_model(config)

            if is_done(env.sim.get_agent(0).get_state().position, goal_loc):
                break
        
            
            current_state = next_state
        
        plt.imsave(f'map{episode}.jpg', map)



# TODO: Visualization for how a model is navigating
# TODO: I believe the image encoder is not good enough
# TODO: Data should be on CUDA, but it's not yet.
# TODO: sampling from the reply buffer is too manual. I'm afraid that maybe this section is too slow.
# One way is to use np functions to speed up. However, I'm not sure if this section is the bottle neck. 
# I can evaluate the required time for sampling and training, and by comparing them I can answer this question.
# TODO: Another thing about the reply buffer is this: I need to sample indices from a bounded geometric dist. 
# However, I'm giving more probability to the last element, and it makes the sampling inaccurate. But, I guess it's gonna be ok. 
#  TODO: some of the variables are spcified in the code such as: get_full_obs -> steps / Environment ->  
# TODO: I'm not sure if I'm adding correct transitions in reply buffer, because of the last transition.

if __name__ == '__main__':
    # config = {
    #     'obs_dim' : (256, 256),
    #     'obs_channel' : 3,
    #     'action_dim' : 4,
    #     'fc_dim' : 32,
    #     'reply_buffer_len' : 1000 * 1000,
    #     'episode': 100,
    #     'episode_len': 100
    # }

    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    
    # MDP hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99)

    # Q-value learning network structure
    parser.add_argument('--lr', type=float, default=0.1) # TODO : needs to be set appropriately
    parser.add_argument('--obs_dim', type=tuple, default=(640, 480)) # TODO: is it a good choice?
    parser.add_argument('--obs_channel', type=int, default=12) # Four images, each has three channels
    parser.add_argument('--latent_dim', type=int, default=32) # TODO: is it a good choice?
    parser.add_argument('--action_dim', type=int, default=4) # {Forward, Backward, Left, Right}
    parser.add_argument('--fc_dim', type=int, default=128) # TODO: is it a good choice?
    parser.add_argument('--batch_size', type=int, default=512) 

    # Reply buffer
    parser.add_argument('--reply_buffer_len', type=int, default=1000 * 1000)

    # Running configurations
    parser.add_argument('--episode', type=int, default=1)
    parser.add_argument('--episode_len', type=int, default=1000)
    
    # Enivornment and Agent configurations
    parser.add_argument('--sim', type=str, default='habitat', help='habitat')
    parser.add_argument('--config_path', type=str, default='configs/datasets/pointnav/gibson.yaml')
    parser.add_argument('--agent_save', type=bool, default=False)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--topdown_map_path', type=str, default='images/topdown_map')
    parser.add_argument('--seen_map_path', type=str, default='images/seen_map')
    parser.add_argument('--agent_mode', type=str, default='4_direction', help='random, repeated_random, 4_direction')
    parser.add_argument('--agent_repeat', type=int, default=1)
    parser.add_argument('--save_root', type=str, default='/scratch/mohammad/explore')

    # Parse hyperpatemeres
    args = parser.parse_args()

    train(args)