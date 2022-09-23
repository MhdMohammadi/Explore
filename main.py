import argparse
from turtle import pos
from tqdm import tqdm
import matplotlib.pyplot as plt
import os

import habitat
import torch.optim as optim
import torch

from models import QNet, ReplayMemory
from Environment import is_done, get_environment, get_action_by_id
from Agent import RandomAgent
from visual import put_mark_on_map, get_topdown_map

net: QNet = None
rb: ReplayMemory = None
optimizer: optim.Adam = None
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

loss_value = []
positive_results_means = []
negative_results_means = []

def optimize_model(config):
    global net, rb, optimizer, X

    if len(rb) < config.batch_size:
        return
    
    optimizer.zero_grad()

    data = rb.sample(config.batch_size, 1 - config.gamma) # (states, actions, positive_states, negative_states)

    # TODO: read models.py because torch.inner is a wrong way to comute phi.T /times psi
    positive_results = net(data[0], data[1], data[2])
    negative_results = net(data[0], data[1], data[3])

    positive_results_means.append(positive_results.mean())
    negative_results_means.append(negative_results.mean())

    loss = -(torch.mean(torch.log(torch.sigmoid(positive_results)) + torch.log(1 - torch.sigmoid(negative_results))))

    loss_value.append(loss.item())

    loss.backward()

    optimizer.step()

def run(config):
    print('--- start creating corresponding directories ---')
    save_dir = f'{config.save_root}/episode_{config.episode}_episodelen_{config.episode_len}_lr_{config.lr}_gamma_{config.gamma}' 
    model_dir = f'{save_dir}/models'
    images_dir = f'{save_dir}/images'
    eval_dir = f'{save_dir}/evals'
        
    if config.mode == 'train':
        os.mkdir(save_dir)
        os.mkdir(model_dir)
        os.mkdir(images_dir)
    else:
        os.mkdir(eval_dir)

    global net, rb, optimizer
    print('--- start creating models and utilities ---')
    net = QNet(config, device).to(device)
    if config.mode == 'eval':
        net.load_state_dict(torch.load(f'{model_dir}/{config.model_address}'))
    # TODO: resume
    
    if config.mode == 'train':
        optimizer = optim.Adam(net.parameters(), lr=config.lr)
        rb = ReplayMemory(config, device)

    print('--- create the environment ---')
    env: habitat.Env = get_environment(config.sim, config.config_path)
    agent = RandomAgent(env, config, device)
    
    print('--- start the learning section ---')
    for episode in range(config.episode):
        print(f'--- episode {episode} has been started ---')
        env.reset()
 
        map = get_topdown_map(env, config.resolution) * 0.5
        
        initial_loc = env.sim.sample_navigable_point()
        goal_loc = env.sim.sample_navigable_point()

        agent.set_position(goal_loc)
        put_mark_on_map(map, env)
        goal_state = agent.get_full_obs()

        agent.set_position(initial_loc)
        put_mark_on_map(map, env)
        current_state = agent.get_full_obs()

        eps = max(config.min_eps, config.max_eps - episode * config.eps_decay) if config.mode == 'train' else 0
        print(f'epsilon in this episode is equal to {eps}')
        for step in tqdm(range(config.episode_len)):
            put_mark_on_map(map, env)
            best_action_id = net.get_best_action([current_state], [goal_state], greedy=True, p=eps)
            best_action = get_action_by_id(best_action_id)

            agent.take_action(best_action)
            next_state = agent.get_full_obs()

            if config.mode == 'train':          
                rb.push((current_state, best_action_id, next_state, episode, step))
            else:
                plt.imsave(f'images/eval_{episode}_step_{step}.jpg', map) # save in a local address
                plt.imsave(f'{eval_dir}/eval_{episode}_step_{step}.jpg', map) # save in a local address
            
            if is_done(env.sim.get_agent(0).get_state().position, goal_loc):
                break        
            current_state = next_state

        print('--- optimization has been started ---')
        for _ in tqdm(range(config.optimization_steps)):
            optimize_model(config)

        if config.mode == 'train':
            # Save the trajectory visualizaiton
            plt.imsave(f'images/map_episode_{episode}.jpg', map) # save in a local address
            plt.imsave(f'{images_dir}/map_episode_{episode}.jpg', map) # save in a global storage
            
            torch.save(net.state_dict(), f'{model_dir}/episode_{episode}.pth') # save in a global storage
            
            plt.plot(loss_value)
            plt.savefig(f'images/loss.jpg')

        else:
            plt.imsave(f'images/final_eval_{episode}.jpg', map) # save in a local address
            plt.imsave(f'{eval_dir}/final_eval_{episode}.jpg', map) # save in a local address
            
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
# TODO: Any preprocessing on images to make the learning quicker
# TODO: resume
# TODO: WANDB
# TODO: finishing constrain is not good enough
# TODO: add eval mode as well
# TODO: Clean code and documentation
# TODO: TSNE
# TODO: do something about the replay buffer
# TODO: change unsqueeze as the warning mentioned 
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    
    # Code mode
    parser.add_argument('--mode', type=str, default='train', help='train, eval')
    
    # MDP hyperparameters
    parser.add_argument('--gamma', type=float, default=0.99)

    # Q-value learning network structure
    parser.add_argument('--lr', type=float, default=1e-4) # TODO : needs to be set appropriately
    parser.add_argument('--obs_dim', type=tuple, default=(120, 160)) # TODO: is it a good choice?
    parser.add_argument('--obs_channel', type=int, default=12) # Four images, each has three channels
    parser.add_argument('--latent_dim', type=int, default=64) # TODO: is it a good choice?
    parser.add_argument('--action_dim', type=int, default=4) # {Forward, Backward, Left, Right}
    parser.add_argument('--fc_dim', type=int, default=256) # TODO: is it a good choice?
    parser.add_argument('--batch_size', type=int, default=128) 

    # Reply buffer
    parser.add_argument('--reply_buffer_len', type=int, default=1000 * 1000)

    # epsilon-greedy parameters
    parser.add_argument('--max_eps', type=float, default=1.0)
    parser.add_argument('--min_eps', type=float, default=0.05)
    parser.add_argument('--eps_decay', type=float, default=1/500)

    # Running configurations
    parser.add_argument('--episode', type=int, default=600)
    parser.add_argument('--episode_len', type=int, default=1000)
    parser.add_argument('--optimization_steps', type=int, default=64)
    
    # Enivornment and Agent configurations
    parser.add_argument('--sim', type=str, default='habitat', help='habitat')
    parser.add_argument('--config_path', type=str, default='configs/datasets/pointnav/gibson.yaml')
    parser.add_argument('--agent_save', type=bool, default=False)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--topdown_map_path', type=str, default='images/topdown_map')
    parser.add_argument('--seen_map_path', type=str, default='images/seen_map')
    parser.add_argument('--agent_mode', type=str, default='4_direction', help='random, repeated_random, 4_direction')
    parser.add_argument('--agent_repeat', type=int, default=1)
    parser.add_argument('--save_root', type=str, default='/scratch/mohammad/palmer++')
    
    # Evaluation
    parser.add_argument('--model_address', type=str, default='episode_100.pth')
    

    # Parse hyperpatemeres

    args = parser.parse_args()

    run(args)

 