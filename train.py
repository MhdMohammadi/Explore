import habitat
import argparse
from tkinter.tix import Tree
from models import QNet, ReplayMemory
from Environment import get_environment
from Agent import RandomAgent


def train(config):
    net = QNet(config)
    rb  = ReplayMemory(config)
    env = get_environment(config.sim, config.config_path)
    env: habitat.Env
    agent = RandomAgent(env, config)
    
    for episode in range(config.episode):
        env.reset()
        current_state = env.render(mode='rgb')

        initial_loc = env.sim.sample_navigable_point()
        agent_state = env.sim.get_agent_state()
        agent_state.position = initial_loc
        env.sim.get_agent(0).set_state(agent_state)
        final_loc = env.sim.sample_navigable_point()

    #     for step in range(config.episode_len):
    #         ## TODO find the action that we have to take in this step
    #         action = None

    #         next_state, action = agent.take_action(action)
    #         next_state = next_state['rgb']

    #         rb.push(current_state, action, next_state, episode, step)
    #         ## TODO: if we are done, terminate the trajectory
            
    #         ## TODO: update the network

    #         current_state = next_state



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
    # Q-value learning network structure
    parser.add_argument('--obs_dim', type=tuple, default=(128, 128))
    parser.add_argument('--obs_channel', type=tuple, default=4)
    parser.add_argument('--latent_dim', type=int, default=32)
    parser.add_argument('--action_dim', type=int, default=4)
    parser.add_argument('--fc_dim', type=int, default=128)

    # Reply buffer
    parser.add_argument('--reply_buffer_len', type=int, default=100)

    # Running configurations
    parser.add_argument('--episode', type=int, default=100)
    parser.add_argument('--episode_len', type=int, default=100)

    # Enivornment and Agent configurations
    parser.add_argument('--sim', type=str, default='habitat', help='habitat')
    parser.add_argument('--config_path', type=str, default='configs/datasets/pointnav/gibson.yaml')
    parser.add_argument('--agent_save', type=bool, default=True)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--topdown_map_path', type=str, default='images/topdown_map')
    parser.add_argument('--seen_map_path', type=str, default='images/seen_map')
    parser.add_argument('--agent_mode', type=str, default='4_direction', help='random, repeated_random, 4_direction')
    parser.add_argument('--agent_repeat', type=int, default=1)
    parser.add_argument('--save_root', type=str, default='/scratch/mohammad/explore')

    # Parse hyperpatemeres
    args = parser.parse_args()

    train(args)