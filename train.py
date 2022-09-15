
import argparse
from models import QNet, ReplayMemory


def train(config):
    net = QNet(config)
    rb  = ReplayMemory(config)

    

    for episode in range(config.episode):
        
        for i in range(config.episode_len):


    pass 

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
    parser.add_argument('--obs_dim', type=tuple, default=(128, 128))
    parser.add_argument('--obs_channel', type=tuple, default=(128, 128))
    parser.add_argument('--latent_dim', type=int, default=4)
    parser.add_argument('--action_dim', type=int, default=4)
    parser.add_argument('--fc_dim', type=int, default=128)
    parser.add_argument('--reply_buffer_len', type=int, default=100)
    parser.add_argument('--episode', type=int, default=100)
    parser.add_argument('--episode_len', type=int, default=100)

    # Parse hyperpatemeres
    args = parser.parse_args()

    train(args)