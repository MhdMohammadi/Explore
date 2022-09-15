from cgitb import reset
from turtle import forward
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet


## Discription of the net for learning Q-values
# self.image_encoder: an encoder that takes observastions (images) and maps it to a vector
# self.sa_encoder: a state-action encoder which takes [self.image_encoder(s_t), a_t], and outputs phi(s_t, a_t)
# self.s_ecnoder: a state encoder which takes self.image_encoder(s_g), and outputs psi(s_g)
# final output is the inner product of phi(s_t, a_t) and psi(s_g)
# This structre is inspired by "Contrastive Learning as Goal-Conditioned Reinforcement Learning"

class QNet(nn.Module):
    
    def __init__(self, config) -> None:
        super().__init__()

        # output the new dimensions after a cnn layer
        def get_new_dim(dim, kernel, stride, padding):
            return (dim - kernel + 2 * padding + stride - 1) / stride 

        # image encoder -> equal to atari torso from acme.jax.networks.atari
        self.image_encoder = self.state_backbone = nn.Sequential(
            nn.Conv2d(config['obs_channel'], 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )

        new_dim = get_new_dim(config['obs_dim'], 8, 4, 0)
        new_dim = get_new_dim(config['obs_dim'], 4, 2, 0)
        new_dim = get_new_dim(config['obs_dim'], 3, 1, 0)
        new_dim = new_dim * 64

        ## state-action encode
        self.sa_encoder = nn.Sequential(
            nn.Linear(new_dim + config['action_dim'], config['fc_dim']), nn.ReLU(), 
            nn.Linear(config['fc_dim'], config['fc_dim']), nn.ReLU(), 
            nn.Linear(config['fc_dim'], config['fc_dim']), nn.ReLU(), 
            nn.Linear(config['fc_dim'], config['latet_dim']))

        ## state encoder
        self.s_encoder = nn.Sequential(
            nn.Linear(new_dim, config['fc_dim']), nn.ReLU(), 
            nn.Linear(config['fc_dim'], config['fc_dim']), nn.ReLU(), 
            nn.Linear(config['fc_dim'], config['fc_dim']), nn.ReLU(), 
            nn.Linear(config['fc_dim'], config['latet_dim']))


    def forward(self, current_states, current_actions, goal_states):
        current_states = self.image_encoder(current_states)
        goal_states = self.image_encoder(goal_states)

        joint_current_states_actions = torch.cat([current_states, current_actions], axis=-1)
        sa_repr = self.sa_encoder(joint_current_states_actions)
        s_repr = self.s_encoder(goal_states)

        return torch.inner(sa_repr, s_repr)


net = QNet()
