from calendar import c
from xxlimited import new
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models.resnet as resnet
from collections import namedtuple, deque
import numpy as np
import random

## Discription of the net for learning Q-values
# self.image_encoder: an encoder that takes observastions (images) and maps it to a vector
# self.sa_encoder: a state-action encoder which takes [self.image_encoder(s_t), a_t], and outputs phi(s_t, a_t)
# self.s_ecnoder: a state encoder which takes self.image_encoder(s_g), and outputs psi(s_g)
# final output is the inner product of phi(s_t, a_t) and psi(s_g)
# This structre is inspired by "Contrastive Learning as Goal-Conditioned Reinforcement Learning"

class QNet(nn.Module):
    
    def __init__(self, config, device) -> None:
        super().__init__()
        
        self.action_dim = config.action_dim
        self.device = device
        
        # output the new dimensions after a cnn layer
        def get_new_dim(dim, kernel, stride, padding):
            return (dim - kernel + 1 + 2 * padding + stride - 1) // stride 

        # image encoder -> equal to atari torso from acme.jax.networks.atari
        self.image_encoder = self.state_backbone = nn.Sequential(
            nn.Conv2d(config.obs_channel, 32, 8, 4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, 1), nn.ReLU(),
            nn.Flatten(),
        )

        new_dim = get_new_dim(np.array(config.obs_dim), 8, 4, 0)
        new_dim = get_new_dim(new_dim, 4, 2, 0)
        new_dim = get_new_dim(new_dim, 3, 1, 0)
        new_dim = np.prod(new_dim) * 64 

        ## state-action encode
        self.sa_encoder = nn.Sequential(
            nn.Linear(new_dim + config.action_dim, config.fc_dim), nn.ReLU(), 
            nn.Linear(config.fc_dim, config.fc_dim), nn.ReLU(), 
            nn.Linear(config.fc_dim, config.fc_dim), nn.ReLU(), 
            nn.Linear(config.fc_dim, config.latent_dim))

        ## state encoder
        self.s_encoder = nn.Sequential(
            nn.Linear(new_dim, config.fc_dim), nn.ReLU(), 
            nn.Linear(config.fc_dim, config.fc_dim), nn.ReLU(), 
            nn.Linear(config.fc_dim, config.fc_dim), nn.ReLU(), 
            nn.Linear(config.fc_dim, config.latent_dim))


    def forward(self, current_states, current_actions, goal_states):
        current_states = current_states.permute(0, 3, 1, 2)
        goal_states = goal_states.permute(0, 3, 1, 2)
        
        current_states = self.image_encoder(current_states)
        goal_states = self.image_encoder(goal_states)

        joint_current_states_actions = torch.cat([current_states, current_actions], axis=-1)
        sa_repr = self.sa_encoder(joint_current_states_actions)
        s_repr = self.s_encoder(goal_states)

        return torch.inner(sa_repr, s_repr)

    def get_best_action(self, current_states, goal_states, greedy=False):
        all_actions = F.one_hot(torch.arange(0, self.action_dim)).to(self.device)
        batch_size = current_states.shape[0]

        results = torch.zeros((batch_size, self.action_dim)).to(self.device)
        with torch.no_grad():
            for i in range(self.action_dim):
                actions = all_actions[i].repeat(batch_size, 1)
                results[:, i] = self.forward(current_states, actions, goal_states)

        return results.argmax(axis=1)


# Transitions stored in the reply buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'episode', 'idx'))

# Reply buffer to save recent transitions
class ReplayMemory(object):

    def __init__(self, config):
        self.memory = deque([], maxlen=config.reply_buffer_len)
        self.end_idx = {}

    def push(self, args):
        """Save a transition"""
        data = Transition(*args)
        self.end_idx[data.episode] = data.idx
        self.memory.append(data)

    def sample(self, batch_size, p):
        indices = np.array(random.sample(range(0, len(self.memory)), k=batch_size))

        states = [self.memory[id].state for id in indices]
        actions = [self.memory[id].action for id in indices] 

        max_delta_indices = np.array([self.end_idx[self.memory[id].episode] - self.memory[id].idx for id in indices])
        delta_indices = np.random.geometric(p, size=batch_size)
        positive_indices = indices + np.minimum(delta_indices, max_delta_indices)

        negative_indices = np.random.randint(0, len(self.memory), size=batch_size)

        positive_states = [self.memory[id].state for id in positive_indices]
        negative_states = [self.memory[id].state for id in negative_indices]

        return [states, actions, positive_states, negative_states]

    def __len__(self):
        return len(self.memory)
