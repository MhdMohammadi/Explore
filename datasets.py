from torch.utils.data import Dataset, DataLoader
from collections import namedtuple, deque
import torch
import numpy as np
import random
import os

# Transitions stored in the reply buffer
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'episode', 'idx'))

# It is useful when one wants to save all the dataset on RAM 
# Reply buffer to save recent transitions
class ReplayMemory(object):

    def __init__(self, config):
        self.memory = deque([], maxlen=config.dataset_max_size)
        self.end_idx = {}

    def push(self, args):
        """Save a transition"""
        data = Transition(*args)
        self.end_idx[data.episode] = data.idx
        self.memory.append(data)

    def sample(self, batch_size, p):
        # transitions = random.sample(self.memory, batch_size) 
        # return Transition(*zip(*transitions))

        # return random.sample(self.memory, batch_size)
        # return None

        indices = np.array(random.sample(range(0, len(self.memory)), k=batch_size))

        states = [self.memory[id].state for id in indices]
        actions = [self.memory[id].action for id in indices] 

        max_delta_indices = np.array([self.end_idx[self.memory[id].episode] - self.memory[id].idx for id in indices])
        delta_indices = np.random.geometric(p, size=batch_size)
        positive_indices = indices + np.minimum(delta_indices, max_delta_indices)

        negative_indices = np.random.randint(0, len(self.memory), size=batch_size)

        positive_states = [self.memory[id].state for id in positive_indices]
        negative_states = [self.memory[id].state for id in negative_indices]
      

        # print([self.memory[id].episode for id in indices])
        # for i in range(0, batch_size, 20):
        #     plt.imsave(f'images/pos_{i}.jpg', positive_states[i][0, :, :, :3].numpy())
        #     plt.imsave(f'images/normal_{i}.jpg', states[i][0, :, :, :3].numpy())
        #     plt.imsave(f'images/neg_{i}.jpg', negative_states[i][0, :, :, :3].numpy())
            
        # positive_states = [torch.from_numpy(self.memory[id].state) for id in positive_indices]
        # negative_states = [torch.from_numpy(self.memory[id].state) for id in negative_indices]

        return [states, np.array(actions), positive_states, negative_states]

        # return [np.array(states), np.array(actions), np.array(positive_states), np.array(negative_states)]

    def __len__(self):
        return len(self.memory)


# When a very fast drive is available, it is better to avoid storing all the data in RAM
# Dataset
class CustomizedDataset(Dataset):
    def __init__(self, config):
        self.current_size = config.dataset_max_size
        self.max_size = config.dataset_max_size
        self.save_root = config.save_root
        self.obs_dir = f'{self.save_root}/obs'
        os.mkdir(self.obs_dir)
        self.actions = deque([], maxlen=config.dataset_max_size)
        self.idx = 0
        self.end_idx = {}
        self.p = 1 - config.gamma

    def __len__(self):
        return self.current_size

    def __getitem__(self, idx):
        current_state = torch.load(f'{self.save_root}/s_{idx}.pt')
        action = self.actions[idx]
        neg_idx = np.random.randint(low=0, high=self.current_size)
        negative_state = torch.load(f'{self.save_root}/s_{neg_idx}.pt')
        # pos_idx = min(np.random.geometric(self.p), 
        ...

        # pass
    
    def push(self, current_state, action, episode):
        self.actions.append(action)
        self.end_idx[episode] = idx
        torch.save(f'{self.save_root}/s_{self.idx}.pt', current_state)
        self.idx = (self.idx + 1) % self.max_size
    
    