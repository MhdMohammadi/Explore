from cgi import print_arguments
from habitat.core.simulator import Observations
import numpy as np
import habitat
from habitat.core.env import Env
from tqdm import tqdm
import os
from Environment import get_action

# This agent takes random actions in an environment in order to explroe the state space
class RandomAgent:
    def __init__(self, env: habitat.Env, args):
        self.env = env
        self.actions = []
        self.observations = []
        self.states = []
        self.mode = args.agent_mode
        self.last_action = None
        self.max_repeats = args.agent_repeat
        self.remaining_repeats = 0
        self.action_space = []
        self.initialize_action_space()
        self.save = args.agent_save

    # Only works for habiatat
    def initialize_action_space(self):
        if self.mode == '4_direction':
            self.action_space.append(get_action('forward'))
            self.action_space.append(get_action('left'))
            self.action_space.append(get_action('right'))
            self.action_space.append(get_action('backward'))
        else:
            self.action_space.append(get_action('turn_right'))
            self.action_space.append(get_action('turn_left'))
            if self.mode == 'random':
                self.action_space.append(get_action('forward'))

    def take_action(self, action):
        if action is None:
            # select an action for this step
            action = self.get_random_action()

        if self.save:
            self.actions.append(action)

            # If this is the first action, current state and observation has to be stored
            if len(self.observations) == 0:
                self.observations.append(self.get_full_obs())
                # self.observations.append(self.env.render(mode='rgb'))
                self.states.append(self.get_state())

        # execute the selected action     
        obs = self.env.step(action)

        if self.save:
            self.observations.append(self.get_full_obs())
            # self.observations.append(obs['rgb'])
            
            # This thing only works on habitat, comment if you are using other simulators
            self.states.append(self.get_state())

        return obs, action

    # Only habitat
    # TODO: Different ways of getting a random action
    def get_random_action(self):
        if self.mode == '4_direction':
            return np.random.choice(self.action_space) 

        if self.mode == 'random':
            return np.random.choice(self.action_space)

        if self.mode == 'repeated_random':
            if self.last_action is None:
                self.last_action = np.random.choice(self.action_space)
                self.remaining_repeats = np.random.randint(1, self.max_repeats + 1)
            if self.remaining_repeats == 0:
                self.last_action = None
                return get_action('forward')
            self.remaining_repeats -= 1
            return self.last_action
    
    # Only Habitat
    def get_state(self):
        return self.env.sim.get_agent(0).get_state()
        
    def get_full_obs(self):
        res = None
        steps = 1
        for i in range(steps * 4):
            if i % steps == 0:
                if res is None:
                    res = self.env.render(mode='rgb')
                else:
                    res = np.concatenate((res, self.env.render(mode='rgb')), axis=2) # shape format : HWC 
            self.env.step(get_action('turn_right'))
        return res.astype(np.float32)
    
    def set_position(self, position):
        agent_state = self.get_state()
        agent_state.position = position
        self.env.sim.get_agent(0).set_state(agent_state)
    
    # # address has to be absolute !
    # def save_attributes(self, address):
    #     print('--------')
    #     print('saving actions list has been started')
    #     with open('actions.npy', 'wb') as f:
    #         np.save(f, self.actions)
    #     print('--------')
        
    #     last_dir = os.getcwd()
    #     os.chdir(f'{address}')
 
    #     os.mkdir(f'{address}/observations')
    #     os.chdir('observations')
    #     print('--------')
    #     print('saving observations has been started')
    #     for i in tqdm(range(len(self.observations))):
    #         with open(f'obs_{i}.npy', 'wb') as f:
    #             np.save(f, self.observations[i])
    #     print('--------')

    #     os.chdir('..')
    #     os.mkdir(f'{address}/states')
    #     print('--------')
    #     print('saving states has been started')
    #     for i in tqdm(range(len(self.states))):
    #         with open(f'state_{i}.npy', 'wb') as f:
    #             np.save(f, self.states[i])
    #     print('--------')
    #     os.chdir('..')

    #     os.chdir(last_dir)

    #     print('!! saving has been successfully finished !!')

    #     pass
    