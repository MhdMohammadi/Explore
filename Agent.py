
import habitat
from habitat.core.env import Env

# This agent takes random actions in an environment in order to explroe the state space
class RandomAgent:
    def __init__(self, env: habitat.Env):
        self.env = env
        self.actions = []
        self.observations = []
        self.states = []
        
    def take_action(self):
        # select an action for this step
        action = self.get_random_action()
        self.actions.append(action)

        # If this is the first action, current state and observation has to be stored
        if len(self.observations) == 0:
            self.observations.append(self.env.render(mode='rgb'))
            self.states.append(self.get_state())

        # execute the selected action     
        obs = self.env.step(action)
        self.observations.append(obs['rgb'])
        
        # This thing only works on habitat, comment if you are using other simulators
        self.states.append(self.get_state())
        return obs

    # Only habitat
    def get_random_action(self):
        # Choose a non-stop action
        a = None
        while a is None or a['action'] == 'STOP':
            a = self.env.action_space.sample()
        return a
    
    # Only Habitat
    def get_state(self):
        return self.env.sim.get_agent(0).get_state()
        
    # TODO: Different ways of getting a random action
