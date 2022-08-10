
import habitat

# This agent takes random actions in an environment in order to explroe the state space
class RandomAgent:
    def __init__(self, env):
        self.env = env
        self.actions = []
        self.observations = []
        self.states = []
        
    def take_action(self):
        action = self.get_random_action()
        self.actions.append(action)
        obs = self.env.step(action)
        self.observations.append(obs['rgb'])
        # This thing only works on habitat, comment if you are using other simulators
        self.states.append(self.get_state())
        return obs

    def get_random_action(self):
        return self.env.action_space.sample()
    
    # Only Habitat
    def get_state(self):
        return self.env.sim.get_agent(0).get_state()
        
    # TODO: Different ways of getting a random action
