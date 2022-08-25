
import habitat
from habitat.core.env import Env

# This agent takes random actions in an environment in order to explroe the state space
class RandomAgent:
    def __init__(self, env: habitat.Env, args):
        self.env = env
        self.actions = []
        self.observations = []
        self.states = []
        self.mode = args.agent_mode
        self.last_action = None
        self.max_repeat = args.agent_repeat
        self.current_repeat = 0
        self.action_space = []
        self.initialize_action_space()

    # Only works for habiatat
    def initialize_action_sapce(self):
        self.action_space.append(self.get_action('right'))
        self.action_space.append(self.get_action('left'))
        if self.mode == 'random':
            self.action_space.append(self.get_action('forward'))

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

    # left, right, forward
    def get_action(self, direction:str):
        if direction == 'forward':
            return {'action': 'MOVE_FORWARD', 'action_args': None}
        if direction == 'right':
            return {'action': 'TURN_RIGHT', 'action_args': None}
        if direction == 'left':
            return {'action': 'TURN_LEFT', 'action_args': None}

    # Only habitat
    # TODO: You can create a set without stop, and make this random sampling faster!
    def get_random_action(self):
        if self.mode == 'random':
            # Choose a non-stop action
            a = None
            while a is None or a['action'] == 'STOP':
                a = self.env.action_space.sample()
        if self.mode == 'repeated_random':
            if self.last_action == None:
                while True:
                    self.last_action 

            pass
        return a
    
    # Only Habitat
    def get_state(self):
        return self.env.sim.get_agent(0).get_state()
        
    # TODO: Different ways of getting a random action
