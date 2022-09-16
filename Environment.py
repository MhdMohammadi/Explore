from habitat.core.dataset import T
import matplotlib.pyplot as plt
import os

import attr
import numpy as np
import os

import habitat
import habitat_sim
from habitat.utils.visualizations import maps
from habitat.sims.habitat_simulator.actions import (
    HabitatSimActions,
    HabitatSimV1ActionSpaceConfiguration,
)
from habitat.tasks.nav.nav import SimulatorTaskAction

import matplotlib.pyplot as plt

# Single-tone pattern is used to avoid "kernel dying" problem
main_env = None
MOVE_DIS = 0.25

# This function is creating an environment based on the given inputs
#### This function is assuming that habitat-lab and project's main folder are in the same directory #### 
def get_environment(sim=None, config_path=None):
    global main_env
    if sim == 'habitat':
        if main_env is not None:
            main_env.close()
        os.chdir('../habitat-lab')
        config_file = habitat.get_config(config_path)
        main_env = create_env(config_file)
        main_env.reset()
        os.chdir('../Explore')
    return main_env

# All actions -> 1 to 4 have to be moving in four directions
all_actions = [{'action': 'MOVE_FORWARD', 'action_args': None}, 
            {'action': 'MOVE_LEFT', 'action_args': None}, 
            {'action': 'MOVE_RIGHT', 'action_args': None}, 
            {'action': 'MOVE_BACKWARD', 'action_args': None}, 
            {'action': 'TURN_RIGHT', 'action_args': None}, 
            {'action': 'TURN_LEFT', 'action_args': None}]

# get action by name
def get_action(direction:str):
    if direction == 'forward':
        return {'action': 'MOVE_FORWARD', 'action_args': None}
    if direction == 'left':
        return {'action': 'MOVE_LEFT', 'action_args': None}
    if direction == 'right':
        return {'action': 'MOVE_RIGHT', 'action_args': None}
    if direction == 'backward':
        return {'action': 'MOVE_BACKWARD', 'action_args': None}
    if direction == 'turn_right':
        return {'action': 'TURN_RIGHT', 'action_args': None}
    if direction == 'turn_left':
        return {'action': 'TURN_LEFT', 'action_args': None}

def get_id_by_action(action):
    global all_actions
    for id, a in enumerate(all_actions):
        if action['action'] == a['action']:
            return id

def get_action_by_id(id):
    global all_actions
    return all_actions[id]

def is_done(current_loc, goal_loc) -> bool:
    dis = ((current_loc - goal_loc)**2).sum()
    threshold = 0.01
    return (dis < threshold)


# The functions below are provided to add 3 new actions to habitat's environment -> Move left - right - backwards
@attr.s(auto_attribs=True, slots=True)
class NoisyMoveActuationSpec:
    move_amount: float
    # Classic strafing is to move perpendicular (90 deg) to the forward direction
    move_angle: float = 90.0
    noise_amount: float = 0.05


def _move_impl(
    scene_node: habitat_sim.SceneNode,
    move_amount: float,
    move_angle: float,
    noise_amount: float,
):
    forward_ax = (
        np.array(scene_node.absolute_transformation().rotation_scaling())
        @ habitat_sim.geo.FRONT
    )
    move_angle = np.deg2rad(move_angle)
    move_angle = np.random.uniform(
        (1 - noise_amount) * move_angle, (1 + noise_amount) * move_angle
    )

    rotation = habitat_sim.utils.quat_from_angle_axis(
        move_angle, habitat_sim.geo.UP
    )
    move_ax = habitat_sim.utils.quat_rotate_vector(rotation, forward_ax)

    move_amount = np.random.uniform(
        (1 - noise_amount) * move_amount, (1 + noise_amount) * move_amount
    )
    scene_node.translate_local(move_ax * move_amount)


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyMoveLeft(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: NoisyMoveActuationSpec,
    ):
        # print(f"moving left with noise_amount={actuation_spec.noise_amount}")
        _move_impl(
            scene_node,
            actuation_spec.move_amount,
            actuation_spec.move_angle,
            actuation_spec.noise_amount,
        )


@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyMoveRight(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: NoisyMoveActuationSpec,
    ):
        # print(f"moving right with noise_amount={actuation_spec.noise_amount}")
        _move_impl(
            scene_node,
            actuation_spec.move_amount,
            -actuation_spec.move_angle,
            actuation_spec.noise_amount,
        )

@habitat_sim.registry.register_move_fn(body_action=True)
class NoisyMoveBackward(habitat_sim.SceneNodeControl):
    def __call__(
        self,
        scene_node: habitat_sim.SceneNode,
        actuation_spec: NoisyMoveActuationSpec,
    ):
        # print(f"moving back with noise_amount={actuation_spec.noise_amount}")
        _move_impl(
            scene_node,
            actuation_spec.move_amount,
            2*actuation_spec.move_angle,
            actuation_spec.noise_amount,
        )


@habitat.registry.register_action_space_configuration
class NoNoiseMove(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.MOVE_LEFT] = habitat_sim.ActionSpec(
            "noisy_move_left",
            NoisyMoveActuationSpec(MOVE_DIS, noise_amount=0.0),
        )
        config[HabitatSimActions.MOVE_RIGHT] = habitat_sim.ActionSpec(
            "noisy_move_right",
            NoisyMoveActuationSpec(MOVE_DIS, noise_amount=0.0),
        )
        config[HabitatSimActions.MOVE_BACKWARD] = habitat_sim.ActionSpec(
            "noisy_move_backward",
            NoisyMoveActuationSpec(MOVE_DIS, noise_amount=0.0),
        )
        return config


@habitat.registry.register_action_space_configuration
class NoiseMove(HabitatSimV1ActionSpaceConfiguration):
    def get(self):
        config = super().get()

        config[HabitatSimActions.MOVE_LEFT] = habitat_sim.ActionSpec(
            "noisy_move_left",
            NoisyMoveActuationSpec(MOVE_DIS, noise_amount=0.05),
        )
        config[HabitatSimActions.MOVE_RIGHT] = habitat_sim.ActionSpec(
            "noisy_move_right",
            NoisyMoveActuationSpec(MOVE_DIS, noise_amount=0.05),
        )
        config[HabitatSimActions.MOVE_BACKWARD] = habitat_sim.ActionSpec(
            "noisy_move_backward",
            NoisyMoveActuationSpec(MOVE_DIS, noise_amount=0.05),
        )
        return config


@habitat.registry.register_task_action
class MoveLeft(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "move_left"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.MOVE_LEFT)


@habitat.registry.register_task_action
class MoveRight(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "move_right"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.MOVE_RIGHT)

@habitat.registry.register_task_action
class MoveBackward(SimulatorTaskAction):
    def _get_uuid(self, *args, **kwargs) -> str:
        return "move_backward"

    def step(self, *args, **kwargs):
        return self._sim.step(HabitatSimActions.MOVE_BACKWARD)

# This function creates a new environemtn with three new actions which are hardcoded
def create_env(config):
    HabitatSimActions.extend_action_space("MOVE_LEFT")
    HabitatSimActions.extend_action_space("MOVE_RIGHT")
    HabitatSimActions.extend_action_space("MOVE_BACKWARD")

    config.defrost()
    config.TASK.POSSIBLE_ACTIONS = config.TASK.POSSIBLE_ACTIONS + [
        "MOVE_LEFT",
        "MOVE_RIGHT",
        "MOVE_BACKWARD",
    ]
    config.TASK.POSSIBLE_ACTIONS.remove('STOP')
    config.TASK.ACTIONS.MOVE_LEFT = habitat.config.Config()
    config.TASK.ACTIONS.MOVE_LEFT.TYPE = "MoveLeft"
    config.TASK.ACTIONS.MOVE_RIGHT = habitat.config.Config()
    config.TASK.ACTIONS.MOVE_RIGHT.TYPE = "MoveRight"
    config.TASK.ACTIONS.MOVE_BACKWARD = habitat.config.Config()
    config.TASK.ACTIONS.MOVE_BACKWARD.TYPE = "MoveBackward"
    config.SIMULATOR.ACTION_SPACE_CONFIG = "NoNoiseMove"
    config.freeze()
    
    return habitat.Env(config=config)
