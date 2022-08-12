from glob import glob
import habitat
from habitat.utils.visualizations import maps

import matplotlib.pyplot as plt
import os

# Single-tone pattern is used to avoid "kernel dying problem"
main_env = None

class Environment():

    # This function is creating an environment based on given inputs
    # This function is assuming that habitat-lab and project's main folder are in the same directory
    def get_environment(sim=None, config_path=None):
        global main_env
        if sim == 'habitat':
            if main_env is not None:
                main_env.close()
            os.chdir('../habitat-lab')
            main_env = habitat.Env(config=habitat.get_config(config_path))
            main_env.reset()
            os.chdir('../EPFL_Summer_Internship')
        return main_env

    # This only works for habitat sim
    def get_topdown_map(env, resolution):
        top_down_map = maps.get_topdown_map_from_sim(
            env.sim, map_resolution=resolution
        )
        return top_down_map

    +
        meters_per_pixel = maps.calculate_meters_per_pixel(map_resolution=1024, pathfinder=env.sim.pathfinder)
        print(meters_per_pixel)


