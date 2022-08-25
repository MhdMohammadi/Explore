from os import access
from habitat.utils.visualizations import maps
import habitat
import Agent
import numpy as np
import matplotlib.pyplot as plt

# This file provides useful tools for visualization, demonstration, etc.

# This only works for habitat sim
def get_topdown_map(env: habitat.Env, resolution):
    top_down_map = maps.get_topdown_map_from_sim(
        env.sim, map_resolution=resolution
    )
    return top_down_map

def save_topdown_map(env: habitat.Env, resolution, address):
    plt.imsave(address, get_topdown_map(env, resolution))

def get_seen_map(env: habitat.Env, agent: Agent.RandomAgent, resolution):
    meters_per_pixel = maps.calculate_meters_per_pixel(map_resolution=resolution, pathfinder=env.sim.pathfinder)
    topdown_map = get_topdown_map(env, resolution)
    seen_map = np.zeros(topdown_map.shape)

    for state in agent.states:
        pos = state.position # Y Z X ?! 
        normalized_pos = ((pos - env.sim.pathfinder.get_bounds()[0]) / meters_per_pixel).astype(int)
        seen_map[normalized_pos[2], normalized_pos[0]] = 1
    
    seen_map = seen_map / seen_map.max()
    return seen_map

def save_seen_map(env, agent, resolution, address):
    plt.imsave(address, get_seen_map(env, agent, resolution))