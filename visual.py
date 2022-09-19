from os import access
from habitat.utils.visualizations import maps
import habitat
import Agent
import numpy as np
import matplotlib.pyplot as plt
import cv2

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
    seen_map = np.zeros(topdown_map.shape).astype(np.int)

    for state in agent.states:
        pos = state.position # Y Z X ?! 
        normalized_pos = ((pos - env.sim.pathfinder.get_bounds()[0]) / meters_per_pixel).astype(int)
        seen_map[normalized_pos[2], normalized_pos[0]] = 1
    
    # normalized_seen_map = np.zeros(seen_map.shape)
    # values = np.unique(seen_map)
    # print(values)
    # for i, value in enumerate(values):
    #     normalized_seen_map[seen_map == value] = (i + 1) / (len(values) + 1)
    # seen_map = normalized_seen_map

    # print(seen_map.max())
    # print(np.unique(seen_map))
    # seen_map = np.log(seen_map + 1)
    return seen_map

def save_seen_map(env, agent, resolution, address):
    plt.imsave(address, get_seen_map(env, agent, resolution))
    

def get_unseen_map(env: habitat.Env, resolution):
    topdown_map = get_topdown_map(env, resolution)
    unseen_map = np.zeros(topdown_map.shape)
    return unseen_map

def put_mark_on_map(map, env):
    meters_per_pixel = maps.calculate_meters_per_pixel(map_resolution=1024, pathfinder=env.sim.pathfinder)
    pos = env.sim.get_agent_state().position - env.sim.pathfinder.get_bounds()[0]
    x, y = int(pos[0] / meters_per_pixel), int(pos[2] / meters_per_pixel)
    cv2.circle(map, (x, y), 3, 1, 4)
    return map


    