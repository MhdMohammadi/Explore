from random import Random
from Exploration import RandomExploration
import matplotlib.pyplot as plt
import argparse
import os

if __name__ == '__main__':
    # Selecting Hyper-parameters
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--sim', type=str, default='habitat', help='habitat')
    parser.add_argument('--config_path', type=str, default='configs/datasets/pointnav/gibson.yaml')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--topdown_map_path', type=str, default='images/topdown_map')
    parser.add_argument('--seen_map_path', type=str, default='images/seen_map')
    parser.add_argument('--agent_mode', type=str, default='repeated_random', help='random, repeated_random, 4_direction')
    parser.add_argument('--agent_repeat', type=int, default=1)
    parser.add_argument('--save_root', type=str, default='/scratch/mohammad/explore')

    # Parse hyperpatemeres
    args = parser.parse_args()

    # Create root directory
    os.makedirs(args.save_root, exist_ok=True)

    # Create an explorer
    exp = RandomExploration(args)

    # Start the exploration process
    exp.start(topdown_map_path=args.topdown_map_path, seen_map_path=args.seen_map_path)
