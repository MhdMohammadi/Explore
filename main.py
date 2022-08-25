from random import Random
from Exploration import RandomExploration
import matplotlib.pyplot as plt
import argparse

if __name__ == '__main__':
    # Selecting Hyper-parameters
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--sim', type=str, default='habitat', help='habitat')
    parser.add_argument('--config_path', type=str, default='configs/datasets/pointnav/gibson.yaml')
    parser.add_argument('--steps', type=int, default=1000)
    parser.add_argument('--resolution', type=int, default=1024)
    parser.add_argument('--topdown_map_path', type=str, default='images/topdown_map')
    parser.add_argument('--seen_map_path', type=str, default='images/seen_map')
    
    # Parse hyperpatemeres
    args = parser.parse_args()

    # Create an explorer
    exp = RandomExploration(args)

    # Start the exploration process
    exp.start(topdown_map_path=args.topdown_map_path, seen_map_path=args.seen_map_path)