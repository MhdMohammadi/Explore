import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Define hyperparameters.')
    parser.add_argument('--obs_dim', type=tuple, default=(128, 128))
    parser.add_argument('--obs_channel', type=tuple, default=(128, 128))
    parser.add_argument('--action_dim', type=int, default=4)
    parser.add_argument('--fc_dim', type=int, default=128)
    


 

