import os
import time
import argparse
from stable_baselines import PPO2
from core.tfmodels.rlrec.movie_env import *

config = {}
def train(args):
    if os.path.exists(args.model_path):
        env = MovieEnv(config)
        # model = A2C.load(args.model_path, env=env, verbose=1)
        model = PPO2("MlpPolicy", env, verbose=1)
    else:
        env = MovieEnv(config)
        # env = make_vec_env(lambda: env_base, n_envs=1)
        # model = A2C("MlpPolicy", env, verbose=1)
        model = PPO2("MlpPolicy", env, verbose=1)

    start = time.time()
    model.learn(total_timesteps=args.steps, reset_num_timesteps=False)
    end = time.time()

    print(round(end - start, 2))
    print("SUCCESS")
    model.save(args.model_path)

def infer():
    pass

def main():
    parser = argparse.ArgumentParser(description="process for movie recommendations")
    parser.add_argument('--policy', choices=['cnn', 'lstm', 'lnlstm'], default='cnn', help='Policy architecture')
    parser.add_argument('--lr_schedule', choices=['constant', 'linear'], default='constant',
                        help='Learning rate schedule')
    parser.add_argument('--model_path', default='./model/movie_rec')
    parser.add_argument('--steps', default=100000)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
