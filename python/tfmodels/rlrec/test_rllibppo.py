import os
import time
import argparse
from stable_baselines import PPO2
import ray.rllib.agents.ppo as ppo
from core.tfmodels.rlrec.movie_env import *

config = ppo.DEFAULT_CONFIG.copy()
config["num_gpus"] = 0
config["num_workers"] = 8
config["env"]=MovieEnv
trainer = ppo.PPOTrainer(config=config)

def main(_):
  env = MovieEnv(config)
  episode_reward_p = 0
  import time
  begin = time.time()
  try:
    nepisode = 0
    step = 0
    while nepisode < 1:
      step = step + 1
      obs, reward, done, info = env.step([])
      if done:
        nepisode = nepisode + 1
        print("episode:{}".format(nepisode), "score:{}".format(obs["score"]))
        env.reset()
    end = time.time()
    print("total time: %.3f" % ((end-begin)/60), "min")
  except KeyboardInterrupt:
    print('Game stopped, writing dump...')
    exit(1)


if __name__ == '__main__':
  main()
