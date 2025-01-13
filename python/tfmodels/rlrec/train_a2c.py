import ray
import ray.rllib.agents.a3c.a2c as a2c
from ray.tune.logger import pretty_print
from core.tfmodels.rlrec.movie_env import *
import time
import sys

#ray.init()
from zoo.ray import RayContext
from zoo import init_spark_on_local

spark_conf = {"spark.executor.memory": "24g", "spark.driver.memory": "24g"}
sc = init_spark_on_local(cores=8, conf=spark_conf)
ray_ctx = RayContext(sc=sc, object_store_memory="4g")
ray_ctx.init()

env_conf= EnvConfig({'user_max': 6040,
          'movie_max': 3952,
          'ncf_dim':20
          })

trainer_conf = a2c.A2C_DEFAULT_CONFIG.copy()
trainer_conf["env_config"] = env_conf._values
# print(sys.argv[1])
# lr = float(sys.argv[1]) * 0.0005
lr = 0.0005
trainer_conf["lr"] = lr
trainer_conf["num_gpus"] = 0
trainer_conf["num_workers"] = 8
trainer_conf["env"]=MovieEnv

trainer = a2c.A2CTrainer(config=trainer_conf)
# Can optionally call trainer.restore(path) to load a checkpoint.

t1 = time.time()
for i in range(1, 10000):
   # Perform one iteration of training the policy with PPOvi run
   result = trainer.train()
   print(pretty_print(result))

   if i % 50 == 0:
       checkpoint = trainer.save()
       print("checkpoint saved at", checkpoint)
t2 = time.time()
print("total time", t2-t1)
# agent = ppo.PPOTrainer(config=config)
# agent.restore("/Users/guoqiong/ray_results/PPO_MovieEnv_2021-01-15_12-19-47e3t_a6pt/checkpoint_10/checkpoint-10")
# env = MovieEnv(config)
# # run until episode ends
# for i in range(100):
#     episode_reward = 0
#     done = False
#     obs = env.reset()
#     while not done:
#         action = agent.compute_action(obs)
#         obs, reward, done, info = env.step(action)
#         episode_reward += reward
#         # if reward > 0:
#         print(reward, done, env._step )
#         print("episode_reward", episode_reward)
#

ray_ctx.stop()
sc.stop()