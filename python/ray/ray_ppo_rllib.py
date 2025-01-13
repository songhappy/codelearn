from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gym
from ray.rllib.agents.ppo import PPOTrainer, DEFAULT_CONFIG
from ray.tune.logger import pretty_print

import ray

# ray.shutdown()
ray.init(num_cpus=4, ignore_reinit_error=True, log_to_driver=False)
config = DEFAULT_CONFIG.copy()
config['num_workers'] = 1
config['num_sgd_iter'] = 30
config['sgd_minibatch_size'] = 128
config['model']['fcnet_hiddens'] = [100, 100]
config[
    'num_cpus_per_worker'] = 0  # This avoids running out of resources in the notebook environment when this cell is re-executed

agent1 = PPOTrainer(config, 'CartPole-v0')
for i in range(2):
    result = agent1.train()
    print(pretty_print(result))

config2 = DEFAULT_CONFIG.copy()
config2['num_workers'] = 4
config2['num_sgd_iter'] = 30
config2['sgd_minibatch_size'] = 128
config2['model']['fcnet_hiddens'] = [100, 100]
config2['num_cpus_per_worker'] = 0

agent2 = PPOTrainer(config2, 'CartPole-v0')
for i in range(2):
    result = agent2.train()
    print(pretty_print(result))

checkpoint_path = agent2.save()
print(checkpoint_path)
trained_config = config2.copy()

test_agent = PPOTrainer(trained_config, 'CartPole-v0')
test_agent.restore(checkpoint_path)

env = gym.make('CartPole-v0')
state = env.reset()
done = False
cumulative_reward = 0

while not done:
    action = test_agent.compute_action(state)
    state, reward, done, _ = env.step(action)
    cumulative_reward += reward
print(cumulative_reward)

# tensorboard --logdir="/Users/guoqiong/ray_results/PPO_CartPole-v0_2020-04-29_18-58-22yq1yq16u/" --host=0.0.0.0
