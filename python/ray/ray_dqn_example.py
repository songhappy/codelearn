import random
import time
from collections import deque

import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

import ray

EPISODES = 1000


@ray.remote
class Runner:
    def __init__(self, nsteps):
        self.env = gym.make('CartPole-v1')
        state = self.env.reset()
        state_size = self.env.observation_space.shape[0]
        self.initial_state = np.reshape(state, [1, state_size])
        self.nsteps = nsteps

    def run(self, agent_id):
        self.agent = agent_id
        memory = []
        state = env.reset()
        state = np.reshape(state, [1, state_size])
        begin = time.time()
        for i in range(self.nsteps):
            # env.render()
            action = agent.act(state)
            next_state, reward, done, _ = env.step(action)
            reward = reward if not done else -10
            next_state = np.reshape(next_state, [1, state_size])
            memory.append([state, action, reward, next_state, done])
            state = next_state
            if done:
                state = env.reset()
                state = np.reshape(state, [1, state_size])
                # print("score: {}, e: {:.2}"
                #       .format(i, agent.epsilon))
                # break
        end = time.time()
        print("in worker", end - begin)
        return memory


# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        # Neural Net for Deep-Q learning Model
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))
        model.add(Dense(24, activation='relu'))
        model.add(Dense(self.action_size, activation='linear'))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    def memorize(self, state, action, reward, next_state, done):
        self.memory.append([state, action, reward, next_state, done])

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])  # returns action

    def train_model(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for [state, action, reward, next_state, done] in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma *
                          np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay
        self.memory = []

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


if __name__ == '__main__':
    start = time.time()
    ncpus = 8
    ray.init(num_cpus=8)
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('CartPole-v1')
    env.reset()
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make REINFORCE agent
    agent = DQNAgent(state_size, action_size)
    agent_id = ray.put(agent)
    runners = [Runner.remote(nsteps=1000) for i in range(ncpus)]
    sum = 0
    for i in range(10):
        memory = []
        start = time.time()
        for update in range(20000):
            results = [r.run.remote(agent_id) for r in runners]
            results = ray.get(results)
            states, rewards, actions = [], [], []
            for result in results:
                for ele in result:
                    memory.append(ele)
            agent.memory = memory
            batch_size = len(memory)
            agent.train_model(batch_size=batch_size)
            print(len(memory))
            agent_id = ray.put(agent)
            if len(memory) >= 8000:
                break
        end = time.time()
        duration = end - start
        print(duration)
        sum = sum + duration

    print("duration of producing 1000000:", sum / 10)
