import gym
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

import ray

EPISODES = 1000
ray.init()


@ray.remote
class Runner:
    def __init__(self, agent, nsteps):
        self.agent = agent
        self.env = gym.make('CartPole-v1')
        state = self.env.reset()
        state_size = self.env.observation_space.shape[0]
        self.initial_state = np.reshape(state, [1, state_size])
        self.nsteps = nsteps

    def run(self):
        # For n in range number of steps
        next_state = None
        state_size = self.env.observation_space.shape[0]
        states, rewards, actions = [], [], []
        done = False
        for _ in range(self.nsteps):
            if done:
                state = self.env.reset()
                self.initial_state = np.reshape(state, [1, state_size])
                continue
            current_state = self.initial_state if next_state is None else next_state
            action = self.agent.get_action(current_state)
            next_state, reward, done, info = self.env.step(action)
            next_state = np.reshape(next_state, [1, state_size])
            states.append(next_state)
            rewards.append(reward)
            actions.append(action)
        return [states, rewards, actions]


# This is Policy Gradient agent for the Cartpole
# In this example, we use REINFORCE algorithm which uses monte-carlo update rule
class REINFORCEAgent:
    def __init__(self, state_size, action_size):
        # if you want to see Cartpole learning, then change to True
        self.render = False
        self.load_model = False
        # get size of state and action
        self.state_size = state_size
        self.action_size = action_size

        # These are hyper parameters for the Policy Gradient
        self.discount_factor = 0.99
        self.learning_rate = 0.001
        self.hidden1, self.hidden2 = 24, 24

        # create model for policy network
        self.model = self.build_model()

        # lists for the states, actions and rewards
        self.states, self.actions, self.rewards = [], [], []

        if self.load_model:
            self.model.load_weights("./save_model/cartpole_reinforce.h5")

    # approximate policy using Neural Network
    # state is input and probability of each action is output of network
    def build_model(self):
        model = Sequential()
        model.add(Dense(self.hidden1, input_dim=self.state_size, activation='relu',
                        kernel_initializer='glorot_uniform'))
        model.add(Dense(self.hidden2, activation='relu', kernel_initializer='glorot_uniform'))
        model.add(
            Dense(self.action_size, activation='softmax', kernel_initializer='glorot_uniform'))
        model.summary()
        # Using categorical crossentropy as a loss is a trick to easily
        # implement the policy gradient. Categorical cross entropy is defined
        # H(p, q) = sum(p_i * log(q_i)). For the action taken, a, you set
        # p_a = advantage. q_a is the output of the policy network, which is
        # the probability of taking the action a, i.e. policy(s, a).
        # All other p_i are zero, thus we have H(p, q) = A * log(policy(s, a))
        model.compile(loss="categorical_crossentropy", optimizer=Adam(lr=self.learning_rate))
        return model

    # using the output of policy network, pick action stochastically
    def get_action(self, state):
        policy = self.model.predict(state, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]

    # In Policy Gradient, Q function is not available.
    # Instead agent uses sample returns for evaluating policy
    def discount_rewards(self, rewards):
        discounted_rewards = np.zeros_like(rewards)
        running_add = 0
        for t in reversed(range(0, len(rewards))):
            running_add = running_add * self.discount_factor + rewards[t]
            discounted_rewards[t] = running_add
        return discounted_rewards

    # save <s, a ,r> of each step
    def append_sample(self, state, action, reward):
        self.states.append(state)
        self.rewards.append(reward)
        self.actions.append(action)

    # update policy network every episode
    def train_model(self):
        episode_length = len(self.states)
        # need to be updated according to position of rewards
        discounted_rewards = self.discount_rewards(self.rewards)
        discounted_rewards -= np.mean(discounted_rewards)
        discounted_rewards /= np.std(discounted_rewards)
        update_inputs = np.zeros((episode_length, self.state_size))
        advantages = np.zeros((episode_length, self.action_size))

        for i in range(episode_length):
            update_inputs[i] = self.states[i]
            advantages[i][self.actions[i]] = discounted_rewards[i]

        self.model.fit(update_inputs, advantages, epochs=1, verbose=0)
        self.states, self.actions, self.rewards = [], [], []


if __name__ == '__main__':
    # In case of CartPole-v1, you can play until 500 time step
    env = gym.make('CartPole-v1')
    env.reset()
    # get size of state and action from environment
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # make REINFORCE agent
    agent = REINFORCEAgent(state_size, action_size)
    agent_id = ray.put(agent)
    runners = [Runner.remote(agent, nsteps=512) for i in range(8)]
    for update in range(20000):
        results = [r.run.remote() for r in runners]
        results = ray.get(results)
        states = []
        rewards = []
        actions = []
        for result in results:
            for ele in result[0]:
                states.append(ele)
            for ele in result[1]:
                rewards.append(ele)
            for ele in result[2]:
                actions.append(ele)
        agent.rewards = rewards
        agent.states = states
        agent.actions = actions
        agent.train_model()
        agent_id = ray.put(agent)
