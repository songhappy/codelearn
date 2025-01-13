import gym
import sys
import numpy as np
from zoo.examples.textclassification.news20 import get_glove
from bigdl.dataset import movielens
import pandas

from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2

# glove_data = get_glove(base_dir="./data/glove.6B", dim=50)
# print(glove_data['hello'])
# print(len(glove_data['hello']))
# print(type(glove_data["hello"]))

# for i in range(1,10):
#     print(i)
#     print({i:np.array([1,2,1])})
#
# env = gym.make('CartPole-v1')
# print(env.action_space)
# print(env.observation_space)
# Optional: PPO2 requires a vectorized environment to run
# the env is now wrapped automatically when passing it to the constructor
# env = DummyVecEnv([lambda: env])

# model = PPO2(MlpPolicy, env, verbose=1)
# model.learn(total_timesteps=100)

# obs = env.reset()
# for i in range(1):
#     action, _states = model.predict(obs)
#     obs, rewards, dones, info = env.step(action)
#     print(obs)
#     print(type(obs))
#     print(dones)
#     print(type(dones))
#     print(rewards)
#     print(type(rewards))
#     print(info)
#     print(type(info))
#     #env.render()
#
# env.close()

# x = np.concatenate([np.array([1,2,4]), np.array([2,4,6])])
# print(x)

from zoo.feature.text import TextSet
from zoo.common.nncontext import init_nncontext
from zoo.pipeline.api.keras.layers import WordEmbedding
import sys
import zoo
from zoo.common.nncontext import *
from zoo.pipeline.api.keras.models import *
from zoo.pipeline.api.keras.layers import *
print(zoo.__version__)

# sc = init_nncontext("movie recommendation")
# embedding_file = "/Users/guoqiong/intelWork/git/learn/src/src/data/glove.6B/glove.6B/glove.6B." + str(50) + "d.txt"
# #
# # text_set = TextSet.read(path="/Users/guoqiong/intelWork/git/learn/src/src/data/movielens/ml-1m/movie/m1" ).to_distributed(sc, int(4))
# # word_idx = text_set.get_word_index()
# # print(word_idx)
# #
# # transformed = text_set.tokenize()\
# #     .normalize()\
# #     .word2idx(remove_topN=1, max_words_num=int(20))\
# #     .shape_sequence(len=int(20)).generate_sample()
#
# embedding_dict = {}
# with open(embedding_file) as f:
#     for line in f:
#        values = line.split(" ")
#        key = values[0]
#        embedding_dict[values[0]] = [float(item) for item in values[1:]]
#
#        # value =[]
#        # for i in range(1,50):
#        #     value.append(float(values[i]))
#        # embedding_dict[key] = value
# # for key,value in embedding_dict.items():
# #     print(key,'=',value)
#
# movie_dict={}
# movie_file= "/src/data/movielens/ml-1m/movies.dat"
# # remove punctuation from each word
# import string
#
# with open(movie_file, encoding='latin-1') as movie_f:
#     for line in movie_f:
#         values = line.split("::")
#         key = values[0]
#         table = str.maketrans('', '', string.punctuation)
#         words = values[2].replace('\'s','').replace('-','|').strip().lower().split("|")
#         words = [w.translate(table) for w in words]
#         vectors = [0.0 for i in range(50)]
#         count = 0
#         for word in words:
#             if word in embedding_dict.keys():
#                 embedding_vector = embedding_dict[word]
#                 count = count + 1
#                 for i in range(0,50):
#                     vectors[i] = vectors[i] + embedding_vector[i]
#         if count == 0:
#             sys.exit("please check the movie title")
#         vectors = [item/count for item in vectors]
#         movie_dict[int(key)] = np.array(vectors)
#     movie_f.close()
#
# user_file = "/Users/guoqiong/intelWork/git/learn/src/src/data/movielens/ml-1m/users.dat"
# user_dict ={}
# with open(user_file) as user_f:
#     for line in user_f:
#         values = line.split("::")
#         vectors = []
#         vectors.append(0.0) if values[1] == "M" else vectors.append(1.0)
#         vectors.append(float(values[2]))
#         vectors.append(float(values[3]))
#         user_dict[int(values[0])]=vectors
#     user_f.close()
#
# import ray
# print(ray.__version__)

movielens_data = movielens.get_id_ratings("./data/movielens/")
df = pandas.DataFrame(movielens_data, columns = ['uid','mid', 'rate'])
# data = df.to_dict()
u_m = df.groupby('uid')['mid'].apply(list).reset_index(name='mids')
print(u_m.head(5))
print(df.head(5))

user_movie = {}
for index, row in u_m.iterrows():
    user_movie[row['uid']] = row['mids']
print(user_movie.keys())
values = user_movie[1]
import random
print(random.choice(values))



