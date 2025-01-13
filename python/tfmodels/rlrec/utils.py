
# create dataframe pyspark
#  https://sparkbyexamples.com/spark/different-ways-to-create-a-spark-dataframe/\
import sys
import numpy as np
import os
from zoo.examples.textclassification.news20 import get_glove

def categorical_from_vocab_list(sth, vocab_list, default=0, start=1):
    if sth in vocab_list:
        return vocab_list.index(sth) + start
    else:
        return default + start


def custom_text_to_sequence(input_text,
                            filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n',
                            split=" "):
    text = input_text.replace('\'s', '').lower()
    translate_dict = {c: split for c in filters}
    translate_map = str.maketrans(translate_dict)
    text = text.translate(translate_map)
    seq = text.split(split)
    return [w for w in seq if w]


def one_hot_encode(item, item_list):
    idx = categorical_from_vocab_list(item, item_list)
    encoded = [int(0) for _ in range(len(item_list))]
    encoded[idx] = 1
    return np.array(encoded)

def encode_ml_users(user_file ="./data/movielens/ml-1m/users.dat"):
    # feature length = 30
    gender_list = ['F','M']
    age_list = ["1", "18", "25", "35", "45", "50", "56"]
    income_list = [str(i) for i in range(21)]
    user_dict = {}
    with open(user_file) as user_f:
        for line in user_f:
            values = line.split("::")
            gender_one_hot = one_hot_encode(values[0], gender_list)
            age_one_hot = one_hot_encode(values[1], age_list)
            income_one_hot = one_hot_encode(values[2],income_list)
            embed = np.concatenate([gender_one_hot, age_one_hot, income_one_hot])
            user_dict[int(values[0])] = np.array(embed)
    return user_dict


def encode_ml_movie(movie_file="./data/movielens/ml-1m/movies.dat", embed_dim = 50):
    glove_dict = get_glove(base_dir="./data/glove.6B", dim=embed_dim)
    movie_dict = {}
    with open(movie_file, encoding='latin-1') as movie_f:
        for line in movie_f:
            values = line.split("::")
            key = values[0]
            words = custom_text_to_sequence(values[2])
            movie_embed = [0.0 for i in range(embed_dim)]
            count = 0
            for word in words:
                if word in glove_dict.keys():
                    word_embed = glove_dict[word]
                    count = count + 1
                    for i in range(0, embed_dim):
                        movie_embed[i] = movie_embed[i] + word_embed[i]
            if count == 0:
                sys.exit("please check the movie title")
            movie_embed = [item / count for item in movie_embed]
            movie_dict[int(key)] = np.array(movie_embed)
    return movie_dict

cosine = lambda vA,vB:  np.dot(vA, vB) / (np.sqrt(np.dot(vA,vA)) * np.sqrt(np.dot(vB,vB)))