import sys

import os
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, IntegerType, ArrayType, LongType
from bigdl.dllib.feature.dataset.news20 import get_glove_w2v
import random
from pyspark.sql.functions import *
from pyspark.sql.functions import udf, array, broadcast, explode, struct, collect_list, \
    rank, percent_rank
from pyspark.ml.feature import VectorAssembler
from pyspark.sql import functions as F
from pyspark.sql import types as T
from bigdl.dllib.nnframes.nn_classifier import *
from pyspark.sql.functions import col
import argparse


def index_strings(names):
    names_dict = {}
    for i, name in enumerate(names):
        if name not in names_dict:
            names_dict[name] = i
    return names_dict


def hash_bucket(content, bucket_size=1000, start=0):
    return (hash(str(content)) % bucket_size + bucket_size) % bucket_size + start


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


def encode_ml_users(user_file="./data/movielens/ml-1m/users.dat"):
    # feature length = 30
    gender_list = ['F', 'M']
    age_list = ["1", "18", "25", "35", "45", "50", "56"]
    income_list = [str(i) for i in range(21)]
    user_dict = {}
    with open(user_file) as user_f:
        for line in user_f:
            values = line.split("::")
            gender_one_hot = one_hot_encode(values[0], gender_list)
            age_one_hot = one_hot_encode(values[1], age_list)
            income_one_hot = one_hot_encode(values[2], income_list)
            embed = np.concatenate([gender_one_hot, age_one_hot, income_one_hot])
            user_dict[int(values[0])] = np.array(embed)
    return user_dict


def encode_ml_movie(movie_file="/Users/guoqiong/data/movielens/ml-1m/movies.dat", embed_dim=50):
    glove_dict = get_glove_w2v(source_dir="/Users/guoqiong/data/glove.6B", dim=embed_dim)
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


cosine = lambda vA, vB: np.dot(vA, vB) / (np.sqrt(np.dot(vA, vA)) * np.sqrt(np.dot(vB, vB)))

gender_udf = udf(lambda gender: categorical_from_vocab_list(gender, ["F", "M"], start=1))
bucket_cross_udf = udf(
    lambda feature1, feature2: hash_bucket(str(feature1) + "_" + str(feature2), bucket_size=100))
genres_list = ["Crime", "Romance", "Thriller", "Adventure", "Drama", "Children's",
               "War", "Documentary", "Fantasy", "Mystery", "Musical", "Animation", "Film-Noir",
               "Horror",
               "Western", "Comedy", "Action", "Sci-Fi"]
genres_udf = udf(lambda genres: categorical_from_vocab_list(genres, genres_list, start=1))


# udf one to multiple, schema is a structType of tuple
def add_negtive_samples(df, item_size, item_col="item", label_col="label", neg_num=1):
    def gen_neg(item):
        result = []
        for i in range(neg_num):
            while True:
                neg_item = random.randint(0, item_size - 1)
                if neg_item == item:
                    continue
                else:
                    result.append((neg_item, 0))
                    break
        result.append((item, 1))
        return result

    structure = StructType().add(item_col, IntegerType()).add(label_col,
                                                              IntegerType())  # StructTyoe of tuples is used to handle multiple columns
    neg_udf = udf(gen_neg, ArrayType(structure))
    df = df.withColumn("item_label", neg_udf(col(item_col))) \
        .withColumn("item_label", explode(col("item_label"))).drop(item_col)
    df = df.withColumn(item_col, col("item_label." + item_col)) \
        .withColumn(label_col, col("item_label." + label_col)).drop("item_label")
    return df


# udf multiple to multiple and select expression
def gen_his_seq(df, user_col='user', cols=['item', 'category'], sort_col='time', min_len=1,
                max_len=100):
    def gen_his(row_list):
        if sort_col:
            row_list.sort(key=lambda row: row[sort_col])  # no getAs, just row[col_name], or row[i]

        def gen_his_one_row(rows, i):
            histories = [[row[col] for row in rows[:i]] for col in cols]
            return (*[rows[i][col] for col in cols], *histories)

        if len(row_list) <= min_len:
            return None
        if len(row_list) > max_len:
            row_list = row_list[-max_len:]
        result = [gen_his_one_row(row_list, i) for i in range(min_len, len(row_list))]
        return result

    structure = StructType()  # for tuples
    for c in cols:
        structure = structure.add(c, LongType())
    for c in cols:
        structure = structure.add(c + "_history", ArrayType(IntegerType()))
    schema = ArrayType(structure)

    gen_his_udf = udf(lambda x: gen_his(x), schema)
    df = df.groupBy(user_col) \
        .agg(
        collect_list(struct(*[col(name) for name in (cols + [sort_col])])).alias("asin_collect"))
    df.show(10)
    df.printSchema()
    df = df.withColumn("item_cat_history", gen_his_udf(col("asin_collect"))).dropna(
        subset=['item_cat_history']) \
        .withColumn("item_cat_history", explode(col("item_cat_history"))) \
        .drop("asin_collect")
    df.show(10)
    df.printSchema()
    selectexp1 = ["item_cat_history." + c + " as " + c for c in cols]
    selectexp2 = ["item_cat_history." + c + "_history" + " as " + c + "_history" for c in cols]
    select_exp = (selectexp1 + selectexp2 + [user_col])
    print(select_exp)
    df = df.selectExpr(*select_exp)
    # for c in cols:
    #     df = df.withColumn(c, col("item_cat_history." + c)) \
    #             .withColumn(c + "_history", col("item_cat_history." + c + '_history'))
    # df = df.drop("item_cat_history")
    df.show(10)
    df.printSchema()
    return df


# udf one to one and select expression
def pad(df, padding_cols, seq_len=5):
    spark = SparkSession.builder.getOrCreate()

    def pad(seq):
        length = len(seq)
        if len(seq) < seq_len:
            return seq + [0] * (seq_len - length)
        else:
            return seq[:seq_len]

    df.createOrReplaceTempView("tmp")
    spark.udf.register("postpad", udf(pad))
    select_statement = ",".join(["postpad(" + c + ") as " + c for c in padding_cols])
    df = spark.sql("select " + select_statement + " from tmp")
    df.printSchema()
    df.show(10)

    # for c in padding_cols:
    #     col_type = df.schema[c].dataType
    #     pad_udf = udf(pad, col_type)
    #     df = df.withColumn(c, pad_udf(col(c)))
    return df
