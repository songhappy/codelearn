##
# from https://colab.research.google.com/drive/1yFphU6PW9Uo6lmDly_ud9a6c4RCYlwdX#scrollTo=Zn0n2S-FWZih

from transformers import BertModel
import time
from bigdl.orca import init_orca_context, OrcaContext
from bigdl.friesian.feature import FeatureTable
from pyspark.sql.functions import col as pyspark_col, udf, broadcast, \
    row_number, desc
from pyspark.sql.types import ArrayType, IntegerType, \
    DoubleType

from pyspark.ml.feature import PCA

import pandas as pd

data_dir = "/Users/guoqiong/intelWork/data/movielens"

sc = init_orca_context("local", cores=8, memory="8g", init_ray_on_spark=True)
# Load pre-trained model (weights)
model = BertModel.from_pretrained('bert-base-uncased',
                                  output_hidden_states=True,
                                  # Whether the model returns all hidden-states.
                                  )

# Put the model in "evaluation" mode, meaning feed-forward operation.
# model.eval()
# print(model)
# import sys
# sys.exit()

# Load pre-trained model tokenizer (vocabulary)
# tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
#
# texts = ["Here is the sentence I want embeddings for.", "After stealing money from the bank vault, the bank robber was seen " \
#        "fishing on the Mississippi river bank."]
# marked_texts = ["[CLS] " + text + " [SEP]" for text in texts]
# rdd = sc.parallelize(marked_texts)
# print(rdd.take(1))
# #
# indexed_rdd = rdd.map(lambda x: tokenizer_br.value.tokenize(x))\
#                  .map(lambda x:tokenizer_br.value.convert_tokens_to_ids(x))


item_df = pd.read_csv(data_dir + "/ml-1m/movies.dat", encoding="ISO-8859-1",
                      delimiter="::", names=["item", "title", "genres"])
item_tbl = FeatureTable.from_pandas(item_df).cast("item", "int")
item_tbl.cache()


def string_emd(df, cols, reduce_dim=None, replace=False):
    # bert_model = 'distilbert-base-uncased'
    # from transformers import DistilBertTokenizer as BertTokennizer
    # from transformers import DistilBertModel as BertModel
    bert_model = 'bert-base-uncased'
    from transformers import BertTokenizer as BertTokenizer
    from transformers import BertModel as BertModel

    import torch
    model = BertModel.from_pretrained(bert_model,
                                      output_hidden_states=True)
    model.eval()
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    sc = OrcaContext.get_spark_context()
    tokenizer_br = sc.broadcast(tokenizer)
    model_br = sc.broadcast(model)

    def str2id(string, tokenizer):
        marked = "[CLS] " + string + " [SEP]"
        tokenized = tokenizer.tokenize(marked)
        ids = tokenizer.convert_tokens_to_ids(tokenized)
        return ids

    def ids2emb(ids, model):
        segments_ids = [1] * len(ids)
        tokens_tensor = torch.tensor([ids])
        segments_tensors = torch.tensor([segments_ids])

        with torch.no_grad():
            outputs = model(tokens_tensor, segments_tensors)
            print(len(outputs))
            hidden_states = outputs[2]

        token_vecs = hidden_states[-1][0]
        # token_vecs = outputs.last_hidden_state
        embedding = torch.mean(token_vecs, dim=0)
        out = embedding.detach().numpy().tolist()
        return out

    str2id_udf = udf(lambda x: str2id(x, tokenizer_br.value), ArrayType(IntegerType()))
    ids2embd_udf = udf(lambda x: ids2emb(x, model_br.value), ArrayType(DoubleType()))

    for c in cols:
        df = df.withColumn(c + "_ids", str2id_udf(pyspark_col(c))) \
            .withColumn(c + "_embds", ids2embd_udf(pyspark_col(c + "_ids"))) \
            .drop(c + "_ids")

    if reduce_dim:
        from pyspark.ml.feature import PCA
        from pyspark.ml.linalg import Vectors, VectorUDT
        tolist = udf(lambda x: x.toArray().tolist(), ArrayType(DoubleType()))
        tovec = udf(lambda x: Vectors.dense(x), VectorUDT())
        for c in cols:
            df = df.withColumn(c + "_embds", tovec(c + "_embds"))
            pca = PCA(k=reduce_dim, inputCol=c + "_embds", outputCol=c + "_pcaFeatures")
            pca_model = pca.fit(df)
            result = pca_model.transform(df)
            df = result.drop(c + "_embds") \
                .withColumnRenamed(c + "_pcaFeatures", c + "_embds") \
                .withColumn(c + "_embds", tolist(c + "_embds"))

    if replace:
        df = df.drop(c).withColumnRenamed(c + "_embds", c)
    return df


def fromsentence(df, cols, bert_model='distilbert-base-uncased', reduce_dim=None, replace=False):
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(bert_model)
    model_br = sc.broadcast(model)
    sentence2embd_udf = udf(lambda x: model_br.value.encode(x).tolist(), ArrayType(DoubleType()))

    for c in cols:
        df = df.withColumn(c + "_embds", sentence2embd_udf(pyspark_col(c)))

    if reduce_dim:
        from pyspark.ml.feature import PCA
        from pyspark.ml.linalg import Vectors, VectorUDT
        tolist = udf(lambda x: x.toArray().tolist(), ArrayType(DoubleType()))
        tovec = udf(lambda x: Vectors.dense(x), VectorUDT())
        for c in cols:
            df = df.withColumn(c + "_embds", tovec(c + "_embds"))
            pca = PCA(k=reduce_dim, inputCol=c + "_embds", outputCol=c + "_pcaFeatures")
            pca_model = pca.fit(df)
            result = pca_model.transform(df)
            df = result.drop(c + "_embds") \
                .withColumnRenamed(c + "_pcaFeatures", c + "_embds") \
                .withColumn(c + "_embds", tolist(c + "_embds"))

    if replace:
        df = df.drop(c).withColumnRenamed(c + "_embds", c)
    return df


item_tbl.show(10, False)

time1 = time.time()
for i in range(10):
    df1 = string_emd(item_tbl.df, ["genres"])
    print(df1.count())
time2 = time.time()

for i in range(10):
    df = fromsentence(item_tbl.df, ["genres"])
    print(df.count())
time3 = time.time()

# df1.show(2, False)
# df1.printSchema()
df.show(2, False)
df.printSchema()
print(f"string_embd preprocessing time: {(time2 - time1):.2f}s")
print(f"sentence trainsformer preprocessing time: {(time3 - time2):.2f}s")
