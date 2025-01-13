# refrences
# 1. https://sparkbyexamples.com/pyspark-tutorial/
from pyspark.sql.functions import col, udf, broadcast, collect_list, \
    rank, percent_rank, max as spark_max
# pass a udf_function to a method
from pyspark.sql.types import StructType, StringType, IntegerType, StructField
from pyspark.sql.window import Window

from pyspark import Row

# from src.basics.useful_utils import *

# start soark from python program
"""SimpleApp.py"""
from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
logFile = "README.md"  # Should be some file on your system
logData = spark.read.text(logFile).cache()
numAs = logData.filter(logData.value.contains('a')).count()
numBs = logData.filter(logData.value.contains('b')).count()
print("Lines with a: %i, lines with b: %i" % (numAs, numBs))


# pyspark read and write textfile
# rdd.coalesce(1).saveAsTextFile(...)
# users = sc.textFile("/tmp/movielens/ml-1m/users.dat") \
#     .map(lambda l: l.split("::")[0:4])\
#     .map(lambda l: (int(l[0]), l[1], int(l[2]), int(l[3])))\
#

# peopleDF = spark.read.json("examples/src/main/resources/people.json")

# DataFrames can be saved as Parquet files, maintaining the schema information.
# peopleDF.write.parquet("people.parquet")
#
# # Read in the Parquet file created above.
# # Parquet files are self-describing so the schema is preserved.
# # The result of loading a parquet file is also a DataFrame.
# parquetFile = spark.read.parquet("people.parquet")


def get_ml_data(spark):
    input = "/Users/guoqiong/intelWork/data/movielens/ml-1m"
    ratedf = spark.sparkContext.textFile(input + "/ratings.dat") \
        .map(lambda x: x.split("::")[0:4]) \
        .map(lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3]))) \
        .toDF(["user", "movie", "rate", "time"])
    # gender = lambda x: 1 (if x = "M") else 0
    userdf = spark.sparkContext.textFile(input + "/users.dat") \
        .map(lambda x: x.split("::")[0:4]) \
        .map(lambda x: (int(x[0]), x[1], int(x[2]), int(x[3]))) \
        .toDF(["user", "gender", "age", "income"])

    n_uid = ratedf.select("user").agg(spark_max(ratedf.user).alias("max")).rdd.map(
        lambda row: row['max']).collect()[0] + 1
    n_mid = ratedf.select("movie").agg(spark_max(ratedf.movie).alias("max")).rdd.map(
        lambda row: row['max']).collect()[0] + 1

    return [ratedf, userdf]


def transform_python_udf(self, in_col, out_col, udf_func):
    df = self.df_scaled.withColumn(out_col, udf_func(col(in_col)))
    return df


def createDFfromRdd(spark):
    data = [("Java", int(20000)), ("Python", 100000), ("Scala", int(3000))]
    columns = ["language", "users_count"]
    rdd = spark.sparkContext.parallelize(data)
    schema = StructType(
        [StructField("language", StringType()), StructField("users_count", IntegerType())])

    dfFromRDD1 = rdd.toDF(columns)
    dfFromRDD1.show()
    dfFromRDD2 = spark.createDataFrame(rdd).toDF(*columns)
    dfFromRDD2.show()
    rowrdd = rdd.map(lambda x: (x[0], int(x[1])))
    dfFromRDD3 = spark.createDataFrame(rowrdd, schema)
    dfFromRDD3.show()
    pass


def createDFfromList(spark):
    data = [("Java", int(20000)), ("Python", 100000), ("Scala", int(3000))]
    columns = ["language", "users_count"]
    rdd = spark.sparkContext.parallelize(data)
    schema = StructType(
        [StructField("language", StringType()), StructField("users_count", IntegerType())])

    dfFromData1 = spark.createDataFrame(data).toDF(*columns)
    # dfFromData1.printSchema()
    rowData = map(lambda x: Row(*x), data)
    dfFromData2 = spark.createDataFrame(rowData, columns)
    # dfFromData2.printSchema()
    mappeddata = map(lambda x: (x[0], int(x[1])), data)
    dfFromData3 = spark.createDataFrame(mappeddata, schema)
    # dfFromData3.printSchema()
    return dfFromData3


# groupby, aggregation functions and
# approx_count_distinct, avg, collect_list, collect_set, countDistinct, count,
# grouping, fist, last, kurtosis, max, min, skewness, stddev, stddev_samp, stddev_pop, sum, sumDistinct, variance, var_samp, var_pop
def agg_functions(spark):
    [ratedf, _] = get_ml_data(spark)

    df = ratedf.groupBy("user").agg(collect_list(col("movie")).alias("movie"),
                                    collect_list(col("rate")).alias("rate"))

    print(df.show(10))
    df.printSchema()


# window functions,
# row_number(), rank(), percent_rank(), dense_rank(), ntile(), cume_dist(), lag(e:column offset:int), lead(e:column, offset:int)
def window_functions(spark):
    simpleData = (("James", "Sales", 3000), \
                  ("Michael", "Sales", 4600), \
                  ("Robert", "Sales", 4100), \
                  ("Maria", "Finance", 3000), \
                  ("James", "Sales", 3000), \
                  ("Scott", "Finance", 3300), \
                  ("Jen", "Finance", 3900), \
                  ("Jeff", "Marketing", 3000), \
                  ("Kumar", "Marketing", 2000), \
                  ("Saif", "Sales", 4100) \
                  )

    columns = ["employee_name", "department", "salary"]
    df = spark.createDataFrame(data=simpleData, schema=columns)
    df.printSchema()
    df.show(truncate=False)
    windowSpec = Window.partitionBy("department").orderBy("salary")
    df.withColumn("rank", rank().over(windowSpec)) \
        .show()

    df.withColumn("percent_rank", percent_rank().over(windowSpec)) \
        .show()
    from pyspark.sql.functions import lag, lead

    df.withColumn("lag", lag("salary", 1).over(windowSpec)) \
        .show()
    df.withColumn("lead", lead("salary", 1).over(windowSpec)) \
        .show()
    from pyspark.sql.functions import col, avg, sum, min, row_number, max as spark_max
    windowSpecAgg = Window.partitionBy("department").orderBy("salary")
    df.withColumn("row", row_number().over(windowSpec)) \
        .withColumn("avg", avg(col("salary")).over(windowSpecAgg)) \
        .withColumn("sum", sum(col("salary")).over(windowSpecAgg)) \
        .withColumn("min", min(col("salary")).over(windowSpecAgg)) \
        .withColumn("max", spark_max(col("salary")).over(windowSpecAgg)) \
        .where(col("row") == 1).select("department", "avg", "sum", "min", "max") \
        .show()


# time functions
# current_date(), to_date(), date_format(), add_months(), date_add(), date_sub(), datediff(),
# months_between(), next_day(), year(), month, dayofmonth, dayofweek, dayofyear, weekofyear, from_unixtime, unix_timestamp

# all kinds of join senarios
# inner join, left, right, full
# broadcast join
# bucket join
def join_senarios(spark):
    df1 = spark.createDataFrame(
        [(1, "a", 2.0), (2, "b", 3.0), (3, "c", 3.0)],
        ("x1", "x2", "x3"))

    df2 = spark.createDataFrame(
        [(1, "f", -1.0), (2, "b", 0.0)], ("x1", "x2", "x3"))

    df = df1.join(broadcast(df2), (df1.x1 == df2.x1) & (df1.x2 == df2.x2))
    df.show()
    df = df1.join(df2, ["x1", "x2"])
    df.show()


def join_skew(spark):
    # if hive，setup, hive.map.aggr = true; hive.groupb.skewindata = true
    [ratedf, userdf] = get_ml_data(spark)
    distribution = ratedf.select("user").groupBy("user").count().orderBy(
        col("count").desc()).persist()  # should sample(0.01)
    total_number = distribution.select("count").groupBy().sum().collect()[0][0]
    distribution.show(10)
    print("*************")
    print(total_number)
    # regular join
    print("regular join count")
    joined1 = ratedf.join(userdf, ["user"])
    print(joined1.count())

    topusers = distribution.filter("count > " + str(total_number) + "* 0.001").select(
        "user").collect()
    topusers1 = distribution.orderBy(col("count").desc()).limit(10).collect()
    print(topusers1)
    topids = list(map(lambda x: x[0], topusers1))
    import random
    gen_key = lambda userid: str(userid) + "_" + str(
        random.randint(1, 10)) if userid in topids else str(userid)
    #  gen_key_udf = udf(gen_key, StringType()) if too many need structfiled and name
    gen_key_udf = udf(gen_key)
    ratedf = ratedf.withColumn("joinkey", gen_key_udf(col("user"))).withColumnRenamed("user",
                                                                                      "userr")
    userdf = userdf.withColumn("joinkey", gen_key_udf(col("user")))
    joined = ratedf.join(userdf,
                         ["joinkey"])  # .filter(col("user").isin(topids))  # filter isin list
    joined.show(2)
    print("skew join count")
    print(joined.count())


def join_bucket(spark):  # too big data
    [ratedf, userdf] = get_ml_data(spark)
    ratedf.show(10)
    ratedf.write.mode("overwrite").bucketBy(4, "user").saveAsTable("rating_table")
    userdf.write.mode("overwrite").bucketBy(4, "user").saveAsTable("user_table")
    rate_tbl = spark.table("rating_table")
    user_tbl = spark.table("user_table")
    joined = rate_tbl.join(user_tbl, ["user"])
    joined.show(10)


# udfs
# one to one, one to multiple,
# one to multiple, multiple to mutiple

def udf_functions(spark):
    [ratedf, userdf] = get_ml_data(spark)
    item_size = ratedf.select("movie").distinct().count()
    hist_df = gen_his_seq(ratedf, user_col='user', cols=['movie', 'rate'], sort_col='time',
                          min_len=1, max_len=10)
    with_negative = add_negtive_samples(df=hist_df, item_size=item_size, item_col="movie",
                                        label_col="label", neg_num=2)
    padded = pad(df=with_negative, padding_cols=["movie_history", "rate_history"])
    padded.show(10)

    # def gen_reindex_mapping(self, columns=[], freq_limit=10):
    #     """
    #     Generate a mapping from old index to new one based on popularity count on descending order
    #      :param columns: str or a list of str
    #      :param freq_limit: int, dict or None. Indices with a count below freq_limit
    #            will be omitted. Can be represented as either an integer or dict.
    #            For instance, 15, {'col_4': 10, 'col_5': 2} etc. Default is 10,
    #
    #     :return: a dictionary of list of dictionaries, a mapping from old index to new index
    #             new index starts from 1, save 0 for default
    #      """
    #     str_to_list(columns, "columns")
    #     if isinstance(freq_limit, int):
    #         freq_limit = {col: freq_limit for col in columns}
    #     assert isinstance(freq_limit, dict), \
    #         "freq_limit should be int or dict, but get a " + type(freq_limit)
    #     index_dicts = []
    #     for c in columns:
    #         c_count = self.select(c).group_by(c, agg={c: "count"}).rename(
    #             {"count(" + c + ")": "count"})
    #         c_count = c_count.filter(pyspark_col("count") >= freq_limit[c]) \
    #             .order_by("count", ascending=False)
    #         c_count_pd = c_count.to_pandas()
    #         c_count_pd.reindex()
    #         c_count_pd[c + "_new"] = c_count_pd.index + 1
    #         index_dict = dict(zip(c_count_pd[c], c_count_pd[c + "_new"]))
    #         index_dicts.append(index_dict)
    #     if isinstance(columns, str):
    #         index_dicts = index_dicts[0]
    #
    #     return index_dicts
    #
    # def add_value_features(self, columns, tbls, key=None, value=None, reindex_only=False):
    #     """
    #      Add features based on columns and another key value table,
    #      for each col in columns, it adds a value_col using key-value pairs from tbls, replace old
    #      index with new one from key-value tbls if reindex_only is True.
    #
    #      :param columns: a list of str
    #      :param tbls: Table with only two columns [key, value]
    #      :param key: str, name of key column in tbl, None while reindex_only
    #      :param value: str, name of value column in tbl, None while reindex_only
    #      :param reindex_only: boolean, if reindex only or add values
    #
    #      :return: FeatureTable, dict
    #      """
    #     if isinstance(columns, str):
    #         columns = [columns]
    #     assert isinstance(columns, list), \
    #         "columns should be str or a list of str but get " + type(columns)
    #     if isinstance(tbls, Table):
    #         tbls = [tbls]
    #     assert isinstance(tbls, list), \
    #         "tbls should be Table or a list of Tables  get " + type(tbls)
    #
    #     if reindex_only:
    #         assert len(columns) == len(tbls), \
    #             "each column of columns should have one corresponding index table while reindex"
    #     else:
    #         assert len(tbls) == 1, \
    #             "all columns should share one index table while add value features"
    #
    #     def lookup(items, keyvalue_map):
    #         getvalue = lambda item: keyvalue_map.get(item, 0)
    #         if isinstance(items, int) or items is None:
    #             values = getvalue(items)
    #         elif isinstance(items, list) and isinstance(items[0], int):
    #             values = [getvalue(item) for item in items]
    #         elif isinstance(items, list) and isinstance(items[0], list) and isinstance(items[0][0],
    #                                                                                    int):
    #             values = []
    #             for line in items:
    #                 line_values = [getvalue(item) for item in line]
    #                 values.append(line_values)
    #         else:
    #             raise ValueError('only int, list[int], and list[list[int]] are supported.')
    #         return values
    #
    #     value_dims = {}
    #     df = self.df
    #     spark = OrcaContext.get_spark_session()
    #     for i, c in enumerate(columns):
    #         (index_tb, new_c) = (tbls[i], c) if reindex_only else (tbls[0], c.replace(key, value))
    #         key_value = dict(index_tb.df.rdd.map(lambda row: (row[0], row[1])).collect())
    #         key_value_bc = spark.sparkContext.broadcast(key_value)
    #         col_type = df.schema[c].dataType
    #         lookup_udf = udf(lambda x: lookup(x, key_value_bc.value), col_type)
    #         df = df.withColumn(new_c, lookup_udf(pyspark_col(c)))
    #         value_dims[c] = max(key_value.values()) + 1
    #
    #     return FeatureTable(df), value_dims
    #
    # def add_value_features(self, key_cols, tbl, key, value):
    #     """
    #      Add features based on key_cols and another key value table,
    #      for each col in key_cols, it adds a value_col using key-value pairs from tbl
    #
    #      :param key_cols: a list of str
    #      :param tbl: Table with only two columns [key, value]
    #      :param key: str, name of key column in tbl
    #      :param value: str, name of value column in tbl
    #
    #      :return: FeatureTable
    #      """
    #     spark = OrcaContext.get_spark_session()
    #     keyvalue_bc = spark.sparkContext.broadcast(dict(tbl.df.distinct().rdd.map(
    #         lambda row: (row[0], row[1])).collect()))
    #
    #     keyvalue_map = keyvalue_bc.value
    #
    #     def gen_values(items):
    #         getvalue = lambda item: keyvalue_map.get(item)
    #         if isinstance(items, int):
    #             values = getvalue(items)
    #         elif isinstance(items, list) and isinstance(items[0], int):
    #             values = [getvalue(item) for item in items]
    #         elif isinstance(items, list) and isinstance(items[0], list) and isinstance(items[0][0],
    #                                                                                    int):
    #             values = []
    #             for line in items:
    #                 line_cats = [getvalue(item) for item in line]
    #                 values.append(line_cats)
    #         else:
    #             raise ValueError('only int, list[int], and list[list[int]] are supported.')
    #         return values
    #     df = self.df
    #     for c in key_cols:
    #         col_type = df.schema[c].dataType
    #         cat_udf = udf(gen_values, col_type)
    #         df = df.withColumn(c.replace(key, value), cat_udf(pyspark_col(c)))
    #     return FeatureTable(df)
    #
    # def reindex(self, columns=[], index_dicts=[]):
    #     """
    #     Replace the value using index_dicts for each col in columns, set 0 for default
    #
    #     :param columns: str of a list of str
    #     :param index_dicts: dict or list of dicts from int to int
    #
    #     :return: FeatureTable and dimentionss of columns
    #      """
    #     if isinstance(columns, str):
    #         columns = [columns]
    #     assert isinstance(columns, list), \
    #         "columns should be str or a list of str, but get a " + type(columns)
    #     if isinstance(index_dicts, dict):
    #         index_dicts = [index_dicts]
    #     assert isinstance(index_dicts, list), \
    #         "index_dicts should be dict or a list of dict, but get a " + type(index_dicts)
    #     assert len(columns) == len(index_dicts), \
    #         "each column of columns should have one corresponding index_dict"
    #
    #     tbl = FeatureTable(self.df)
    #     for i, c in enumerate(columns):
    #         index_dict = index_dicts[i]
    #         spark = OrcaContext.get_spark_session()
    #         index_dict_bc = spark.sparkContext.broadcast(index_dict)
    #         index_lookup = lambda x: index_dict_bc.value.get(x, 0)
    #         tbl = tbl.apply(c, c, index_lookup, "int")
    #     return tbl
    #
    # def gen_reindex_mapping(self, columns=[], freq_limit=10):
    #     """
    #     Generate a mapping from old index to new one based on popularity count on descending order
    #      :param columns: str or a list of str
    #      :param freq_limit: int, dict or None. Indices with a count below freq_limit
    #            will be omitted. Can be represented as either an integer or dict.
    #            For instance, 15, {'col_4': 10, 'col_5': 2} etc. Default is 10,
    #
    #     :return: a dictionary of list of dictionaries, a mapping from old index to new index
    #             new index starts from 1, save 0 for default
    #      """
    #     if isinstance(columns, str):
    #         columns = [columns]
    #     assert isinstance(columns, list), \
    #         "columns should be str or a list of str, but get a " + type(columns)
    #     if isinstance(freq_limit, int):
    #         freq_limit = {col: freq_limit for col in columns}
    #     assert isinstance(freq_limit, dict),\
    #         "freq_limit should be int or dict, but get a " + type(freq_limit)
    #     index_dicts = []
    #     for c in columns:
    #         c_count = self.select(c).group_by(c, agg={c: "count"}).rename(
    #             {"count(" + c + ")": "count"})
    #         c_count = c_count.filter(pyspark_col("count") >= freq_limit[c])\
    #             .order_by("count", ascending=False)
    #         c_count_pd = c_count.to_pandas()
    #         c_count_pd.reindex()
    #         c_count_pd[c + "_new"] = c_count_pd.index + 1
    #         index_dict = dict(zip(c_count_pd[c], c_count_pd[c + "_new"]))
    #         index_dicts.append(index_dict)
    #     if isinstance(columns, str):
    #         index_dicts = index_dicts[0]
    #
    #     return index_dicts
    #
    # def test_add_value_features_reindex(self):
    #     file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
    #     feature_tbl = FeatureTable.read_parquet(file_path)
    #     string_idx_list = feature_tbl.gen_string_idx(["col_4", "col_5"],
    #                                                  freq_limit={"col_4": 1, "col_5": 1},
    #                                                  order_by_freq=False)
    #     tbl_with_index = feature_tbl.encode_string(["col_4", "col_5"], string_idx_list)
    #     tbl_with_index.show(100)
    #     index_dicts = tbl_with_index.gen_reindex_mapping(["col_4", "col_5"], 2)
    #     tbls = []
    #     for d in index_dicts:
    #         dict_tbl = StringIndex.from_dict(d, "tmp").cast("tmp", "int")
    #         tbls.append(dict_tbl)
    #     reidxed, _ = tbl_with_index.add_value_features(["col_4", "col_5"], tbls, reindex_only=True)
    #     assert (reidxed.filter(col("col_4") == 0).size() == 3)
    #     assert (reidxed.filter(col("col_4") == 1).size() == 2)
    #     assert (reidxed.filter(col("col_5") == 0).size() == 2)
    #     assert (reidxed.filter(col("col_5") == 1).size() == 3)
    #
    # def test_reindex(self):
    #     file_path = os.path.join(self.resource_path, "friesian/feature/parquet/data1.parquet")
    #     feature_tbl = FeatureTable.read_parquet(file_path)
    #     string_idx_list = feature_tbl.gen_string_idx(["col_4", "col_5"],
    #                                                  freq_limit={"col_4": 1, "col_5": 1},
    #                                                  order_by_freq=False)
    #     tbl_with_index = feature_tbl.encode_string(["col_4", "col_5"], string_idx_list)
    #     index_dicts = tbl_with_index.gen_reindex_mapping(["col_4", "col_5"], 2)
    #     reindexed = tbl_with_index.reindex(["col_4", "col_5"], index_dicts)
    #     assert(reindexed.filter(col("col_4") == 0).size() == 3)
    #     assert(reindexed.filter(col("col_4") == 1).size() == 2)
    #     assert(reindexed.filter(col("col_5") == 0).size() == 2)
    #     assert(reindexed.filter(col("col_5") == 1).size() == 3)


if __name__ == '__main__':
    spark = SparkSession.builder.enableHiveSupport().appName('SparkByExamples.com').getOrCreate()
    conf = spark.sparkContext.getConf()
    print("8888")
    print(conf)
    # create dataframe
    # createDFfromRdd(spark)
    # createDFfromList(spark)

    # filter dataframe
    # df = createDFfromList(spark)
    # df.orderBy("users_count").filter(df["users_count"] > 100).show()
    # df.filter(df.users_count > 100)
    # df.filter(col("users_count") > 100)
    # df.filter("users_count = 3000")
    # df.filter("users_count == 3000")

    # window_functions(spark)
    # join_senarios(spark)
    # join_bucket(spark)
    join_skew(spark)
    # agg_functions(spark)

    # all kinds of udfs
    # udf_functions(spark)
