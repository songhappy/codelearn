from pyspark.sql import SparkSession

spark = SparkSession.builder.appName("SimpleApp").getOrCreate()
sc = spark.sparkContext

print(sc.version)

rdd = sc.parallelize([('a', 7), ('a', 2), ('b', 2)])
rdd2 = sc.parallelize([('a', 2), ('c', 2), ('d', 2)])
rdd3 = sc.parallelize(range(100))
rdd4 = sc.parallelize([("a", ["x", "y", "z"])])
text_file = sc.wholeTextFiles('/Users/guoqiong/intelWork/data/house_price/all')

# retrievaling RDD information
print(rdd.getNumPartitions())
rdd.count()
rdd.countByKey()
rdd.countByValue()

# summary
rdd3.max()
rdd3.min()
rdd3.mean()
rdd3.stdev()
rdd3.stats()

# applying functions
rdd.map(lambda x: x + (x[1], x[0])).collect()
rdd.flatMap(lambda x: x + (x[1], x[0]))  # flat every record of results after map function
print(rdd4.flatMapValues(lambda x: x).collect())

# select data
rdd.filter(lambda x: "a" in x).collect()
print(rdd.sampleByKey(fractions={"a": 0.7, "b": 0.1}, withReplacement=False).collect())
print(rdd.keys().collect())

# reduce data
rdd.reduceByKey(lambda x, y: x + y).collect()
rdd.reduce(lambda a, b: a + b)
print(rdd3.groupBy(lambda x: x % 2).mapValues(list).collect())
print("*****")
print(rdd.groupByKey().mapValues(list).collect())

# aggregate data # not sure ???
sepOp = (lambda x, y: (x[0] + y, x[1] + 1))
combOp = (lambda x, y: (x[0] + y[0], x[1] + y[1]))
print(rdd3.collect())
print(rdd3.aggregate((0, 0), seqOp=sepOp, combOp=combOp))

# mathematical operations
rdd.subtract(rdd2).collect()
rdd.subtractByKey(rdd2).collect()
rdd.sortBy(lambda x: x[1]).collect()
rdd.sortByKey().collect()
