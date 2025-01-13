# https://www.kaggle.com/code/yerramvarun/fine-tuning-faster-rcnn-using-pytorch
# https://www.kaggle.com/code/sandhyakrishnan02/face-mask-detection-using-pytorch#Model-Details

from pyspark.sql import SparkSession
from pyspark.sql.functions import col
spark = SparkSession.builder.config("spark.driver.memory", "16g").appName("SimpleApp").getOrCreate()
input = "/home/arda/intelWork/data/coco2017/annotations/instances_train2017.json"
df = spark.read.json(input)
df.printSchema()
df.select("images.file_name").show(5)

with open("/home/arda/intelWork/data/coco2017/small/filenames.txt") as f:
    lines = [line.rstrip() for line in f]
print(lines)

df = df.filter(col("images.file_name").isin(lines))
df.show(5, False)

# df.printSchema()
# df.select("annotations.bbox", "images.file_name").show()

# input = "/home/arda/intelWork/data/movielens/ml-1m"
# ratedf = spark.sparkContext.textFile(input + "/ratings.dat") \
#     .map(lambda x: x.split("::")[0:4]) \
#     .map(lambda x: (int(x[0]), int(x[1]), int(x[2]), int(x[3]))) \
#     .toDF(["user", "movie", "rate", "time"])
# ratedf.show()