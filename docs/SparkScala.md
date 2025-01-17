# Spark Scala CheatSheet
This puts my utils of spark together, especially stats related
https://spark.apache.org/docs/latest/rdd-programming-guide.html
http://twitter.github.io/effectivescala/
http://www.tutorialspoint.com/scala/index.htm
https://twitter.github.io/scala_school/
https://sparkbyexamples.com/spark/different-ways-to-create-a-spark-dataframe/

content
1. create dataframe 
2. functions, udfs and partial functions
3. join scenarios 
4. tune spark jobs
5. streaming job

## Use spark
download a version, from htFtps://spark.apache.org/downloads.html, cd that home path
```
spark-shell
```

## basic rdd apis(spark core) and dataframe apis
http://spark.apache.org/docs/latest/api/scala/org/apache/spark/rdd/RDD.html
https://spark.apache.org/docs/latest/api/scala/org/apache/spark/sql/Dataset.html

```
val rdd = sc.textFile(filepath)
rdd.map(x=> x)
rdd.filter(x=> x< 0.2)
val totalCount = rdd.reduce((a, b) => a + b)
dictinct(), getNumPartitions(), getStorageLeverl
groupBy(), reduceByKey()
mapPartitions() 
repartition(10)

val df = sqlContext.read.json(filepath)
df.

```


### create Dataframe
easiest way to create dataframe is using case class for schema
```
    val sqlContext2 = new SQLContext(sc)
    import sqlContext2.implicits._
    val df2: DataFrame = sc.parallelize(List((1.0, -1.0, 2.0), (2.0, 0.0, 0.0), (0.0, 1.0, -1.0)))
      .toDF("c1", "c2", "c3")

    val dataFrame = sqlContext.createDataFrame(Seq(
      (0, Vectors.dense(1.0, 0.1, -8.0)),
      (1, Vectors.dense(2.0, 1.0, -4.0)),
      (2, Vectors.dense(4.0, 10.0, 8.0))
    )).toDF("id", "features")
```

```
    case class Rec(id: String, val1: String, val2: String, val3: String) // should be defined outside of the function, parallel to class
    // issue: Task not serializable org.apache.spark.SparkException: Task not serializable

    val sqlContext = new org.apache.spark.sql.SQLContext(sc)
    import sqlContext.implicits._
    val data: RDD[String] = sc.parallelize(Seq("a,1,2,3", "b,1,2,3", "c,2,2,3"))
    data.take(10).foreach(println)
    val df = data.map(x => {
      val lines = x.split(",")
      Rec(lines(0), lines(1), lines(2), lines(3))
    }).toDF()
    df.show()
  }
```

### partial functions and UDFs 
### general usage
#### 1. Partially applied functions and udf, multiple to 1 column

```
// prefered udf and lumbda function, anominous functions, do not need define all the signatures at beginning
val bucketUdf = udf((col:String*, bucketSize) => Math.abs(col.reduce(_ + "_" + _).hashCode()) % bucketSize + 0)

val bucketUdf =((col:String*, bucketSize) => Math.abs(col.reduce(_ + "_" + _).hashCode()) % bucketSize + 0)
sqlContext.udf.register("bucketudf", bucketUdf)
val result = sqlContext.sql("select bucketUdf('age', 100) from ratings)

private val buckeSize = 500    
val bucketUdf = udf((col:String*) => Math.abs(col.reduce(_ + "_" + _).hashCode()) % bucketSize + 0)

//Partially applied function is good for giving parameters partially.  
def buckBucket(bucketSize: Int): (String, String) => Int = {
  val func = (col1: String, col2: String) =>
    (Math.abs((col1 + "_" + col2).hashCode()) % bucketSize + 0)
  func
}
val bucketUDF = udf(buckBucket(100)) //here 100 is partially applied in advance.
Df.withColumn(“x”,bucketUDF(col1, col2))

//or, this is prefered
def buckBucket(bucketSize: Int): (String, String) => Int = {
  val func = (col1: String, col2: String) =>
    (Math.abs((col1 + "_" + col2).hashCode()) % bucketSize + 0)
  udf(func)
}
df.withColumn("x",buckBucket(100)(col1, col2)


def searchGeohash(geohashSet: Set[String], max: Int, min: Int) = {
  val func: (String => String) = (arg: String) => {
    val geohashKeys = (for (i <- max to min by -1) yield arg.slice(0, i))
    geohashKeys.find(x => geohashSet.contains(x)).getOrElse(arg.slice(0, min))
  }
  udf(func)
}
locationDF.select(col("*"))
      .withColumn("dynamicGeohash", searchGeohash(dynamicGeohashSet.value, maxPrecision, minPrecision)(col("geohash")))

```

``` //redundant
  def buckBuckets(bucketSize: Int)(col: String*): Int = {
    Math.abs(col.reduce(_ + "_" + _).hashCode()) % bucketSize + 0
  }
  
  val bucket1UDF = udf(Utils.buckBuckets(500)(_: String))
  val bucket2UDF = udf(Utils.buckBuckets(500)(_: String, _: String))
  val bucket3UDF = udf(Utils.buckBuckets(500)(_: String, _: String, _: String))
```

####2, 1 column to 1 column, with or without any parameters
Percentile, usually 5th, 25th, 50th, 75th, 95th are used
// anomalous function take parameters, prefered
``` 
// anomanous 
val percentile = udf((values: scala.collection.mutable.WrappedArray[Float], p: Float) => {

      val sortedValues = values.sorted
      val index = (sortedValues.length * p).toInt

      if (sortedValues.length % 2 == 0) {
        (sortedValues(Math.max(index - 1, 0)) + sortedValues(index)) / 2
      } else {
        sortedValues(index)
      }
    })
df.withColumn("99", percentile(col("dataList"), lit(0.99)))

//partial function
def lookupCatUdf = {
  val func = (categories: WrappedArray[WrappedArray[String]]) => //only get [0,last] here
    if (categories == null) 0L else categoriesMap(categories(0).last)
  udf(func)
}
metaBooks.select("asin", "categories")
  .withColumn("categories_index", lookupCatUdf(col("categories")))
}
```

#### multiple to multiple columns 3 in 5 out
```
 case class GroupedIndex(asin_index: Long, cat_index: Long, unixReviewTime: Long, asin_history: Array[Long], cat_history: Array[Long])
 def createHistorySeq(df: DataFrame, maxLength: Int): DataFrame = {

    val asinUdf = udf((asin_collect: Seq[Row]) => {
      val full_rows = asin_collect.sortBy(x => x.getAs[Long](2)).toArray

      val n = full_rows.length

      val range: Seq[Int] = if (maxLength < n) {
        (n - maxLength to n - 1)
      } else {
        (0 to n - 1)
      }

      range.map(x =>
        GroupedIndex(asin_index = full_rows(x).getAs[Long](0),
          cat_index = full_rows(x).getAs[Long](1),
          unixReviewTime = full_rows(x).getAs[Long](2),
          asin_history = full_rows.slice(0, x).map(row => row.getAs[Long](0)),
          cat_history = full_rows.slice(0, x).map(row => row.getAs[Long](1))))
    })

    val aggDF = df.groupBy("reviewerID_index")  //  3 in  5 out
      .agg(collect_list(struct(col("asin_index"), col("categories_index"), col("unixReviewTime"))).as("asin_collect"))

    aggDF.withColumn("item_history", asinUdf(col("asin_collect")))
      .withColumn("item_history", explode(col("item_history")))
      .drop("asin_collect")
      .select(col("reviewerID_index"),
        col("item_history.asin_index").as("asin_index"),
        col("item_history.cat_index").as("cat_index"),
        col("item_history.unixReviewTime").as("unixReviewTime"),
        col("item_history.asin_history").as("asin_history"),
        col("item_history.cat_history").as("cat_history"))
      .filter("size(asin_history) > 0 and size(cat_history) > 0")
  }
```

#### Window Functions, add a rank
```
def recommend4Items(featureDF: DataFrame, maxUsers: Float): DataFrame = {
  val results = predictUserItemPair(featureDF)
  results.groupBy("prediction").count().show()
  val window = Window.partitionBy("itemId").orderBy(desc("prediction"), desc("probability"))
  results.withColumn("rank", rank.over(window))
    .where(col("rank") <= maxUsers)
    .drop("rank")
}
```

### join scenarios 
#### one big table, one small table, broadcast the small table 
```
val dynamicGeohashSet = sc.broadcast(dynamicGeohashDF.select("dynamicGeohash")
  .collect().map(x => x(0).toString()).toSet)

locationDF.select(col("*"))
  .withColumn("dynamicGeohash", searchGeohash(dynamicGeohashSet.value, maxPrecision, minPrecision)(col("geohash")))
```
```
val ratings = sc.textFile(ratefile).map(x=>x.split("::")).map(x=> (x(0).toInt, x(1).toInt, x(2).toInt)).toDF("uid","iid","rate" )
val userDF = sc.textFile(userfile).map(x=>x.split("::")).map(x=> (x(0).toInt, x(1), x(2))).toDF("uid","gender","age" )
val userDFbr = broadcast(userDF)
userDFbr.cache()
val out = ratings.join(userDFbr, Seq("uid"))
    
``` 

#### one key is too big, get some control of the content or create multiple keys of the key
for example, count movies a user has reviewed.
``` 
df.sample(0.001).groupBy(col1) see the distribution
divide the heavy key into 10 randomly
groupby
then get the data together.

```


### tune a spark job
1. add memory 
2. add shuffle partitions and cores 
3. repartition 

### lazy load fix serialization
In spark, map, ruduce, groupby and other functions, driver gets everything closed in a closure, then sends it to executors.
Other things are done by driver, for example, val ran = new Random(), since the closure needs it then driver has to send it through socket, but ran is not serializable, then it cause problems. 
So it needs lazy load, lazy means, the driver does not new it or send the object, but the executer will new an object when it needs it.
In this example, each executor has an object of ran to produce random numnbers, it is better than to put the ran in closure in terms of distribution.
```
def getNegativeSamples(indexed: DataFrame): DataFrame = {
  val indexedDF = indexed.select("userId", "itemId", "label")
  val minMaxRow = indexedDF.agg(max("userId"), max("itemId")).collect()(0)
  val (userCount, itemCount) = (minMaxRow.getInt(0), minMaxRow.getInt(1))
  val sampleDict = indexedDF.rdd.map(row => row(0) + "," + row(1)).collect().toSet
  val dfCount = indexedDF.count.toInt
  import indexed.sqlContext.implicits._

@transient lazy val ran = new Random(System.nanoTime())

  val negative = indexedDF.rdd
    .map(x => {
      val uid = x.getAs[Int](0)
      val iid = Math.max(ran.nextInt(itemCount), 1)
      (uid, iid)
    })
    .filter(x => !sampleDict.contains(x._1 + "," + x._2)).distinct()
    .map(x => (x._1, x._2, 1))
    .toDF("userId", "itemId", "label")
  negative
}
```

Abstract class vs trait both define signatures of some methods which will be implemented later in child class
Abstract class takes constructors and parameters, used less and less.
Trait can be extended with multiple traits,  but trait does not have constructors, used more often.

Functions VS methods applied for udfs
No (x:String) things like this needed.
Anonymous functions are first-class functions → Function values are objects
Assign function values to variables.
Pass function values as arguments to higher order functions
```
val categoricalUDF = udf(Utils.categoricalFromVocabList(Array("F", "M")))
def categoricalUDF(list:Array[String]) = udf(Utils.categoricalFromVocabList(list))
categoricalUDF(Array("F", "M”))
```
Simple udf functions
```
  /**
    * find dynamic geohash given a fine geohash and the dynamic geohash dictionary
    */
  private def matchSingleGeohashUdf(geohashSet: Set[String], max: Int, min: Int) = {
    val func: (String => String) = (arg: String) => {
      val geohashKeys = (for (i <- max to min by -1) yield arg.slice(0, i))
      geohashKeys.find(x => geohashSet.contains(x)).getOrElse(arg.slice(0, min))
    }
    udf(func)
  }
```


### Functional language and why

Functional programming supports higher-order functions and lazy evaluation features.

Functional programming languages don’t support flow Controls like loop statements and conditional statements like If-Else and Switch Statements. They directly use the functions and functional calls.

Efficient Parallel Programming − Functional programming languages have NO Mutable state, so there are no state-change issues. One can program "Functions" to work parallel as "instructions". Such codes support easy reusability and testability. Especially for big data.

Lazy Evaluation − Functional programming supports Lazy Functional Constructs like Lazy Lists, Lazy Maps, etc.

Apply function in Scala

1. Every function in Scala can be treated as an object, every object can be treated as a function, provided it has the apply method.
There are many usage cases when we would want to treat an object as a function.
Such objects can be used in the function notation:
```$xslt
// we will be able to use this object as a function, as well as an object
object Foo {
  var y = 5
  def apply (x: Int) = x + y
}

Foo (1) // using Foo object in function notation
```
2. The most common scenario of using an apply function is a factory pattern, and companion. Syntactic sugar, and multiple ways of building objects
```
class Foo(x) {
  val y = 5
}

object Foo {
  def apply (x: Int) = New class Foo(x)
  def apply (x: Float) = New class Foo(x)
}

val foo1 = Foo (1) // build an object
val foo2 = Foo (1.0) // build an object
```
3. You can also define an apply function in class, after you build an object of that class in whatever way.
then you can call apply function. In this exmaple,
```
class c1(x:Float) ={
    def apply(x:String)= {
        x.toInt
    }
}
class c2(y:Float) extends c1(y:Float)
object c2 ={
    def apply(y:Float) = new c2(y)
}

val tmp = c2(10.0)
tmp("10.0") // no need to call a method name. 
```

### scala companion
object methods, apply load any methods which return that class, as well as utils functions no need to exposed to comsuners.
real logic related methods go to class.
example BigDL DenseTensor 

``` 
class NeuralCF[T: ClassTag](
    val userCount: Int,
    val itemCount: Int,
    val numClasses: Int,
    val userEmbed: Int = 20,
    val itemEmbed: Int = 20,
    val hiddenLayers: Array[Int] = Array(40, 20, 10),
    val includeMF: Boolean = true,
    val mfEmbed: Int = 20)(implicit ev: TensorNumeric[T])
  extends Recommender[T] {

  override def buildModel(): AbstractModule[Tensor[T], Tensor[T], T] = {
  ... 
    val model = Model[T](input, linearLast)
    model.asInstanceOf[AbstractModule[Tensor[T], Tensor[T], T]]
  }
}

object NeuralCF {
  /**
   * The factory method to create a NeuralCF instance.
   */
  def apply[@specialized(Float, Double) T: ClassTag](
      userCount: Int,
      itemCount: Int,
      numClasses: Int,
      userEmbed: Int = 20,
      itemEmbed: Int = 20,
      hiddenLayers: Array[Int] = Array(40, 20, 10),
      includeMF: Boolean = true,
      mfEmbed: Int = 20)(implicit ev: TensorNumeric[T]): NeuralCF[T] = {
    new NeuralCF[T](userCount, itemCount, numClasses, userEmbed,
      itemEmbed, hiddenLayers, includeMF, mfEmbed).build()
  }

    def loadModel[T: ClassTag](
      path: String,
      weightPath: String = null)(implicit ev: TensorNumeric[T]): NeuralCF[T] = {
    ZooModel.loadModel(path, weightPath).asInstanceOf[NeuralCF[T]]
  }
    def xyz(x:int, y:float, z:string) ={
    x+y+z.toFloat()}

```

### Scala implicit

implicit class

implicit object(need better understanding)
    Like any object, an implicit object is a singleton but it is marked implicit so that the compiler can find if it is looking for an implicit value of the appropriate type.
    A typical use case of an implicit object is a concrete, singleton instance of a trait which is used to define a type class.
    People use implicit object instead of implicit class so you don't need to explicitly import the class with implicits, since implicits in companion object will be searched by Scala compiler as well.
```
trait TensorNumeric[@specialized(Float, Double) T]
abstract UndefinedTensorNumeric(typeName:String) extends TensorNumeric
object TensorNumeric{
    implicit object NumericFloat extends UndefinedTensorNumeric[Float]{}
    implicit object NumericDouble extends UndefinedTensorNumeric[Double]{}
}
val predict: Int = ev.toType[Int](_output.max(1)._2.valueAt(1))

```
### MLLib and ML Vectors
ML Vector
DataFrame related APIs use org.apache.spark.ml.linalg.Vector, in double, but the old mllib use org.apache.spark.mllib.linalg.Vector,
org.apache.spark.mllib.util.MLUtils.convertVectorColumnsToML and other APIs are used to convert the data from one type to another.
BigDL uses array usually float, it needs array2vec or vec2array
```
  val array2vec = udf((arr: scala.collection.mutable.WrappedArray[Float]) => {
    val d = arr.map(x => x.toDouble)
    Vectors.dense(d.toArray)
  })


  val vec2array = udf((arr: scala.collection.mutable.WrappedArray[Double]) => {
    val d = arr.map(x => x.toFloat)
    Array(d.toArray)
  })
```

### advanced spark programming 
1. RDD
 the more partitions you have, the more parellism you have, each partition requires a thread of computation
 number of partitions should > number of cores
 spark.parallelize(["fish", "cat"])
 spark.read.text(pathtofile)
 rdd.filter().coalesce(right number of your cluster)
 cassandra.input.split.size default = 100,000 rows for each partition 
 
 
2.
questions:
1. when need lazy 
2. serilization, what is seriable, what iss not serilable
3. window function
4. common implicits
5. difference between dataset and dataframes
6. use data as sql longuage, need register table
7. mapPartitions() vs map, when to use which
8. serializer(why KryoSerilizer is better)

## serializable issue and fixation


## interview questions 
25. What are the different levels of persistence in Spark?
DISK_ONLY 
MEMORY_ONLY_SER
MEMORY_ONLY 
OFF_HEAP 
MEMORY_AND_DISK 
MEMORY_AND_DISK_SER 

