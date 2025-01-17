# Python CheetSeet
### outline
1. basic concepts, numbers, data structures,files in and out
2. useful libraries, numpy, scipy, pandas, scikit-learn
3. pyspark and related
4. connecting python to jvm of scala and java
5. databases
6. tensorflow and related model
7. algorithms

## Install Python packages 

code and notes Pip, easy_install for python packages
install an older version of python
``` bash
Brew unlink python #brew is apt_get(linux) for mac
brew install python@3.7
brew link -f python@3.7
```

```bash
Pip for python 2 default
Pip install tensorflow
Pip install numpy
Pip install scipy
Pip install mat
pip install --upgrade matplotlib
pip install  keras==2.2
Pip install jupyter
pip install seaborn
Brew install python3, then you have python3 and pip3, pip3 is for python3, if you do not have pip3
python3 -m pip install --user --upgrade tensorflow
```
### problems have seen
1. Install python 3.6.5 since 3.7.0 has no tensorflow(old)
https://stackoverflow.com/questions/51125013/how-can-i-install-a-previous-version-of-python-3-in-macos-using-homebrew

2. The Jupyter Notebook needs ipython kernel, this fix the dead kernel error
python2 -m ipykernel install --user
python3 -m ipykernel install --user

3. Python setup.py if not found in pip
Annaconda, conda install other packages

4. requests.exceptions.SSLError: EOF occurred in violation of protocol (_ssl.c:661)
   LIB-2176:osf-cli nmunn$
``` 
pip install pyopenssl   or use python3
```

## python basic concepts
### some syntax and wrappers
```python
class Foo:
    "this is a test"
    def __init__(self):
        self._a = 0 #_variable name means private menber of the class
        self._b = 1
    def __call__(self, a): #call an instance as method
        print(a)
foo = Foo()
print(__name__) # __name__ is assigned by python interpretor when it reads a source file at that level

class EnvCore:  
    pass
class Env(gym.Env): # it is a wrapper of EnvCore
    def __init__(self):
        self._env = EnvCore()
    def step(self, action):
        pass
```

### python numbers
int, float and complex how many bytes per each number
```python
print(type(a))
print(isinstance(a, int))
import sys
a = [1,2,3,4]
print(sys.getsizeof(a))
# casting
int(3.0); float(4); print(str(a))
```
### python datastructures
list based on array, used as stack
```python
a = list()
a = [1,2,3,4]
b = a.copy()
a.append(5)
a.pop(0) #o(k)
a.pop() #o(1)
a.remove(2) #o(n)
a.insert(0, 9) #o(n)
a.extend([6,7]) #o(k)
a.sort(reverse=True)#o(n log n)
len(a)
```

collections.deque, based on doublely linked list, used as queue
```python
import collections
a = collections.deque([1, 2, 3, 4, 5])
a.copy()
a.append(6) #o(1)
a.appendleft(0) #o(1)
a.pop() #o(1)
a.popleft() #o(1)
a.extend([7,8,9]) #o(k)
a.remove(2) #o(n)
print(a.index(3))
```

tuples, can not change elements
```python
a = (1,2,3)
print(a.index(1)) #index of value
print(a[0])
```

set, unique unordered
```python
a=set()
a={1,2,3}
a.add(4)
print(a.difference({5}))
print(a.intersection({2,3}))
print(a.intersection_update({2,3}))
a.union({4, 6, 7})
# a.discard()
# a.remove()
# a.pop()
# a.clear()
a.copy()
```

string, storaged in list
```python
a = "helloworld"
a[-1]
a + "how are you"
```

dictionary
```python
a = dict()
print(a)
a =  {'name':'Jack', 'age': 26}
print(a.get("name"))
a["name"]
print(a.update({"sex": "female"}))
print(a)
a["address"]="cupertino"
print(a)
print(a.pop("key"))
print(a.popitem())
a.keys()
a.values()
print(a)
```

### read and write files 
```python
X = ["1,2,3,4,5","6,7,8,9,10"]
with open(path0 +"demofile.txt", "w") as f:
    for item in X:
     f.write("%s\n" % item)

with open(path0 +"demofile.txt") as f:
    X = [line.strip() for line in f]
```
```python
# pickle serilize objects directly
import pickle
X_file = open(path0 +"demofile2.txt", 'w')
pickle.dump(X, X_file)
X_file = open(path0 +"demofile2.txt", 'r')
X = pickle.load(X_file)
```

## python libs with a focus on pandata for quick analysis
example:
file:///Users/guoqiong/intelWork/projects/travelSky/shane/TravelSky_apachelog_full.html
https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf
https://chrisalbon.com/python/data_wrangling/pandas_join_merge_dataframe/
https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
basic plot and resample to take a look at data

### create data frame 
```python
# from csv file
log_df = pd.read_csv("log.csv") #load csv into dataframe
# from numpy matrix
pd.DataFrame(np.random.randn(10, 5))
d = {'col1': ts1, 'col2': ts2}
df = pd.DataFrame(data=d, index=index)
```
### Basic operations
```python
pd.DataFrame.from_csv(“csv_file”)
pd.read_csv(“csv_file”)
data = pd.read_csv('output_list.txt', sep=" ", header=None)
data.columns = ["a", "b", "c", "etc."]
pd.read_excel("excel_file")
df.to_csv("data.csv", sep=",", index=False)
df.info()
print(df.describe())
df.columns 
df.dropna(axis=0, how='any')
df.replace(to_replace=None, value=None)
pd.isnull(object)
df.drop('feature_variable_name', axis=1)
pd.to_numeric(df["feature_name"], errors='coerce') #to float
df.as_matrix()  %to numpy matrix
df.head(n)
df.loc[feature_name]
pd.melt(df) #columns into rows  
pd.pivot(df) #rows into columns
```

### DataFrame operations
```python
df["height"].apply(*lambda* height: 2 * height)
def multiply(x):
 return x * 2
df["height"].apply(multiply)
df.rename(columns = {df.columns[2]:'size'}, inplace=True)
df["name"].unique()
new_df = df[["name", "size"]]
```
### summarize and sort
```python
# Sum of values in a data frame
df.sum()
# Lowest value of a data frame
df.min()
# Highest value
df.max()
# Index of the lowest value
df.idxmin()
# Index of the highest value
df.idxmax()
# Statistical summary of the data frame, with quartiles, median, etc.
df.describe()
# Average values
df.mean()
# Median values
df.median()
# Correlation between columns
df.corr()
# To get these values for only one column, just select it like this#
df["size"].median()

df.sort_values(ascending = False)

df.loc([0], ['size'])
```
run basic queries 
```
users.query('occupation=="writer"')
#filter
df[df["size"] == 5]
```
example
```python
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
sns.set_style("whitegrid")
df = pd.read_csv("log.csv") #load csv into dataframe
df = pd.read_excel("test.exel")
df.to_csv("out_file)
df.head()
df.info()
print(df.describe())
from dateutil import parser
df.groupby('host').agg('count')
df.replace(to_replace=None, value= None)
df['processed_date'] = df['logDate'].str.replace(':', ' ', 1).str.replace(r'\[|\]', '')
#add a column and set index for plot
one_ip_sample_df = df[df['host'] == '122.119.64.67']
one_ip_sample_df['dt_index'] = pd.to_datetime(one_ip_sample_df['processed_date'])
one_ip_sample_df = one_ip_sample_df.set_index('dt_index')
def resample_plot(df,freq,col='value'):
    resample_df = df[col].resample(freq).mean().fillna(value=0)
    resample_df.plot()
resample_plot(one_ip_sample_df,'30Min')

#plot original values and resampled values in the same figure
new_df_1['in_value_min'].plot()
resample_plot(new_df_1,'1D',col='in_value_min')
plt.legend(['sampling freq = 1 hour(original)','sampling freq = 1 day'])
plt.title("in_value_min");
```
## pyspark and related

### add memory for spark in python module
```bash 
    export _JAVA_OPTIONS="-Xms512m -Xmx1024m"
```

```python
    from zoo import init_spark_on_local
    conf = {"spark.executor.memory":"20g","spark.driver.memory":"20g"}
    sc = init_spark_on_local(cores=8, conf=conf)
```

### pyspark create dataframe 
````python
    >>> spark = SparkSession.builder.master("local").getOrCreate()
    >>> df = spark.createDataFrame([(0.5,)], ["values"])
    >>> df = spark.createDataFrame([[0.5],[0.6]], ["values"])
    >>> data = [(0, Vectors.dense([-1.0, -1.0 ]),),
             (1, Vectors.dense([-1.0, 1.0 ]),),
             (2, Vectors.dense([1.0, -1.0 ]),),
             (3, Vectors.dense([1.0, 1.0]),)]
    >>> df = spark.createDataFrame(data, ["id", "features"])
    
    >>> df = spark.createDataFrame(
    [(0, ["a", "b", "c"]), (1, ["a", "b", "b", "c", "a"])],
    ["label", "raw"])
    
    >>> df1 = spark.createDataFrame([(Vectors.dense([5.0, 8.0, 6.0]),)], ["vec"])
    
    >>> df = spark.createDataFrame([(["a", "b", "c"],)], ["words"])

#pyspark read and write textfile
rdd.coalesce(1).saveAsTextFile(...)
users = sc.textFile("/tmp/movielens/ml-1m/users.dat") \
    .map(lambda l: l.split("::")[0:4])\
    .map(lambda l: (int(l[0]), l[1], int(l[2]), int(l[3])))\   
    
#pyspark read and write parquet/jason files into and outof dataframes
peopleDF = spark.read.json("examples/src/main/resources/people.json")
peopleDF.write.json("people.json)

peopleDF.write.parquet("people.parquet")
parquetFile = spark.read.parquet("people.parquet")   

# pandas DF to pyspark DF
import numpy as np
import pandas as pd

# Enable Arrow-based columnar data transfers
spark.conf.set("spark.sql.execution.arrow.enabled", "true")

# Generate a pandas DataFrame
pdf = pd.DataFrame(np.random.rand(100, 3))

# Create a Spark DataFrame from a pandas DataFrame using Arrow
df = spark.createDataFrame(pdf)  
```

## connecting to scala and jvm


