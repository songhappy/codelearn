# example:
# file:///Users/guoqiong/intelWork/projects/travelSky/shane/TravelSky_apachelog_full.html
# https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.html
# https://github.com/pandas-dev/pandas/blob/master/doc/cheatsheet/Pandas_Cheat_Sheet.pdf
# https://chrisalbon.com/python/data_wrangling/pandas_join_merge_dataframe/
# https://jakevdp.github.io/PythonDataScienceHandbook/04.14-visualization-with-seaborn.html
# https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.html
# basic plot and resample to take a look at data

import numpy as np
import pandas as pd

input_path = "../../data/movielens/ml-1m/ratings.dat"

ratings = []
with open(input_path) as infile:
    for cnt, line in enumerate(infile):
        x = line.strip().split("::")
        y = map(lambda item: int(item), x)
        ratings.append(y)
    print("total records:", cnt)

df = pd.DataFrame(ratings, columns=['uid', 'mid', 'rate', 'timestamp'])
print(df.head())
print(df.groupby('mid').count())
df.groupby('mid').agg('count')
df.groupby(['col1', 'col2'], as_index=False).agg(lambda x: list(x))

## python libs with a focus on pandata for quick analysis

### create data frame
# from csv file
log_df = pd.read_csv("log.csv")  # load csv into dataframe
# from numpy matrix
pd.DataFrame(np.random.randn(10, 5))
d = {'col1': ts1, 'col2': ts2}
df = pd.DataFrame(data=d, index=index)

### Basic operations
pd.DataFrame.from_csv(“csv_file”)
pd.read_csv(“csv_file”)
data = pd.read_csv('output_list.txt', sep=" ", header=None)
data.columns = ["a", "b", "c", "etc."]
pd.read_excel("excel_file")
pd.melt(df)  # columns into rows
pd.pivot(df)  # rows into columns

df.to_csv("data.csv", sep=",", index=False)
df.info()
print(df.describe())
df.columns
df.dropna(axis=0, how='any')
df.replace(to_replace=None, value=None)
pd.isnull(object)
df.drop('feature_variable_name', axis=1)
pd.to_numeric(df["feature_name"], errors='coerce')  # to float
df.as_matrix() % to
numpy
matrix
df.head(n)
df.loc[col_name]
df.iloc(18)
df.resample(freqency).mean().fillna(value=0)

### DataFrame operations and stats
df["height"].apply(*lambda *height: 2 * height)


def multiply(x):
    return x * 2


df["height"].apply(multiply)
df.rename(columns={df.columns[2]: 'size'}, inplace=True)
df["name"].unique()
new_df = df[["name", "size"]]
### summarize and sort
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

df.sort_values(ascending=False)

df.loc([0], ['size'])

# run basic queries, filter
df.query('occupation=="writer"')
# filter
df[df["size"] == 5]
df.filter(df["name"] == 5)
df.filter(like='bbi', axis=0)  # get rows which have "bbi"
df.filter(items=['col1', 'col2'])  # get columns col1 col2
# https://stackoverflow.com/questions/17071871/how-to-select-rows-from-a-dataframe-based-on-column-values

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_style("whitegrid")
df = pd.read_csv("log.csv")  # load csv into dataframe
df = pd.read_excel("test.exel")
df.to_csv("out_file)read
df.head()
df.info()
print(df.describe())

df.groupby('host').agg('count')
df.replace(to_replace=None, value=None)
df['processed_date'] = df['logDate'].str.replace(':', ' ', 1).str.replace(r'\[|\]', '')
# add a column and set index for plot
one_ip_sample_df = df[df['host'] == '122.119.64.67']  # filter example
one_ip_sample_df['dt_index'] = pd.to_datetime(one_ip_sample_df['processed_date'])
one_ip_sample_df = one_ip_sample_df.set_index('dt_index')


def resample_plot(df, freq, col='value'):
    resample_df = df[col].resample(freq).mean().fillna(value=0)
    resample_df.plot()


resample_plot(one_ip_sample_df, '30Min')

# plot original values and resampled values in the same figure
new_df_1['in_value_min'].plot()
resample_plot(new_df_1, '1D', col='in_value_min')
plt.legend(['sampling freq = 1 hour(original)', 'sampling freq = 1 day'])
plt.title("in_value_min");

# read as csv into dataframe
import pandas

try:
    df = pandas.read_csv("./data/WF1_632822_20181120.txt")
except Exception as e:
    print("problem in reading files")

# dump and load objects, should be larger than the demofile1
import pickle

X_file = open(path0 + "demofile2.txt", 'rb')
pickle.dump(X, X_file)
X = pickle.load(X_file)
print(X)
