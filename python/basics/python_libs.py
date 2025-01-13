# numpy
import numpy as np

b = np.array([(1.5, 2, 3), (4, 5, 6)], dtype=float)
print(b.shape)
np.linspace(1, 20, 2)
np.ones((2, 3, 4), dtype=np.int)
a = np.random.random((2, 3))
c = a - b
d = a + b
e = a * b
f = a / b
g = np.multiply(a, b)  # same as *, element by element, dot is the matrix calculation
h = a.dot(np.array([1, 2, 3]))
print(np.array([1, 2, 3]).shape)
np.sqrt(a)
np.power(a, 3)
np.sin(a)
np.cos(a)
np.log(a)
np.exp(a)

print(c, d)
print(np.sum(a))
print(np.min(a, 0))
print(np.max(a))
print(a.sum())
print(a.min())
print(np.mean(a))
print("********")
print(e, f, g, h)

# view and original object share same data, but create another way of viewing the data
x = a[:]
print(x)
x.resize(3, 2)
print(x)
print(a)
np.concatenate((a, b), axis=0)
np.transpose(a)
print(np.hsplit(a, 3))  # split at horizontal dimention into 3 parts
print(np.vsplit(a, 2))

np.savez('my_array.npz', a=a, b=b, c=c, d=d)
np.save('my_array1', a)
y = np.load("my_array.npz")
print(y['b'])
print(np.load('my_array1.npy'))

# scipy
print("------------")
print(a, b)
print(np.inner(a, b))  # [[sum(a1i*b1i), sum(a2i*b1i)],[sum(a2i*b1i), sum(a2i*b2i)]]
print(np.outer(a, b))
print("------------")

print(np.inner(np.array([(1, 2, 3), (1, 2, 3)]), np.array([(4, 5, 6), (1, 2, 3)])))
B = np.asmatrix(b)
np.mat(np.random.random((10, 5)))
np.matrix(np.random.random((2, 3)))

# Numpy matrices are strictly 2-dimensional, while numpy arrays (ndarrays) are N-dimensional. Matrix objects are a subclass of ndarray, so they inherit all the attributes and methods of ndarrays.
# The main advantage of numpy matrices is that they provide a convenient notation for matrix multiplication: if a and b are matrices, then a*b is their matrix product.
A = np.asmatrix(a)
B = np.asmatrix(b)
from scipy import linalg

print(A.T)
print(A * B.T)  # matrix multipication
print("-----------")
U, s, Vh = linalg.svd(B)
print(U, s, Vh)

# pandas
# read as csv into dataframe
import pandas

try:
    df = pandas.read_csv("/Users/guoqiong/intelWork/git/goldwind-poc/data/WF1_632822_20181120.txt")
except Exception as e:
    print("problem in reading files")
print(df.head(5))
headers = ["col1", "col2"]
a = df.nsmallest(5, 'value')
print(type(a))
print(a)
df2 = df[df['value'] > 7]
print(df2)


def resample_plot(df, freq, col='value'):
    resample_df = df[col].resample(freq).mean().fillna(value=0)
    resample_df.plot()
