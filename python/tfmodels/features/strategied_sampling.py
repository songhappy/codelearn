import pandas as pd
import numpy as np
df = pd.DataFrame({'col':np.random.randn(100), 'target':np.random.randint(low = 0, high = 2, size=100)})
df1 = df[df['target']==1]
df0 = df[df['target']==0]
df1 = df1.groupby('target').apply(lambda x: x.sample(n = 100, replace=True)).reset_index(drop=True)
df0 = df0.groupby('target').apply(lambda x: x.sample(n = 50, replace=True)).reset_index(drop=True)
out = df1.append(df0)
print(out.groupby('target').size())

# randomly split data
import random
with open("datafile.txt", "rb") as f:
    data = f.read().split('\n')

random.shuffle(data)

train_data = data[:50]
test_data = data[50:]