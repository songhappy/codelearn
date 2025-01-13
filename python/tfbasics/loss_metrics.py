# the first dimention is batch size
from sklearn.metrics import roc_auc_score, accuracy_score
import tensorflow as tf
y_true = [[0., 1.], [0., 0.]]
y_pred = [[1., 1.], [1., 0.]]
# Using 'auto'/'sum_over_batch_size' reduction type.
mae = tf.keras.losses.MeanAbsoluteError()
x = mae(y_true, y_pred).numpy()
print(x)

m = tf.keras.metrics.AUC()
m.update_state([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
print(m.result().numpy())

sk_auc = roc_auc_score([0, 0, 1, 1], [0, 0.5, 0.3, 0.9])
print(sk_auc)

y_true = [1, 2]
y_pred = [[0.05, 0.95, 0], [0.1, 0.8, 0.1]]
scce = tf.keras.losses.SparseCategoricalCrossentropy()
print(scce(y_true, y_pred).numpy())