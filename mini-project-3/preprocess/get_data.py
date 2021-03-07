# %%
import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
x_train, x_test = x_train[..., np.newaxis] / 255.0, x_test[..., np.newaxis] / 255.0

# %%
plt.imshow(x_train[0, :, :, 0])
plt.colorbar()
plt.show()

# %%
