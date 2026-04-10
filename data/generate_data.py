import numpy as np
import tensorflow as tf

def create_dataset(samples=5000):
    # Generate input features
    X = np.random.randn(samples, 10)

    # Task A: binary classification
    y_a = (np.sum(X, axis=1) > 0).astype(int)

    # Task B: regression (conflicting signal)
    y_b = np.sum(X * np.random.randn(10), axis=1)

    dataset = tf.data.Dataset.from_tensor_slices(
        (X.astype('float32'), (y_a.astype('float32'), y_b.astype('float32')))
    )

    return dataset.shuffle(1000).batch(32)