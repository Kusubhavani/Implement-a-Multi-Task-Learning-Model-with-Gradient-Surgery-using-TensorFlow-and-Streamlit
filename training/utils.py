import tensorflow as tf
import numpy as np

def flatten_gradients(grads):
    return tf.concat([tf.reshape(g, [-1]) for g in grads if g is not None], axis=0)

def cosine_similarity(grad_a, grad_b):
    a = flatten_gradients(grad_a)
    b = flatten_gradients(grad_b)

    return tf.reduce_sum(a * b) / (tf.norm(a) * tf.norm(b) + 1e-8)

def apply_pcgrad(grad_a, grad_b):
    processed_a, processed_b = [], []

    for g_a, g_b in zip(grad_a, grad_b):
        if g_a is None or g_b is None:
            processed_a.append(g_a)
            processed_b.append(g_b)
            continue

        dot = tf.reduce_sum(g_a * g_b)

        if dot < 0:
            g_a = g_a - (dot / (tf.norm(g_b)**2 + 1e-8)) * g_b
            g_b = g_b - (dot / (tf.norm(g_a)**2 + 1e-8)) * g_a

        processed_a.append(g_a)
        processed_b.append(g_b)

    return processed_a, processed_b