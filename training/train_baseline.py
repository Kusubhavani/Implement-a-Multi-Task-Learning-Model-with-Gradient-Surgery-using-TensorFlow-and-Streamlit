import tensorflow as tf
import pandas as pd
from models.model import MultiTaskModel
from data.generate_data import create_dataset

model = MultiTaskModel()

loss_a_fn = tf.keras.losses.BinaryCrossentropy()
loss_b_fn = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(0.001)

dataset = create_dataset()

logs = []

@tf.function
def train_step(x, y_a, y_b):
    with tf.GradientTape() as tape:
        pred_a, pred_b = model(x, training=True)

        loss_a = loss_a_fn(y_a, pred_a)
        loss_b = loss_b_fn(y_b, pred_b)

        total_loss = loss_a + loss_b

    grads = tape.gradient(total_loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))

    return loss_a, loss_b

for epoch in range(5):
    for step, (x, (y_a, y_b)) in enumerate(dataset):
        la, lb = train_step(x, y_a, y_b)

    logs.append({
        "epoch": epoch,
        "loss_a": float(la),
        "loss_b": float(lb)
    })

pd.DataFrame(logs).to_csv("results/baseline_metrics.csv", index=False)