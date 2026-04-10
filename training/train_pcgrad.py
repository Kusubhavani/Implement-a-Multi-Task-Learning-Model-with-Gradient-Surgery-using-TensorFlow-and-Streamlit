import tensorflow as tf
import pandas as pd
from models.model import MultiTaskModel
from data.generate_data import create_dataset
from training.utils import cosine_similarity, apply_pcgrad

model = MultiTaskModel()

loss_a_fn = tf.keras.losses.BinaryCrossentropy()
loss_b_fn = tf.keras.losses.MeanSquaredError()

optimizer = tf.keras.optimizers.Adam(0.001)

dataset = create_dataset()

logs = []
grad_logs = []
step_counter = 0

for epoch in range(5):
    for x, (y_a, y_b) in dataset:

        with tf.GradientTape(persistent=True) as tape:
            pred_a, pred_b = model(x, training=True)

            loss_a = loss_a_fn(y_a, pred_a)
            loss_b = loss_b_fn(y_b, pred_b)

        grad_a = tape.gradient(loss_a, model.backbone.trainable_variables)
        grad_b = tape.gradient(loss_b, model.backbone.trainable_variables)

        cos_sim = cosine_similarity(grad_a, grad_b)

        grad_logs.append({
            "step": step_counter,
            "cosine_similarity": float(cos_sim)
        })

        grad_a, grad_b = apply_pcgrad(grad_a, grad_b)

        final_backbone = [ga + gb for ga, gb in zip(grad_a, grad_b)]

        grad_head_a = tape.gradient(loss_a, model.head_a.trainable_variables)
        grad_head_b = tape.gradient(loss_b, model.head_b.trainable_variables)

        all_vars = (
            model.backbone.trainable_variables +
            model.head_a.trainable_variables +
            model.head_b.trainable_variables
        )

        all_grads = final_backbone + grad_head_a + grad_head_b

        optimizer.apply_gradients(zip(all_grads, all_vars))

        logs.append({
            "epoch": epoch,
            "loss_a": float(loss_a),
            "loss_b": float(loss_b)
        })

        step_counter += 1

pd.DataFrame(logs).to_csv("results/pcgrad_metrics.csv", index=False)
pd.DataFrame(grad_logs).to_csv("results/gradient_conflict.csv", index=False)