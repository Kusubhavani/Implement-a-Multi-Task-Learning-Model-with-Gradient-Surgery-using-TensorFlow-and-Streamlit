import tensorflow as tf

class MultiTaskModel(tf.keras.Model):
    def __init__(self):
        super().__init__()

        # Shared backbone
        self.backbone = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.Dense(32, activation='relu')
        ])

        # Task A (classification)
        self.head_a = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1, activation='sigmoid')
        ])

        # Task B (regression)
        self.head_b = tf.keras.Sequential([
            tf.keras.layers.Dense(16, activation='relu'),
            tf.keras.layers.Dense(1)
        ])

    def call(self, x, training=False):
        shared = self.backbone(x, training=training)
        return self.head_a(shared), self.head_b(shared)