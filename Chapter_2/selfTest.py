import numpy as np
# weights = []
# weights0=[[1,2,3],[4,5,6]]
# def weight_plus(weights):
#         for i in range(10):
#             weights += weights0
#         return weights
#
# print(weight_plus(weights))
import tensorflow as tf
import keras.layers as layers
class CustomEmbedding(tf.keras.layers.Layer):

    def __init__(self, input_dim, output_dim, mask_zero=False, **kwargs):
        super(CustomEmbedding, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.mask_zero = mask_zero
        self.embeddings = None

    def build(self, input_shape):
        e1 = self.add_weight(
            shape=(int(self.input_dim / 2), self.output_dim),
            dtype="float32", trainable=True,
            name="e1")

        e2 = self.add_weight(
            shape=(self.input_dim - int(self.input_dim / 2), self.output_dim),
            dtype="float32", trainable=True,
            name="e2")

        self.embeddings = tf.concat((e1, e2), 0)

        tf.print(self.embeddings)

    def call(self, inputs):
        return tf.nn.embedding_lookup(self.embeddings, inputs)

    def compute_mask(self, inputs, mask=None):
        if not self.mask_zero:
            return None
        return tf.not_equal(inputs, 0)


model = tf.keras.Sequential()
model.add(layers.Embedding(3, 32, trainable=False))
model.add(layers.LSTM(32))
model.add(layers.Dense(16, "relu"))
model.add(layers.Dense(2, "softmax"))

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

model.fit(data)