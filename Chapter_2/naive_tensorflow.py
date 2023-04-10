import tensorflow as tf
from keras import optimizers
from keras.datasets import mnist
import math
import numpy as np

# maybe can solve some kind of error like 'framework problem'
# tf.compat.v1.disable_eager_execution()
# session = tf.compat.v1.Session()
# init = tf.compat.v1.global_variables_initializer()
# session.run(init)

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


# 2.5 Tensorflow realize first e.g again

# input transformation
class NaiveDense:
    def __init__(self, input_size, output_size, activation):
        self.activation = activation
        w_shape = (input_size, output_size)  # 根据input、output创建一个矩阵并且随机初始化
        w_initial_value = tf.random.uniform(w_shape, minval=0, maxval=1e-1)  # 随机初始化的最大值为0.1，最小为0
        self.W = tf.Variable(w_initial_value)  # 初始化权重

        b_shape = (output_size,)  # 创建一个点积右侧向量的零向量
        b_initial_value = tf.zeros(b_shape)
        self.b = tf.Variable(b_initial_value)

    def __call__(self, inputs):  # 向前传播
        return self.activation(tf.matmul(inputs, self.W) + self.b)

    @property
    def weights(self):  # 获取该层权重的便捷方法
        return [self.W, self.b]


# layers connection
class NaiveSequential:
    def __init__(self, layers):
        self.layers = layers

    def __call__(self, inputs):
        x = inputs
        for layer in self.layers:
            x = layer(x)
            return x

    @property
    def weights(self):
        weights = []
        for layer in self.layers:
            weights += layer.weights
        return weights


model = NaiveSequential([
    NaiveDense(input_size=28 * 28, output_size=512, activation=tf.nn.relu),
    NaiveDense(input_size=512, output_size=10, activation=tf.nn.softmax)
])
assert len(model.weights) == 4


# batch creatrer


class BatchGenerator:
    def __init__(self, images, labels, batch_size=128):
        assert len(images) == len(labels)
        self.index = 0
        self.images = images  # MNIST_pictures input
        self.labels = labels  # each pictures' index?
        self.batch_size = batch_size  # each batch's size
        self.num_batches = math.ceil(len(images) / batch_size)  # how many batches r there

    def next(self):  # The purpose is to pile up image data
        images = self.images[self.index: self.index + self.batch_size]  # pooling
        labels = self.labels[self.index: self.index + self.batch_size]  # rewrite labels according to how to pool
        self.index += self.batch_size  # mark-pointer
        return images, labels


# realize one training step
# remember:
# (1) 计算模型对图像批量的预测值。
# (2) 根据实际标签，计算这些预测值的损失值。
# (3) 计算损失相对于模型权重的梯度。
# (4) 将权重沿着梯度的反方向移动一小步。
# to calculate gradiation TensorFlow GradientTape should be utilized

learning_rate = 1e-3


def update_weights(gradients, weights):
    for g, w in zip(gradients, weights):
        w.assign_sub(g * learning_rate)  # assign_sub equivalent to TensorFlow's '-='


# actually you won't use aforementioned function, instead, following's keras_Optimizer

# optimizer = optimizers.SGD(learning_rate=1e-3)  # set learning rate
#
#
# def update_weights(gradients, weights):
#     optimizer.apply_gradients(zip(gradients, weights))


def one_training_step(model, images_batch, labels_batch):
    with tf.GradientTape() as tape:  # # calling and prediction in GradientTape
        predictions = model(images_batch)
        per_sample_losses = tf.keras.losses.sparse_categorical_crossentropy(
            labels_batch, predictions)  # #
        average_loss = tf.reduce_mean(
            per_sample_losses)
        # calculate the gradient of loss, "output-gradient" is a list and each element fits weights in "model.weights"
    gradients = tape.gradient(average_loss, model.weights)
    update_weights(gradients, model.weights)
    return average_loss


# 2.5.3 Complete training loop
# one-training-step means using aforementioned one-time training in each batch and
# complete training means repeating many, many times
def fit(model, images, labels, epochs, batch_size=128):
    for epoch_counter in range(epochs):
        print(f"Epoch {epoch_counter}")  # show time of each batch
        batch_generator = BatchGenerator(images, labels)  # invoke BatchGenerator
        for batch_counter in range(batch_generator.num_batches):
            images_batch, labels_batch = batch_generator.next()  # index and label marked
            loss = one_training_step(model, images_batch, labels_batch)  # calling function invoke
            if batch_counter % 100 == 0:
                print(f"loss at batch {batch_counter}: {loss:.2f}")

if __name__ == '__main__':
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images = train_images.reshape((60000, 28 * 28))
    train_images = train_images.astype("float32") / 255
    test_images = test_images.reshape((10000, 28 * 28))
    test_images = test_images.astype("float32") / 255
    # model.compile(optimizer='adam',
    #               loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    #               metrics=['accuracy'])
    fit(model, train_images, train_labels, epochs=10, batch_size=128)

    predictions = model(test_images)
    predictions = predictions.numpy()
    predicted_labels = np.argmax(predictions, axis=1)
    matches = predicted_labels == test_labels
    print(f"accuracy: {matches.mean():.2f}")
