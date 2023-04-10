# 5.3 改进模型拟合
# 为了实现完美的拟合，你必须首先实现过拟合。由于事先并不知道界线在哪里，因此你必须穿过界线才能找到它。
# 即找到界限稳健拟合。
# 在开始处理一个问题时，你的初始目标是构建一个具有一定泛化能力并且能够过拟合的模型。
# 得到这样一个模型之后，你的重点将是通过降低过拟合来提高泛化能力。
# 在这一阶段，你会遇到以下 3 种常见问题。
#  训练不开始：训练损失不随着时间的推移而减小。
#  训练开始得很好，但模型没有真正泛化：模型无法超越基于常识的基准。
#  训练损失和验证损失都随着时间的推移而减小，模型可以超越基准，
# 但似乎无法过拟合，这表示模型仍然处于欠拟合状态。

# 5.3.1 调节关键的梯度下降参数：
# 损失保持不变。
# 出现这种情况时，问题总是出在梯度下降过程的配置：优化器、模型权重初始值的分布、学习率或批量大小。
# 所有这些参数都是相互依赖的，因此，保持其他参数不变，调节***学习率和批量大小***通常就足够了。

# 我们来看一个具体的例子：训练第 2 章的 MNIST 模型，但选取一个过大的学习率（取值为 1）。
import keras
from keras import layers
from keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer=tf.optimizers.RMSprop(1.),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history_lr1 = model.fit(train_images, train_labels,
                        epochs=40,
                        batch_size=128,
                        validation_split=0.2)

model = keras.Sequential([
    layers.Dense(512, activation="relu"),
    layers.Dense(10, activation="softmax")
])
model.compile(optimizer=tf.optimizers.RMSprop(1e-2),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history_lr1e2 = model.fit(train_images, train_labels,
                          epochs=40,
                          batch_size=128,
                          validation_split=0.2)

acc_lr1 = history_lr1.history["val_accuracy"]
acc_lr1e2 = history_lr1e2.history["val_accuracy"]
epochs = range(1, 41)
plt.plot(epochs, acc_lr1, "b-",
         label="Validation accuracy lr = 1")
plt.plot(epochs, acc_lr1e2, "b--",
         label="Validation accuracy lr = 1e-2")
plt.title("Effect of learning rate on validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.legend()
plt.show()

# 解决方法：
#  降低或提高学习率。学习率过大，可能会导致权重更新大大超出正常拟合的范围，
# 就像前面的例子一样。学习率过小，则可能导致训练过于缓慢，以至于几乎停止。
#  增加批量大小。如果批量包含更多样本，那么梯度将包含更多信息且噪声更少（方差更小）。
# 最终，你会找到一个能够开始训练的配置。
