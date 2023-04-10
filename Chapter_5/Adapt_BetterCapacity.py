# 5.3.3 提高模型容量
# 如果成功得到了一个能够拟合的模型，验证指标正在下降，且模型似乎具有一定的泛化能力。
# 接下来，你需要让模型过拟合。
import matplotlib.pyplot as plt

from HEAD import *

(train_images, train_labels), _ = mnist.load_data()
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255

model = keras.Sequential([layers.Dense(10, activation="softmax")])
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history_small_model = model.fit(
    train_images, train_labels,
    epochs=20,
    batch_size=128,
    validation_split=0.2)
val_loss = history_small_model.history["val_loss"]

# 验证损失达到了 0.26，然后就保持不变。你可以拟合模型，但无法实现过拟合。
# 在你的职业生涯中，你可能会经常遇到类似的曲线。
# 请记住，任何情况下应该都可以实现过拟合。**

# 如果无法实现过拟合，可能是因为模型的表示能力（representational power）存在问题：
# 你需要一个容量（capacity）更大的模型，也就是一个能够存储更多信息的模型。
# 若要提高模型的表示能力，你可以添加更多的层、使用更大的层（拥有更多参数的层），
# 或者使用更适合当前问题的层类型（也就是更好的架构预设）

model = keras.Sequential([  # 这里用了更加大的且深入的模型来进行层设计
    layers.Dense(96, activation="relu"),
    layers.Dense(96, activation="relu"),
    layers.Dense(10, activation="softmax"),
])
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
history_large_model = model.fit(
    train_images, train_labels,
    epochs=20,
    batch_size=128,
    validation_split=0.2)
large_val_loss = history_large_model.history["val_loss"]

epochs = range(1, 21)
plt.plot(epochs, val_loss, "b--",
         label="small Validation loss")
plt.plot(epochs, large_val_loss, "b-",
         label="big Validation loss")
plt.title("Effect of insufficient model capacity on validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()
