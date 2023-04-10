import matplotlib.pyplot as plt

from HEAD import *


# 创建模型（我们将其包装为一个单独的函数，以便后续复用）

def get_mnist_model():
    inputs = keras.Input(shape=(28 * 28,))
    features = layers.Dense(512, activation="relu")(inputs)
    features = layers.Dropout(0.5)(features)
    outputs = layers.Dense(10, activation="softmax")(features)
    model = keras.Model(inputs, outputs)
    return model


# 加载数据，保留一部分数据用于验证
(images, labels), (test_images, test_labels) = mnist.load_data()

images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255  # 保留一部分作为测试数据
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
# 编译模型，指定模型的优化器、需要最小化的损失函数和需要监控的指标
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# 使用 fit() 训练模型，可以选择提供验证数据来监控模型在前所未见的数据上的性能
model.fit(train_images, train_labels,
          epochs=3,
          validation_data=(val_images, val_labels))
# 使用 evaluate() 计算模型在新数据上的损失和指标
test_metrics = model.evaluate(test_images, test_labels)
# 使用 predict() 计算模型在新数据上的分类概率
predictions = model.predict(test_images)


# 要想自定义这个简单的工作流程，可以采用以下方法：
#  编写自定义指标；
#  向 fit() 方法传入回调函数，以便在训练过程中的特定时间点采取行动。

# 7.3.1 编写自定义指标
# 指标是衡量模型性能的关键，尤其是衡量模型在训练数据上的性能与在测试数据上的性能之间的差异。
# 常用的分类指标和回归指标内置于 keras.metrics 模块中。大多数情况下，你会使用这些指标。
# 但如果想做一些不寻常的工作，你需要能够编写自定义指标。这很简单！
# Keras 指标是 keras.metrics.Metric 类的子类。
# 与层相同的是，指标具有一个存储在TensorFlow 变量中的内部状态。
# 与层不同的是，这些变量无法通过反向传播进行更新，所以你必须自己编写状态更新逻辑。
# 这一逻辑由 update_state() 方法实现。举个例子，用于衡量均方根误差（RMSE）

class RootMeanSquaredError(keras.metrics.Metric):  # 将 Metric 类子类化
    # 在构造函数中定义状态变量。与层一样， 你可以访问add_weight()方法

    def __init__(self, name="rmse", **kwargs):
        super().__init__(name=name, **kwargs)
        self.mse_sum = self.add_weight(name="mse_sum", initializer="zeros")
        self.total_samples = self.add_weight(
            name="total_samples", initializer="zeros", dtype="int32")

    # 在update_state()中实现状态更新逻辑。y_true参数是一个数据批量对应的目标（或标签），
    # y_pred则表示相应的模型预测值。你可以忽略sample_weight参数，这里不会用到

    def update_state(self, y_true, y_pred, sample_weight=None):
        # 为了匹配MNIST模型，我们需要分类预测值与整数标签
        y_true = tf.one_hot(y_true, depth=tf.shape(y_pred)[1])
        mse = tf.reduce_sum(tf.square(y_true - y_pred))
        self.mse_sum.assign_add(mse)
        num_samples = tf.shape(y_pred)[0]
        self.total_samples.assign_add(num_samples)

    # 我们可以使用 result() 方法返回指标的当前值。
    def result(self):
        return tf.sqrt(self.mse_sum / tf.cast(self.total_samples, tf.float32))

        # 此外，你还需要提供一种方法来重置指标状态，而无须将其重新实例化。如此一来，
        # 相同的指标对象可以在不同的训练轮次中使用，或者在训练和评估中使用。
        # 这可以用 reset_state()方法来实现。

    def reset_state(self):
        self.mse_sum.assign(0.)
        self.total_samples.assign(0)


# 自定义指标的用法与内置指标相同。下面来测试一下我们的自定义指标。
model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy", RootMeanSquaredError()])
model.fit(train_images, train_labels,
          epochs=3,
          validation_data=(val_images, val_labels))
test_metrics = model.evaluate(test_images, test_labels)

# 7.3.2 使用回调函数
# Keras 的回调函数（callback）API 可以让model.fit() 的调用从纸飞机变为自主飞行的无人机，
# 使其能够观察自身状态并不断采取行动。
# 回调函数是一个对象（实现了特定方法的类实例），它在调用 fit() 时被传入模型，
# 并在训练过程中的不同时间点被模型调用。回调函数可以访问关于模型状态与模型性能的所有可用数据，
# 还可以采取以下行动：中断训练、保存模型、加载一组不同的权重或者改变模型状态。
# 回调函数的一些用法示例如下。
#  模型检查点（model checkpointing）：在训练过程中的不同时间点保存模型的当前状态。
#  提前终止（early stopping）：如果验证损失不再改善，则中断训练（当然，同时保存在训练过程中的最佳模型）。
#  在训练过程中动态调节某些参数值：比如调节优化器的学习率。
#  在训练过程中记录训练指标和验证指标，或者将模型学到的表示可视化（这些表示在不断更新）：
# fit() 进度条实际上就是一个回调函数。

# keras.callbacks 模块包含许多内置的回调函数，下面列出了其中一些，还有很多没有列出来。
# 通过keras.callbacks. 来调用
# ModelCheckpoint, EarlyStopping, LearningRateScheduler,
# ReduceLROnPlateau, CSVLogger


# 回调函数 EarlyStopping 和 ModelCheckpoint
# 训练模型时，很多事情一开始无法预测，尤其是你无法预测需要多少轮才能达到最佳验证损失。
# 一种更好的处理方法是，发现验证损失不再改善时，停止训练。这可以通过EarlyStopping 回调函数来实现。
# 如果监控的目标指标在设定的轮数内不再改善，那么可以用 EarlyStopping 回调函数中断训练。
# 比如，这个回调函数可以在刚开始过拟合时就立即中断训练，从而避免用更少的轮数重新训练模型。
# 这个回调函数通常与 ModelCheckpoint 结合使用，后者可以在训练过程中不断保存模型
# （你也可以选择只保存当前最佳模型，即每轮结束后具有最佳性能的模型）。

# 我的理解：即时止损在从而使得效率快增

# 通过 fit() 的 callbacks 参数将回调函数传入模型中，
# 该参数接收一个回调函数列表，可以传入任意数量的回调函数
callbacks_list = [
    keras.callbacks.EarlyStopping(  # 如果不再改善，则中断训练
        monitor="val_accuracy",  # 监控模型的验证精度
        patience=2,  # 如果精度在两轮内都不再改善，则中断训练
    ),
    # 在每轮过后保存当前权重
    keras.callbacks.ModelCheckpoint(
        filepath="checkpoint_path.keras",  # 文件的保存路径
        monitor="val_loss",
        # 这两个参数的含义是，只有当 val_loss改善时，才会覆盖模型文件，
        # 这样就可以一直保存训练过程中的最佳模型
        save_best_only=True,
    )
]
model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])  # 监控精度，它应该是模型指标的一部分
model.fit(train_images, train_labels,
          epochs=10,
          callbacks=callbacks_list,
          validation_data=(val_images, val_labels))
# 因为回调函数要监控验证损失和验证指标，所以在调用 fit() 时需要传入validation_data（验证数据）

# 注意，你也可以在训练完成后手动保存模型，只需调用
# model.save('my_checkpoint_path')
model = keras.models.load_model("checkpoint_path.keras")


# 7.3.3 编写自定义回调函数
# 如果想在训练过程中采取特定行动，而这些行动又没有包含在内置回调函数中，那么你可以编写自定义回调函数。
# 回调函数的实现方式是将 keras.callbacks.Callback 类子类化。
# 然后，你可以实现下列方法（从名称中即可看出这些方法的作用），它们在训练过程中的不同时间点被调用。

# on_epoch_begin(epoch, logs)  # 每轮开始的时候调用
# on_epoch_end(epoch, logs)  # 每轮结束的时候调用
# on_batch_begin(batch, logs)  # 处理每个batch之前调用
# on_batch_end(batch, logs)  # 处理每个batch之后调用
# on_train_begin(logs)  # 训练开始的时候调用
# on_train_end(logs)  # 训练结束的时候调用
# 调用这些方法时，都会用到参数 logs。这个参数是一个字典，
# 它包含前一个批量、前一个轮次或前一次训练的信息，比如训练指标和验证指标等。
# on_epoch_* 方法和 on_batch_* 方法还将轮次索引或批量索引作为第一个参数（整数）。

# 给出了一个简单示例，它在训练过程中保存每个批量损失值组成的列表，
# 还在每轮结束时保存这些损失值组成的图。

class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs):
        self.per_batch_losses = []

    def on_batch_end(self, batch, logs):
        self.per_batch_losses.append(logs.get("loss"))

    def on_epoch_end(self, epoch, logs):
        plt.clf()
        plt.plot(range(len(self.per_batch_losses)), self.per_batch_losses,
                 label="Training loss for each batch")
        plt.xlabel(f"Batch (epoch {epoch})")
        plt.ylabel("Loss")
        plt.legend()
        plt.savefig(f"plot_at_epoch_{epoch}")
        self.per_batch_losses = []


model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
model.fit(train_images, train_labels,
          epochs=10,
          callbacks=[LossHistory()],
          validation_data=(val_images, val_labels))
plt.show()

# 这里的图全部在这一层中显示了
