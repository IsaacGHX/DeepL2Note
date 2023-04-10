# 7.4.4 利用 tf.function 加快运行速度
# 你可能已经注意到，尽管实现了基本相同的逻辑，但自定义循环的运行速度比内置的fit() 和 evaluate() 要慢很多。
# 这是因为默认情况下，TensorFlow 代码是逐行急切执行的，就像 NumPy 代码或常规 Python 代码一样。
# 急切执行让调试代码变得更容易，但从性能的角度来看，它远非最佳。
# 更高效的做法是，将 TensorFlow 代码编译成计算图，对该计算图进行全局优化，这是逐行解释代码所无法实现的。
# 这样做的语法非常简单：对于需要在执行前进行编译的函数，只需添加 @tf.function
from HEAD import *
from Chapter_7.AssessTrain.SelfDefined_AssessNTrain import *


@tf.function
def test_step(inputs, targets):
    predictions = model(inputs, training=False)
    loss = loss_fn(targets, predictions)
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs["val_" + metric.name] = metric.result()
    loss_tracking_metric.update_state(loss)
    logs["val_loss"] = loss_tracking_metric.result()
    return logs


val_dataset = tf.data.Dataset.from_tensor_slices((val_images, val_labels))
val_dataset = val_dataset.batch(32)
reset_metrics()
for inputs_batch, targets_batch in val_dataset:
    logs = test_step(inputs_batch, targets_batch)
print("Evaluation results:")
for key, value in logs.items():
    print(f"...{key}: {value:.4f}")

# 请记住，调试代码时，最好使用急切执行，不要使用 @ tf.function装饰器。这样做有利于跟踪错误。
# 一旦代码可以运行，并且你想加快运行速度，就可以将 @ tf.function
# 装饰器添加到训练步骤和评估步骤中，或者添加到其他对性能至关重要的函数中。


# 7.4.5 在 fit() 中使用自定义训练循环
# 在前几节中，我们从头开始编写了自定义训练循环。
# 这样做具有最大的灵活性，但需要编写大量代码，同时无法利用 fit() 提供的许多方便的特性，比如回调函数或对分布式训练的支持。
# 如果想自定义训练算法，但仍想使用 Keras 内置训练逻辑的强大功能，那么要怎么办呢？
# 实际上，在使用 fit() 和从头开始编写训练循环之间存在折中：你可以编写自定义的训练步骤函数，然后让框架完成其余工作。
# 你可以通过覆盖 Model 类的 train_step() 方法来实现这一点。它是 fit() 对每批数据调用的函数。
# 然后，你就可以像平常一样调用 fit()，它将在后台运行你自定义的学习算法。
# 下面看一个简单的例子。
#  创建一个新类，它是 keras.Model 的子类。
#  覆盖 train_step(self, data) 方法，其内容与 7.4.3 节中的几乎相同。
# 它返回一个字典，将指标名称（包括损失）映射到指标当前值。
#  实现 metrics 属性，用于跟踪模型的 Metric 实例。这样模型可以在每轮开始时和调用
# evaluate() 时对模型指标自动调用 reset_state()，你不必手动执行此操作。

loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()
loss_tracker = tf.keras.metrics.Mean(name="loss")


# 这个指标对象用于跟踪训练过程和评估过程中每批数据的损失均值


class CustomModel(keras.Model):
    def train_step(self, data):  # 覆盖train_step的方法：
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            # 这里使用self(inputs,training=True)，而不是model(inputs, training = True)，
            # 因为模型就是类本身。
            loss = loss_fn(targets, predictions)
            gradients = tape.gradient(loss, self.trainable_weights)
            self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
            loss_tracker.update_state(loss)  # 更新损失跟踪器指标，该指标用于跟踪损失均值
            return {"loss": loss_tracker.result()}  # 通过查询损失跟踪器指标返回当前的损失均值

    @property
    def metrics(self):  # 这里应列出需要在不同轮次之间进行重置的指标
        return [loss_tracker]


inputs = tf.keras.Input(shape=(28 * 28,))
features = layers.Dense(512, activation="relu")(inputs)
features = layers.Dropout(0.5)(features)
outputs = layers.Dense(10, activation="softmax")(features)
model = CustomModel(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.RMSprop())
model.fit(train_images, train_labels, epochs=3)


# 有两点需要注意。
#  这种方法并不妨碍你使用函数式 API 构建模型。
# 无论是构建序贯模型、函数式模型还是子类化模型，你都可以这样做。
#  覆盖 train_step() 时，无须使用 @tf.function 装饰器，框架会帮你完成这一步骤。
# 接下来，指标怎么处理？如何通过 compile() 配置损失？在调用 compile() 之后，你可以访问以下内容。
#  self.compiled_loss：传入 compile() 的损失函数。
#  self.compiled_metrics：传入的指标列表的包装器，它允许调用 self.compiled_
# metrics.update_state() 来一次性更新所有指标。
#  self.metrics：传入 compile() 的指标列表。请注意，它还包括一个跟踪损失的指标，
# 类似于之前用 loss_tracking_metric 手动实现的例子。

class CustomModel(keras.Model):
    def train_step(self, data):
        inputs, targets = data
        with tf.GradientTape() as tape:
            predictions = self(inputs, training=True)
            loss = self.compiled_loss(targets, predictions)  # 利用 self.compiled_loss 计算损失
        gradients = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_weights))
        self.compiled_metrics.update_state(targets, predictions)  # 通过 self.compiled_metrics 更新模型指标
        return {m.name: m.result() for m in self.metrics}  # 返回一个字典，将指标名称映射为指标当前值


inputs = keras.Input(shape=(28 * 28,))
features = layers.Dense(512, activation="relu")(inputs)
features = layers.Dropout(0.5)(features)
outputs = layers.Dense(10, activation="softmax")(features)
model = CustomModel(inputs, outputs)
model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss=tf.keras.losses.SparseCategoricalCrossentropy(),
              metrics=[tf.keras.metrics.SparseCategoricalAccuracy()])
model.fit(train_images, train_labels, epochs=3)
