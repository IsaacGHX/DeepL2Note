from HEAD import *

# 加载数据，保留一部分数据用于验证
(images, labels), (test_images, test_labels) = mnist.load_data()

images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255  # 保留一部分作为测试数据
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()

# 7.4 编写自定义的训练循环和评估循环
# fit() 工作流程在易用性和灵活性之间实现了很好的平衡。你在大多数情况下会用到它。
# 然而，即使有了自定义指标、自定义损失函数和自定义回调函数，它也无法实现深度学习研究人员想做的一切事情。
# 毕竟，内置的 fit() 工作流程只针对于**监督学习（supervised learning）。
# 监督学习是指，已知与输入数据相关联的目标（也叫标签或注释），将损失计算为这些目标和模型预测值的函数。
# 然而，并非所有机器学习任务都属于这个类别。
# 还有一些机器学习任务没有明确的目标，
# 比如生成式学习（generative learning）、自监督学习（self-supervised learning，目标是从输入中得到的）和
# 强化学习（reinforcement learning，学习由偶尔的“奖励”驱动，就像训练狗一样）。
# 即使是常规的监督学习，研究人员也可能想添加一些新奇的附加功能，需要用到低阶灵活性。
# 如果你发现内置的 fit() 不够用，那么就需要编写自定义的训练逻辑。

# 典型的训练循环包含以下内容：
# (1) 在梯度带中运行前向传播（计算模型输出），得到当前数据批量的损失值；
# (2) 检索损失相对于模型权重的梯度；
# (3) 更新模型权重，以降低当前数据批量的损失值。
# 这些步骤需要对多个批量重复进行。这基本上就是 fit() 在后台所做的工作。
# 本节将从头开始重新实现 fit()，你将了解编写任意训练算法所需的全部知识。

# 7.4.1 训练与推断
# 在前面的低阶训练循环示例中，
# 步骤 1（前向传播）是通过 predictions = model(inputs)完成的，
# 步骤 2（检索梯度带计算的梯度）是通过 gradients = tape.gradient(loss, model.weights) 完成的。
# 在一般情况下，还有两个细节需要考虑。
# 某些 Keras 层（如 Dropout 层），在训练过程和推断过程（将其用于预测时）中具有不同的行为。
# 这些层的 call() 方法中有一个名为 training 的布尔参数。
# 调用 dropout(inputs, training=True) 将舍弃一些激活单元，而调用 dropout(inputs, training=False) 则不会舍弃。
# 推而广之，函数式模型和序贯模型的 call() 方法也有这个 training 参数。
# 在前向传播中调用 Keras 模型时，一定要记得传入 training=True。
# 也就是说，前向传播应该变成 predictions = model(inputs, training=True)。
# 此外请注意，检索模型权重的梯度时，不应使用 tape.gradients(loss, model.weights)，
# 而应使用 tape.gradients(loss, model.trainable_weights)。

# 层和模型具有以下两种权重。
#  可训练权重（trainable weight）：通过反向传播对这些权重进行更新，
# 以便将模型损失最小化。比如，Dense 层的核和偏置就是可训练权重。
#  不可训练权重（non-trainable weight）：在前向传播过程中，这些权重所在的层对它们进行更新。
# 如果你想自定义一层，用于记录该层处理了多少个批量，那么这一信息需要存储在一个不可训练权重中。
# 每处理一个批量，该层将计数器加 1。

# 在 Keras 的所有内置层中，唯一具有不可训练权重的层是 BatchNormalization 层，第 9章会介绍它。
# BatchNormalization 层需要使用不可训练权重，以便跟踪关于传入数据的均值和标准差的信息，从而实时进行特征规范化。
# 将这两个细节考虑在内，监督学习的训练步骤如下所示。
# def train_step(inputs, targets):
#     with tf.GradientTape() as tape:
#         predictions = model(inputs, training=True)
#     loss = loss_fn(targets, predictions)
#     gradients = tape.gradients(loss, model.trainable_weights)
#     optimizer.apply_gradients(zip(model.trainable_weights, gradients))

# 7.4.2 指标的低阶用法
# 在低阶训练循环中，你可能会用到 Keras 指标（无论是自定义指标还是内置指标）。
# 你已经了解了指标 API：只需对每一个目标和预测值组成的批量调用
# update_state(y_true, y_pred)，然后使用 result() 查询当前指标值。
metric = tf.keras.metrics.SparseCategoricalAccuracy()
targets = [0, 1, 2]
predictions = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
metric.update_state(targets, predictions)
current_result = metric.result()
print(f"result: {current_result:.2f}")

# 你可能还需要跟踪某个标量值（比如模型损失）的均值。
# 这可以通过 keras.metrics.Mean 指标来实现。
values = [0, 1, 2, 3, 4]
mean_tracker = tf.keras.metrics.Mean()
for value in values:
    mean_tracker.update_state(value)
print(f"Mean of values: {mean_tracker.result():.2f}")
# 如果想重置当前结果（在一轮训练开始时或评估开始时），
# 记得使用 metric.reset_state()。


# 7.4.3 完整的训练循环和评估循环
# 我们将前向传播、反向传播和指标跟踪组合成一个类似于 fit() 的训练步骤函数。
# 这个函数接收数据和目标组成的批量，并返回由 fit() 进度条显示的日志。

model = get_mnist_model()
loss_fn = tf.keras.losses.SparseCategoricalCrossentropy()  # 损失函数
optimizer = tf.keras.optimizers.RMSprop()  # 优化器
metrics = [tf.keras.metrics.SparseCategoricalAccuracy()]  # 监控指标
loss_tracking_metric = tf.keras.metrics.Mean()  # 准备Mean指标跟踪器，跟踪损失值


def train_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs, training=True)  # 向前传播
        loss = loss_fn(targets, predictions)  # 传入了training = True
    gradients = tape.gradient(loss, model.trainable_weights)
    optimizer.apply_gradients(zip(gradients, model.trainable_weights))
    # 优化器反向传播
    logs = {}
    for metric in metrics:
        metric.update_state(targets, predictions)
        logs[metric.name] = metric.result()
    loss_tracking_metric.update_state(loss)
    logs["loss"] = loss_tracking_metric.result()  # 跟踪损失值
    return logs  # 返回的是当前指标与损失的值


# 逐步编写训练循环：重置指标
def reset_metrics():
    for metric in metrics:
        metric.reset_state()
    loss_tracking_metric.reset_state()


# 逐步编写训练循环：循环本身
training_dataset = tf.data.Dataset.from_tensor_slices(
    (train_images, train_labels))
training_dataset = training_dataset.batch(32)
epochs = 3
for epoch in range(epochs):
    reset_metrics()
    for inputs_batch, targets_batch in training_dataset:
        logs = train_step(inputs_batch, targets_batch)
    print(f"Results at the end of epoch {epoch}")
    for key, value in logs.items():
        print(f"...{key}: {value:.4f}")


# 逐步编写评估循环
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
