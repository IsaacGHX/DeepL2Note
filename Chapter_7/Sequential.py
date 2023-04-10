from HEAD import *

model = keras.Sequential([
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

model = keras.Sequential()
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(10, activation="softmax"))
# 只有在第一次调用层时，层才会被构建（创建层权重）。这是因为层权重的形状取决于输入形状，
# 只有知道输入形状之后才能创建权重。

# model.weights
# ValueError: Weights for model sequential_1 have not yet been created.
model.build(input_shape=(None, 3))
# model.weights
# [<tf.Variable "dense_2/kernel:0" shape=(3, 64) dtype=float32, ... >,
#  <tf.Variable "dense_2/bias:0" shape=(64,) dtype=float32, ... >
#  <tf.Variable "dense_3/kernel:0" shape=(64, 10) dtype=float32, ... >,
#  <tf.Variable "dense_3/bias:0" shape=(10,) dtype=float32, ... >]

model.summary()

# 可以看到，这个模型刚好被命名为 sequential_1。
# 你可以对 Keras 中的所有对象命名，包括每个模型和每一层。

model = keras.Sequential(name="my_example_model")
model.add(layers.Dense(64, activation="relu", name="my_first_layer"))
model.add(layers.Dense(10, activation="softmax", name="my_last_layer"))
model.build((None, 3))
model.summary()
# Model: "my_example_model"
# 逐步构建序贯模型时，每添加一层就打印出当前模型的概述信息，这是非常有用的。
# 但在模型构建完成之前是无法打印概述信息的。有一种方法可以实时构建序贯模型：
# 只需提前声明模型的输入形状。你可以通过 Input 类来做到这一点。

model = keras.Sequential()
model.add(keras.Input(shape=(3,)))
# 利用 Input 声明输入形状。请注意，shape 参数应该是单个样本的形状，而不是批量的形状
model.add(layers.Dense(64, activation="relu"))

model.summary()
# Model: "sequential_2"
# 这是一种常用的调试工作流程，用于处理那些对输入进行复杂变换的层

# 序贯模型适用范围非常有限：它只能表示具有单一输入和单一输出的模型，按顺序逐层进行处理。
# 我们在实践中经常会遇到其他类型的模型，比如多输入模型（如图像及其元数据）、
# 多输出模型（预测数据的不同方面）或具有非线性拓扑结构的模型。


