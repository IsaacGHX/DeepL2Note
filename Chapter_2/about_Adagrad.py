# output = relu(dot(input, W) + b)
# W 和 b 是张量，均为该层的属性。
# 它们被称为该层的权重（weight）或可训练参数（trainable parameter），
# 分别对应属性 kernel 和 bias。这些权重包含模型从训练数据中学到的信息。
# 一开始，这些权重矩阵取较小的随机值，这一步叫作随机初始化（random initialization）。W 和 b 都是随机的。
# 但这是一个起点。下一步则是根据反馈信号逐步调节这些权重。这个逐步调节的过程叫作训练（training），也就是机器学习中的“学习”过程。
# 一个训练循环（training loop）内，重复下列步骤，直到损失值变得足够小。
# (1) 抽取训练样本 x 和对应目标 y_true 组成的一个数据批量。
# (2) 在 x 上运行模型［这一步叫作前向传播（forward pass）］，得到预测值 y_pred。
# (3) 计算模型在这批数据上的损失值，用于衡量 y_pred 和 y_true 之间的差距。
# (4) 更新模型的所有权重，以略微减小模型在这批数据上的损失值。

# # 可微函数与梯度的关系，只有可微的函数才能求梯度
# loss_value = f(W) 在 W0 附近最陡上升方向的张量，也表示这一上升方向的斜率。
# grad(loss_value, W0) 可以看作表示 loss_value = f(W) 在 W0 附近最陡上升方向的张量
# 只是 W0 附近曲率的近似值，所以不能离W0 太远。

# 2.4.3 随机梯度下降
# 找到所有导数为 0 的点，然后比较函数在其中哪个点的取值最小。

# (1) 抽取训练样本 x 和对应目标 y_true 组成的一个数据批量。
# (2) 在 x 上运行模型，得到预测值 y_pred。这一步叫作前向传播。
# (3) 计算模型在这批数据上的损失值，用于衡量 y_pred 和 y_true 之间的差距。
# (4) 计算损失相对于模型参数的梯度。这一步叫作反向传播（backward pass）。
# (5) 将参数沿着梯度的反方向移动一小步，比如 W -= learning_rate * gradient，从而使这批数据上的损失值减小一些。
# 学习率（learning_rate）是一个调节梯度下降“速度”的标量因子。
# 上述为小批量随机梯度下降（mini-batch stochastic gradient descent，简称小批量 SGD）。
# 随机（stochastic）是指每批数据都是随机抽取的（stochastic 在科学上是random 的同义词 ）。

# learning_rate= 0.05
# past_velocity = 0.
# momentum = 0.1
# while loss > 0.01:
#      w, loss, gradient = get_current_parameters()
#      velocity = past_velocity * momentum - learning_rate * gradient
#      w = w + momentum * velocity - learning_rate * gradient
#      past_velocity = velocity
#      update_parameter(w)


# 2.4.4 链式求导：反向传播算法（backpropagation algorithm）:
# 1. 链式法则：
# 的复合函数 fg：fg(x) == f(g(x))#
# def fg(x):
#     x1 = g(x)
#     y = f(x1)
#     return y
# 链式法则规定：grad(y, x) == grad(y, x1) * grad(x1, x)。

# def fghj(x):
#     x1 = j(x)
#     x2 = h(x1)
#     x3 = g(x2)
#     y = f(x3)
#     return y

# grad(y, x) == (grad(y, x3) * grad(x3, x2) * grad(x2, x1) * grad(x1, x))


import tensorflow as tf

x = tf.Variable(0.)
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y, x)

# GradientTape 也可用于张量运算。
x = tf.Variable(tf.zeros((2, 2)))  # 将 Variable 初始化为形状为 (2, 2) 的零张量
with tf.GradientTape() as tape:
    y = 2 * x + 3
grad_of_y_wrt_x = tape.gradient(y,
                                x)  # grad_of_y_wrt_x 是一个形状为 (2, 2) 的张量（形状与 x 相同），表示 y = 2 * a + 3在 x = [[0, 0], [0, 0]] 附近的曲率

# 它还适用于变量列表。
W = tf.Variable(tf.random.uniform((2, 2)))
b = tf.Variable(tf.zeros((2,)))
x = tf.random.uniform((2, 2))
with tf.GradientTape() as tape:
    y = tf.matmul(x, W) + b  # 在 TensorFlow 中，matmul 是指点积
grad_of_y_wrt_W_and_b = tape.gradient(y, [W, b])  # grad_of_y_wrt_W_and_b 是由两个张量组成的列表，这两个张量的形状分别与 W 和 b 相同
