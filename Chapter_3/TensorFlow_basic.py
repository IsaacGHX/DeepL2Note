# 3.5.1常数张量与变量
import tensorflow as tf
import numpy as np

# 全为0/ 1 的张量
x = tf.ones(shape=(2, 1))  # np.实现的也是相同的效果
y = tf.zeros(shape=(2, 1))  # np.实现的也是相同的效果
# print(x)
# print(y)

# 随机张量
x = tf.random.normal(shape=(3, 1), mean=0., stddev=1.)
# 从均值为 0、标准差为 1 的正态分布中抽取的随机张量，等同于 np.random.normal(size=(3, 1), loc=0., scale=1.)
# 考点，正态分布
y = tf.random.uniform(shape=(3, 1), minval=0., maxval=1.)
# 从 0 和 1 之间的均匀分布中抽取的随机张量，等同于 np.random.uniform(size=(3, 1), low=0., high=1.)
# 考点：均匀分布
# print(x)
# print(y)

# TensorFlow 张量是不可赋值的，它是常量！！！

# 创建一个变量张量
v = tf.Variable(initial_value=tf.random.normal(shape=(3, 1)))
# print(v)

# 变量张量子集赋值
v[0, 0].assign(3.)
# print(v)
# v.assign(3.)  #NO!  只能对于单个的元素进行加减与赋值
# print(v)
# assign_add() 和 assign_sub()相当于+=，-=

# 使用assign_add()
v.assign_add(tf.ones((3, 1)))  # 给所有的元素加一个1
# print(v)

# 3.5.2用Tensorflow进行数学运算

a = tf.ones((2, 2))
b = tf.square(a)  # 求平方
c = tf.sqrt(a)  # 求平方根
d = b + c  # 两个张量相加
e = tf.matmul(a, b)  # 两个张量的积（点积）
e *= d  # 两个张量逐项元素相乘，乘数可以是一个单个的数字
# 以上的任意运算都是可以随时打印的
# print(a)
# print(b)
# print(c)
# print(d)
# print(e)

# 3.5.3GradientTape API
input_var = tf.Variable(initial_value=3.)  # 单个数字
with tf.GradientTape() as tape:
    result = tf.square(input_var)  # (9.)
gradient = tape.gradient(result, input_var)
# print(gradient)   #(6.)

# 对于常数张量的梯度运算
input_const = tf.constant(3.)
with tf.GradientTape() as tape:
    tape.watch(input_const)  # 对于常数张量的监视，通过tape.watch手动标记
    result = tf.square(input_const)
gradient = tape.gradient(result, input_const)
# print(gradient)   #(6.)


# if 测量一个垂直下落的苹果的位置随时间的变化，并且发现它满足:
# position(time) =4.9 * time ** 2
time = tf.Variable(0.)
with tf.GradientTape() as outer_tape:
    with tf.GradientTape() as inner_tape:
        position = 4.9 * time ** 2  # 函数式 position = f(time)
    speed = inner_tape.gradient(position, time)  # 函数式 speed = f ' (time)
    # print(speed)  # 不是常数输出就是0
acceleration = outer_tape.gradient(speed, time)  # 函数式 speed = f '' (time)
# print(acceleration)  # 是常数输出 (9.8)

# 3.5.4 端到端：TensorFlow 编写线性分类器
num_samples_per_class = 1000
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],  # 均值
    cov=[[1, 0.5], [0.5, 1]],  # 协方差
    size=num_samples_per_class)
# 生成第一个类别的点：1000 个二维随机点。
# 协方差矩阵为 [[1, 0.5], [0.5, 1]]，对应于一个从左下方到右上方的椭圆形点云
# 考点：协方差
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class)
# 协方差相同，均值不同

targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))  # 生成对应标签

inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)  # 类堆叠



# 线性分类器
input_dim = 2  # 输入是二维点
output_dim = 1  # 每个样本的输出预测值是一个分数值
# （若分类器预测样本属于类别 0，那么这个分数值会接近 0；若预测样本属于类别 1，那么这个分数值会接近 1）
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))  # 权重的输入输出形状的取决
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))


# 向前传播函数，说白了就是进行一层的权重偏置运算
def model(inputs):
    return tf.matmul(inputs, W) + b


# 均方误差损失函数
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)  # 与 targets 和 predictions具有相同形状的张量,每个元素对应的是targets其中每个预测的损失
    return tf.reduce_mean(per_sample_losses)  # 损失值平均为一个标量损失值


# 训练步骤！
learning_rate = 0.1


def training_step(inputs, targets):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)  # 均方差损失
        # print(loss)
    grad_loss_wrt_W, grad_loss_wrt_b = tape.gradient(loss, [W, b])  # 分别计算，W, b分别对于loss函数的求导
    W.assign_sub(grad_loss_wrt_W * learning_rate)  # 对于权重参数进行调整
    b.assign_sub(grad_loss_wrt_b * learning_rate)  # 对于偏置参数进行调整
    return loss


if __name__ == '__main__':
    # 显示图像
    import matplotlib.pyplot as plt

    plt.scatter(inputs[:, 0], inputs[:, 1], c=targets[:, 0])
    plt.show()

    # 批量训练循环
    print(inputs)
    print(targets)
    for step in range(40):
        loss = training_step(inputs, targets)
        print(f"Loss at step {step}: {loss:.4f}")
    predictions = model(inputs)
    plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
    plt.show()
    x = np.linspace(-1, 4, 100)
    y = - W[0] / W[1] * x + (0.5 - b) / W[1]
    plt.plot(x, y, "-r")
    plt.scatter(inputs[:, 0], inputs[:, 1], c=predictions[:, 0] > 0.5)
