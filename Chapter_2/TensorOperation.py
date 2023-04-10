import keras.layers
import numpy as np

keras.layers.Dense(512, activation="relu")


# output = keras.relu(dot(input, W) + b)
# W 是一个矩阵，b 是一个向量，二者都是该层的属性

# 逐元素（element-wise）运算：下列代码是对逐元素 relu 运算的简单实现。
def naive_relu(x):
    assert len(x.shape) == 2  # x是一个二阶Numpy向量
    x = x.copy()
    for i in range(x.shape[0]):  # 避免覆盖输入张量
        for j in range(x.shape[1]):
            x[i, j] = max(x[i, j], 0)
    return x


# 对逐元素加法
def naive_add(x, y):
    assert len(x.shape) == 2
    assert x.shape == y.shape
    x = x.copy()
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[i, j]
    return x


import time

x = np.random.random((20, 100))
y = np.random.random((20, 100))
z = x + y
z = np.maximum(z, 0.)
# print(z)
t0 = time.time()
for _ in range(1000):
    z = x + y
    z = np.maximum(z, 0.)
print("Took: {0:.2f} s".format(time.time() - t0))
# 只需要 0.02 秒。

# 与之相对，前面手动编写的简单naive_add实现耗时长达 2.45 秒。
t0 = time.time()
for _ in range(1000):
    z = naive_add(x, y)
    z = naive_relu(z)
print("Took: {0:.2f} s".format(time.time() - t0))
# 同样，在 GPU 上运行 TensorFlow 代码，逐元素运算都是通过完全向量化的 CUDA 来完成的，可以最大限度地利用高度并行的 GPU 芯片架构。

# 2.3.2 广播——BroadCast
# (1) 向较小张量添加轴 [ 叫作广播轴（broadcast axis）]，使其 ndim 与较大张量相同。
# (2) 将较小张量沿着新轴重复，使其形状与较大张量相同。
X = np.random.random((32, 10))
y = np.random.random((10,))
# print("initial: ",y)
# 向 y 添加第 1 个轴（空的），加了一阶，或者说加了一个中括号
y = np.expand_dims(y, axis=0)
# print("add one: ",y)
for i in range(0, 32):
    Y = np.concatenate([y] * 32, axis=0)
    # print("the ",i," time: ",Y)

x = np.random.random((64, 3, 32, 10))
y = np.random.random((32, 10))
z = np.maximum(x, y)


# 事实上不同形状张量加法的简单运算：
def naive_add_matrix_and_vector(x, y):
    assert len(x.shape) == 2  # x 是一个 2 阶 NumPy 张量
    assert len(y.shape) == 1  # y 是一个 NumPy 向
    assert x.shape[1] == y.shape[0]
    x = x.copy()  # 避免覆盖输入张量
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x[i, j] += y[j]
    return x


# 分解了看内积——点积运算：
def naive_vector_dot(x, y):
    assert len(x.shape) == 1
    assert len(y.shape) == 1  # x 和 y 都是 NumPy 向量
    assert x.shape[0] == y.shape[0]
    z = 0.
    for i in range(x.shape[0]):
        z += x[i] * y[i]
    return z


# 还可以对一个矩阵 x 和一个向量 y 做点积运算，其返回值是一个向量，其中每个元素是y 和 x 每一行的点积。实现过程如下。
def naive_matrix_vector_dot(x, y):
    assert len(x.shape) == 2  # x 是一个 NumPy 矩阵
    assert len(y.shape) == 1  # y 是一个 NumPy 向量
    assert x.shape[1] == y.shape[0]  # x 的第 1 维与 y 的第 0 维必须大小相同！
    z = np.zeros(x.shape[0])
    for i in range(x.shape[0]):  # 这个运算返回一个零向量，其形状与 x.shape[0] 相同，元素全部是0
        for j in range(x.shape[1]):
            z[i] += x[i, j] * y[j]
    return z


# ！！千万注意：点积不符合交换律，dot(x, y) 不等于 dot(y, x)。

# 点积可以推广到具有任意轴数的张量。
def naive_matrix_dot(x, y):
    assert len(x.shape) == 2
    assert len(y.shape) == 2  # x 和 y 都是 NumPy 矩阵
    assert x.shape[1] == y.shape[0]  # x 的第 1 维与 y 的第 0 维必须大小相同！
    z = np.zeros((x.shape[0], y.shape[1]))  # 这个运算返回一个特定形状的零矩阵
    for i in range(x.shape[0]):
        for j in range(y.shape[1]):
            row_x = x[i, :]  # 遍历 x 的所有行……
            column_y = y[:, j]  # ……然后遍历 y 的所有列
            z[i, j] = naive_vector_dot(row_x, column_y)
    return z


# 2.3.4 张量变形：
# 在预处理数据时用到了这种运算。
# train_images = train_images.reshape((60000, 28 * 28))
x = np.array([[0., 1.],
              [2., 3.],
              [4., 5.]])
print(x.shape)
x = x.reshape((6, 1))
print(x)
x = x.reshape((2, 3))
print(x)
# 转置（transpose）
x = np.zeros((300, 20))
print("转置前：", x.shape)
x = np.transpose(x)
print("转置后：", x.shape)

# 2.3.5 张量运算的几何解释
# 1、平移（translation）。容易理解，在一个点上加一个向量，会使这个点在某个方向上移动一段距离。
# 2、旋转（rotation）。要将一个二维向量逆时针旋转 theta 角，可以通过与一个 2×2 矩阵做点积运算来实现。
# 这个矩阵为 R = [[cos(theta), -sin(theta)], [sin(theta), cos(theta)]]。
# 3、缩放（scaling）。要将图像在垂直方向和水平方向进行缩放，可以通过与一个 2×2 矩阵做点积运算来实现。
# 这个矩阵为 S=[[horizontal_factor, 0], [0, vertical_factor]]。（horizontal_factor控制x轴变换-hf，x_new = x0 * hf, 同理y_new = y0 * vf）
# 4、线性变换（linear transform）。与任意矩阵做点积运算，都可以实现一次线性变换。
# 注意，前面所说的缩放和旋转，都属于线性变换。
# 5、仿射变换（affine transform）。仿射变换是一次线性变换（通过与某个矩阵做点积运算来实现）与一次平移（通过向量加法来实现）的组合。
# 你可能已经发现，这正是 Dense 层所实现的 y = W • x + b 运算！一个没有激活函数的 Dense 层就是一个仿射层。
# 实现效果就是只保留x_new>0 and y_new>0 （第一象限）的部分


# 机器学习的目的：为高维空间中复杂、高度折叠的数据流形（manifold）找到简洁的表示