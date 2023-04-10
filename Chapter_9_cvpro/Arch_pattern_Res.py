# 9.3 现代卷积神经网络架构模式

# 哪些层，如何配置、如何连接

# 定义了模型的假设空间，即梯度下降可以搜索的函数空间，其参数为模型权重。

# 良好的模型架构可以加速学习过程，让模型可以有效利用训练数据，并降低对大型数据集的需求。
# 一个良好的模型架构可以减小搜索空间，或者更容易收敛到搜索空间的良好位置。
# 就像特征工程和数据收集一样，模型架构就是为了能够利用梯度下降更轻松地解决问题。
# 请记住，梯度下降是非常呆板的搜索过程，所以它需要尽可能获得帮助。

# 基本的卷积神经网络架构最佳实践，
# 特别是
# 残差连接（residual connection）、
# 批量规范化（batch normalization）
# 可分离卷积（separable convolution）。

# 9.3.1 模块化、层次结构和复用
# 将无定形的复杂内容构建为模块（module）
# 将模块组织成层次结构（hierarchy），并多次复用（reuse）相同的模块
# 滤波器的数量随着模型深度的增加而增大，而特征图的尺寸则相应减小。
# 一般来说，尺寸较小的层的深度堆叠比尺寸较大的层的浅层堆叠性能更好。
# 然而，由于梯度消失问题，层的堆叠深度是有限的。

# 9.3.2 残差连接
# 由于每一步函数变换都会引入噪声，函数链如果过长，噪声会掩盖梯度信息
# 只需将一层或一个层块的输入添加到它的输出中，
# 残差连接的作用是提供信息捷径，围绕着有损的或有噪声的层块（如包含 relu 激活或 dropout 层的层块），
# 让来自较早的层的误差梯度信息能够通过深度网络以无噪声的方式传播。
# 将输入与层块输出相加，意味着输出与输入应该具有相同的形状。
# 如果层块中有包含更多滤波器的卷积层或最大汇聚层，那么二者的形状就不相同。
# 在这种情况下，可以使用一个没有激活的 1×1 Conv2D 层，将残差线性投影为输出形状，

# 我们通常会在目标层块的卷积层中使用 padding="same"，以避免由于填充导致的空间下采样。
# 此外，我们还会在残差投影中使用步幅，以匹配由于最大汇聚层导致的下采样。

from HEAD import *


# normal concat
def layer_basicRES(layers):
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    residual = x
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)  # 用 padding="same"，以避免由于填充导致的下采样
    residual = layers.Conv2D(64, 1)(residual)  # use 1*1 but same shape as Conv2D's filter num
    x = layers.add([x, residual])
    return x


# concat with Maxpool
def layer_with_Maxpool(layers):
    inputs = keras.Input(shape=(32, 32, 3))
    x = layers.Conv2D(32, 3, activation="relu")(inputs)
    residual = x
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.MaxPooling2D(2, padding="same")(x)  # here still using the "same" padding same reason as former
    residual = layers.Conv2D(64, 1, strides=2)(residual)  # 在残差投影中使用 strides=2，以匹配最大汇聚层导致的下采样
    x = layers.add([x, residual])
    return x


inputs = keras.Input(shape=(32, 32, 3))
x = layers.Rescaling(1. / 255)(inputs)


def residual_block(x, filters, pooling=False):
    residual = x
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(filters, 3, activation="relu", padding="same")(x)
    if pooling:
        x = layers.MaxPooling2D(2, padding="same")(x)
        residual = layers.Conv2D(filters, 1, strides=2)(residual)  # if maxpool watching the strides
    elif filters != residual.shape[-1]:
        residual = layers.Conv2D(filters, 1)(residual)  # if not maxpool and there's not match using 1step
    x = layers.add([x, residual])
    return x


x = residual_block(x, filters=32, pooling=True)
x = residual_block(x, filters=64, pooling=True)
x = residual_block(x, filters=128, pooling=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs=inputs, outputs=outputs)
model.summary()
