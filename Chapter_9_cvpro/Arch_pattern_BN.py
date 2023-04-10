# 9.3.3 批量规范化
# 将数据减去均值，使其中心为 0，然后除以标准差，使其标准差为 1。
# 将前一层的激活放在批量规范化层之后

# 不应如此使用批量规范化
# x = layers.Conv2D(32, 3, activation="relu")(x)
# x = layers.BatchNormalization()(x)
"""can prevent the saturation of the activation function and
    improve the performance of the network.
"""
# 如何使用批量规范化：将激活放在批量规范化层之后
# x = layers.Conv2D(32, 3, use_bias=False)(x)
# x = layers.BatchNormalization()(x)
# x = layers.Activation("relu")(x)

""" it allows the batch normalization layer to adjust the scale and shift of the activation function, 
which can improve the representational power of the network.
"""
