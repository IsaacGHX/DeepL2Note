# 11.4 Transformer 架构
# 11.4.1 理解自注意力
# 类似概念：maxpool, TF-DIF

# 重点：能够获得上下文感知：context-aware
# 相关性分数***，早出现于transformer，属于一种计算效率很高的距离函数
from HEAD import *


def self_attention(input_sequence):
    output = np.zeros(shape=input_sequence.shape)
    for i, pivot_vector in enumerate(input_sequence):  # 对于输入序列的词源迭代
        scores = np.zeros(shape=(len(input_sequence),))
        for j, vector in enumerate(input_sequence):
            scores[j] = np.dot(pivot_vector, vector.T)  # 计算词源之间的卷积
            scores /= np.sqrt(input_sequence.shape[1])
            scores = keras.activations.softmax(scores)  # 利用规范化因子进行缩放，并应用 softmax
            new_pivot_representation = np.zeros(shape=pivot_vector.shape)
            for j, vector in enumerate(input_sequence):
                new_pivot_representation += vector * scores[j]  # l利用注意力分数进行加权，对所有词元进行求和
        output[i] = new_pivot_representation
    return output


# keras 自带的multi head注意力模块
from keras.layers import MultiHeadAttention

num_heads = 4
embed_dim = 256
mha_layer = MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
# outputs = mha_layer(inputs, inputs, inputs)

# 一般的自注意力：查询-键-值模型
# Transformer是一个序列到序列的模型
# 机制如下
# outputs = sum(inputs_C * pairwise_scores(inputs_A, inputs_B))
# 这个表达式的含义是：“对于 inputs（A）中的每个词元，计算该词元与 inputs（B）中每个词元的相关程度，
# 然后利用这些分数对 inputs（C）中的词元进行加权求和。”
# 重要的是，A、B、C 不一定是同一个输入序列。
# 一般情况下，你可以使用 3 个序列，我们分别称其为查询（query）、键（key）和值（value）。
# 这样一来，上述运算的含义就变为：“对于查询中的每个元素，计算该元素与每个键的相关程度，然后利用这些分数对值进行加权求和。”

# 目的是实现：键值匹配
# 多头注意力机制的意义就是要把输入输出全部变成多头，来实现我们的内容

# 如下是一个Transformer enconder
import tensorflow as tf


class TransformerEncoder(layers.Layer):
    def __init__(self, embed_dim, dense_dim, num_heads, **kwargs):
        super().__init__(**kwargs)
        self.embed_dim = embed_dim  # 输入向量元的尺寸
        self.dense_dim = dense_dim  # 内部密集层的尺寸
        self.num_heads = num_heads  # 注意力头的个数
        self.attention = layers.MultiHeadAttention(
            num_heads=num_heads, key_dim=embed_dim)
        self.dense_proj = keras.Sequential(
            [layers.Dense(dense_dim, activation="relu"),
             layers.Dense(embed_dim), ]
        )  # 每个米基层都是先激活，再密集
        self.layernorm_1 = layers.LayerNormalization()
        self.layernorm_2 = layers.LayerNormalization()

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = mask[:, tf.newaxis, :]
            # Embedding 层生成的掩码是二维的，但注意力层的输入应该是三维或四维的，所以我们需要增加它的维数
        attention_output = self.attention(
            inputs, inputs, attention_mask=mask)  # 两个输入
        proj_input = self.layernorm_1(inputs + attention_output)
        proj_output = self.dense_proj(proj_input)
        return self.layernorm_2(proj_input + proj_output)

    # 保存自定义层在编写自定义层时，一定要实现get_config()方法：这样我们可以利用
    # config字典将该层重新实例化，这对保存和加载模型很有用。
    # 该方法返回一个Python字典，其中包含用于创建该层的构造函数的参数值。
    def get_config(self):
        config = super().get_config()
        config.update({
            "embed_dim": self.embed_dim,
            "num_heads": self.num_heads,
            "dense_dim": self.dense_dim,
        })
        return config


# config 并不包含权重值，因为他是从头开始初始化的

# 在保存包含自定义层的模型时，保存文件中会包含这些 config 字典。
# 从文件中加载模型时，你应该在加载过程中提供自定义层的类，以便其理解 config 对象，如下所示。
# model = keras.models.load_model(
#  filename, custom_objects={"PositionalEmbedding": PositionalEmbedding})
# 注意到，这里使用的规范化层并不是之前在图像模型中使用的 BatchNormalization层。
# 这是因为 BatchNormalization 层处理序列数据的效果并不好。
# 相反，我们使用的是LayerNormalization 层，它对每个序列分别进行规范化，与批量中的其他序列无关。

# 原因解释：因为LN是全部汇聚到最后的一个轴（x = -1）上，另外一个BN是在（0，1，2）轴上汇聚，从而后者能够形成汇聚作用

vocab_size = 20000
embed_dim = 256
num_heads = 2
dense_dim = 32
inputs = keras.Input(shape=(None,), dtype="int64")
x = layers.Embedding(vocab_size, embed_dim)(inputs)
x = TransformerEncoder(embed_dim, dense_dim, num_heads)(x)
x = layers.GlobalMaxPooling1D()(x)
x = layers.Dropout(0.5)(x)
outputs = layers.Dense(1, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
model.summary()

# 自注意力是一种集合处理机制，它关注的是序列元素对之间的关系
# Transformer 编码器根本就不是一个序列模型。
# 它由密集层和注意力层组成，前者独立处理序列中的词元，后者则将词元视为一个集合。