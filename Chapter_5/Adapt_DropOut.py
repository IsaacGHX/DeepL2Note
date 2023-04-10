# 3. 添加 dropout：
# dropout 是神经网络最常用且最有效的正则化方法之一，
# 对某一层使用 dropout，就是在训练过程中随机舍弃该层的一些输出特征（将其设为 0）。
# 比方说，某一层在训练过程中对给定输入样本的返回值应该是向量 [0.2, 0.5, 1.3, 0.8, 1.1]。
# 使用 dropout 之后，这个向量会有随机几个元素变为 0，比如变为 [0,0.5, 1.3, 0, 1.1]。
# dropout 比率（dropout rate）是指被设为 0 的特征所占的比例，它通常介于 0.2 ～ 0.5。
# 测试时没有单元被舍弃，相应地，该层的输出值需要按 dropout 比率缩小，
# 因为这时比训练时有更多的单元被激活，需要加以平衡。
# 考虑一个包含某层输出的 NumPy 矩阵 layer_output，其形状为 (batch_size, features)。
# 训练时，我们随机将矩阵中的一些值设为 0。

# layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
# layer_output *= 0.5
# layer_output *= np.random.randint(0, high=2, size=layer_output.shape)
# layer_output /= 0.5

# dropout 的核心思想是在层的输出值中引入噪声，打破不重要的偶然模式。
# 如果没有噪声，那么神经网络将记住这些偶然模式。
#  Keras 中，你可以通过 Dropout 层向模型中引入 dropout。dropout 将被应用于前一层的输出。
#  下面我们向 IMDB 模型中添加两个 Dropout 层，看看它降低过拟合的效果如何
from HEAD import *

(train_data, train_labels), _ = imdb.load_data(num_words=10000)


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


train_data = vectorize_sequences(train_data)

model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_original = model.fit(train_data, train_labels,
                             epochs=20, batch_size=512, validation_split=0.4)


model = keras.Sequential([
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(16, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_dropout = model.fit(
    train_data, train_labels,
    epochs=20, batch_size=512, validation_split=0.4)

loss_origin = history_original.history["val_loss"]
loss_dropout = history_dropout.history["val_loss"]
epochs = range(1, 21)
plt.plot(epochs, loss_origin, "b-",
         label="Val loss origin module")
plt.plot(epochs, loss_dropout, "b--",
         label="Val loss dropout module")
plt.title("Effect of drop out on validation loss")
plt.xlabel("Epochs")
plt.ylabel("val_loss")
plt.legend()
plt.show()

# dropout 的效果比初始模型有了明显改善，似乎比 L2 正则化的效果也要好得多，因为最小验证损失值变得更小。
