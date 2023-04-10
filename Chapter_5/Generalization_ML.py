from HEAD import *

# 5.4 提高泛化能力
# 你的模型表现出**一定的泛化能力，并且**能够过拟合，接下来应该专注于将**泛化能力最大化。

# 5.4.1 数据集管理
# 你已经知道，深度学习的泛化来源于数据的潜在结构。如果你的数据允许在样本之间进行平滑插值，
# 你就可以训练出一个具有泛化能力的深度学习模型。
# 如果你的数据**过于嘈杂或者本质上是离散的，比如列表排序问题，那么深度学习将无法帮助你解决这类问题。
# 深度学习是曲线拟合，而不是魔法。**错误，后续的实验家们已经做到。
# 因此，你必须确保使用适当的数据集。**在收集数据上花费更多的精力和金钱，
# 几乎总是比在开发更好的模型上花费同样的精力和金钱产生更大的投资回报。
#  确保拥有足够的数据。请记住，你需要对输入 − 输出空间进行**密集采样。
# 利用更多的数据可以得到更好的模型。
# 有时，一开始看起来无法解决的问题，在拥有更大的数据集之后就能得到解决。
#  尽量减少标签错误。将**输入可视化，以检查异常样本并核查标签。
#  清理数据并处理缺失值（第 6 章将详述）。
#  如果有很多特征，而你不确定哪些特征是真正有用的，那么需要进行特征选择。
# 提高数据泛化潜力的一个特别重要的方法就是**特征工程（feature engineering）。
# 对于大多数机器学习问题，特征工程是成功的关键因素。

# 5.4.2 特征工程
# 特征工程是指将数据输入模型之前，利用你自己关于数据和机器学习算法（这里指神经网络）
# 的知识对数据进行硬编码的变换（这种变换不是模型学到的），以改善算法的效果。
# 在多数情况下，机器学习模型无法从完全随意的数据中进行学习。
# 呈现给模型的数据应该便于模型进行学习。
# 人先理解问题后让机器去理解大数据，要比直接让机器理解大数据更省时省力。

# 特征工程的本质：用更简单的方式表述问题，从而使问题更容易解决。
# 特征工程可以让潜在流形变得更平滑、更简单、更有条理。特征工程通常需要深入理解问题。
# 对于现代深度学习，大多数特征工程是不需要做的，因为神经网络能够从原始数据中自动提取有用的特征。

#  良好的特征仍然有助于更优雅地解决问题，同时使用更少的资源。
# 例如，使用卷积神经网络解决读取时钟问题是非常可笑的。
#  良好的特征可以用更少的数据解决问题。深度学习模型自主学习特征的能力依赖于拥有**大量的训练数据。
# 如果只有很少的样本，那么特征的信息价值就变得非常重要。

# 5.4.3 提前终止
# 在深度学习中，我们总是使用**过度参数化的模型：模型自由度远远超过拟合数据潜在流形所需的最小自由度。
# 这种过度参数化并不是问题，因为永远不会完全拟合一个深度学习模型。
# 这样的拟合根本没有泛化能力。你总是在达到最小训练损失之前很久就会中断训练。
# 在训练过程中找到最佳泛化的拟合，即欠拟合曲线和过拟合曲线之间的**确切界线，是提高泛化能力的最有效的方法之一。
# 在第 4 章的例子中，我们首先让模型训练时间比需要的时间更长，以确定最佳验证指标对应的轮数，
# 然后重新训练一个新模型，正好训练这个轮数。这是很标准的做法，但需要做一些冗余工作，有时代价很高。
# 当然，你也可以在每轮结束时保存模型，一旦找到了最佳轮数，就重新使用最近一次保存的模型。
# 在 Keras 中，我们通常使用 **EarlyStopping 回调函数来实现这一点，
# 它会在验证指标停止改善时立即中断训练，同时记录最佳模型状态。

# 5.4.4 模型正则化
# 正则化可以主动降低模型完美拟合训练数据的能力，其目的是提高模型的验证性能。
# 它之所以被称为模型的“正则化”，是因为它通常使模型变得更简单、更“规则”，曲线更平滑、更“通用”。
# 因此，模型对训练集的针对性更弱，能够更好地近似数据的潜在流形，从而具有更强的泛化能力。
# 请记住，模型正则化过程应该始终由一个准确的评估方法来引导。
# 只有能够衡量泛化，你才能实现泛化。

# 以下是几种正则化方法：

# 1. 缩减模型容量
# 一个太小的模型不会过拟合。降低过拟合最简单的方法，就是缩减模型容量。
# 即减少模型中可学习参数的个数（这由层数和每层单元个数决定）。
# 如果模型的记忆资源有限，它就不能简单地记住训练数据；
# 为了让损失最小化，它必须学会对目标有预测能力的压缩表示，这也正是我们感兴趣的数据表示。
# 同时请记住，你的模型应该具有足够多的参数，以防欠拟合，即模型应避免记忆资源不足。
# 在容量过大和容量不足之间，要找到一个平衡点。


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

# model = keras.Sequential([
#     layers.Dense(4, activation="relu"),
#     layers.Dense(4, activation="relu"),
#     layers.Dense(1, activation="sigmoid")
# ])
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# history_smaller_model = model.fit(
#     train_data, train_labels,
#     epochs=20, batch_size=512, validation_split=0.4)

loss_origin = history_original.history["val_loss"]
epochs = range(1, 21)
plt.plot(epochs, loss_origin, "b-",
         label="Val loss origin module")
# loss_smaller = history_smaller_model.history["val_loss"]

# plt.plot(epochs, loss_smaller, "b--",
#          label="Val loss smaller module")

# 较小模型开始过拟合的时间要晚于初始模型，而且开始过拟合之后，它的性能下降速度也更慢。
# 我们现在添加一个容量更大的模型——其容量远大于问题所需。
# 虽然过度参数化的模型很常见，但肯定会有这样一种情况：模型的记忆容量过大。
# 如果模型立刻开始过拟合，而且它的验证损失曲线看起来很不稳定、方差很大，你就知道模型容量过大了。
# （不过验证指标不稳定的原因也可能是验证过程不可靠，比如验证集太小）

# model = keras.Sequential([
#     layers.Dense(512, activation="relu"),
#     layers.Dense(512, activation="relu"),
#     layers.Dense(1, activation="sigmoid")
# ])
# model.compile(optimizer="rmsprop",
#               loss="binary_crossentropy",
#               metrics=["accuracy"])
# history_larger_model = model.fit(
#     train_data, train_labels,
#     epochs=20, batch_size=512, validation_split=0.4)
#
# loss_lager = history_larger_model.history["val_loss"]
# plt.plot(epochs, loss_lager, "g--",
#          label="Val loss larger module")


# 仅仅过了一轮，较大模型几乎立即开始过拟合，而且过拟合程度要严重得多。它的验证损失波动更大。
# 此外，它的训练损失很快就接近于零。
# 模型的容量越大，它拟合训练数据的速度就越快（得到很小的训练损失），
# 但也更容易过拟合（导致训练损失和验证损失有很大差异）。

# 2. 添加权重正则化
# 你可能知道奥卡姆剃刀原理：如果一件事有两种解释，那么最可能正确的解释就是更简单的那种，即假设更少的那种。
# 这个原理也适用于神经网络学到的模型：给定训练数据和网络架构，多组权重值（多个模型）都可以解释这些数据。
# 简单模型比复杂模型更不容易过拟合。
# 这里的简单模型是指参数值分布的熵更小的模型（或参数更少的模型，比如上一节中的例子）。
# 因此，降低过拟合的一种常见方法就是强制让模型权重只能取较小的值，从而限制模型的复杂度，
# 这使得权重值的分布更加规则。这种方法叫作权重正则化（weight regularization），
# 其实现方法是向模型损失函数中添加与较大权重值相关的成本（cost）。这种成本有两种形式。
#  L1 正则化：添加的成本与权重系数的绝对值（权重的 L1 范数）成正比。
#  L2 正则化：添加的成本与权重系数的平方（权重的 L2 范数）成正比。
# 神经网络的 L2正则化也叫作权重衰减（weight decay）。
# 不要被不同的名称迷惑，权重衰减与 L2 正则化在数学上是完全相同的。
# 在 Keras 中，添加权重正则化的方法是向层中传入权重正则化项实例（weight regularizer  instance）
# 作为关键字参数。下面我们向最初的影评分类模型中添加 L2 权重正则化。

from keras import regularizers

# regularizers.l1(0.001)
# regularizers.l1_l2(l1=0.002, l2=0.002)
model = keras.Sequential([
    layers.Dense(16,
                 kernel_regularizer= \
                     # regularizers.l2(0.002),
                 # regularizers.l1(0.001),
                 regularizers.l1_l2(l1=0.002, l2=0.002),
                 activation="relu"),
    layers.Dense(16,
                 kernel_regularizer= \
                     # regularizers.l2(0.002),
                 # regularizers.l1(0.001),
                 regularizers.l1_l2(l1=0.002, l2=0.002),
                 activation="relu"),
    layers.Dense(1, activation="sigmoid")
])
model.compile(optimizer="rmsprop",
              loss="binary_crossentropy",
              metrics=["accuracy"])
history_l2_reg = model.fit(
    train_data, train_labels,
    epochs=20, batch_size=512, validation_split=0.4)

loss_regL2 = history_l2_reg.history["val_loss"]
plt.plot(epochs, loss_regL2, "g-",
         label="Val loss RegulizeL2 module")

plt.xlabel("Epochs")
plt.ylabel("val_loss")
plt.legend()
plt.show()

# 权重正则化更常用于较小的深度学习模型。大型深度学习模型往往是过度参数化的，
# 限制权重值大小对模型容量和泛化能力没有太大影响。
# 在这种情况下，应首选另一种正则化方法：dropout。


