# 4.3 标量回归问题实例：预测房价
import keras
# 常见的机器学习问题是回归（regression）问题，它预测的是一个连续值，而不是离散标签，
# 比如根据气象数据预测明日气温，或者根据软件说明书预测完成软件项目所需时间

# 波士顿房价数据集：
# 预测目标：20世纪70年代中期波士顿郊区房价的中位数
# 数据集类别标签：犯罪率、地方房产税税率，包含的数据点一共只有506——小数据集
# 为 404 个训练样本和 102 个测试样本。
# 观察数据集类别的特征：比如犯罪率）都有不同的取值范围。
# 有的特征是比例，取值在 0 和 1 之间；有的取值在 1 和 12之间；还有的取值在 0 和 100 之间。

# 数据集加载：
from keras.datasets import boston_housing
from keras import layers
import numpy as np

(x_train, y_train), (x_test, y_test) = (
    boston_housing.load_data())

# 看数据集的，(拥有的数据量、标签类量)
print(x_train.shape)
print(x_test.shape)

# 看预测目标的内容,房价大都介于 10 000 美元～ 50 000 美元。单位是千美元
print(y_train)

# 数据处理：标准化
# 取值范围差异很大的数据输入到神经网络中，这是有问题的。
# 模型可能会自动适应这种取值范围不同的数据，但这肯定会让学习变得更加困难。
# 对于这类数据，普遍采用的最佳处理方法是对每个特征进行标准化，
# 即对于输入数据的每个特征（输入数据矩阵的每一列），减去特征平均值，再除以标准差，
# 这样得到的特征平均值为 0，标准差为 1。

mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
x_test -= mean
x_test -= std


# *****注意，对测试数据进行标准化的平均值和标准差都是在训练数据上计算得到的。
# 在深度学习工作流程中，你不能使用在测试数据上计算得到的任何结果，
# 即使是像数据标准化这么简单的事情也不行。

# 模型构建：样本数小，所以模型小
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    # 由于需要同个模型多次实例化，用一个函数来构建模型。即函数不变但是layer会加多
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model
    # mse损失函数，即均方误差（mean squared error，MSE），
    # 预测值与目标值之差的平方。这是回归问题常用的损失函数。
    # 平均绝对误差（mean absolute error，MAE）。
    # 它是预测值与目标值之差的绝对值。
    # 如果这个问题的MAE等于0.5，就表示预测房价与实际价格平均相差500美元。


# 模型的最后一层只有一个单元且没有激活，它是一个线性层。
# 这是标量回归（标量回归是预测单一连续值的回归）的典型设置。
# 添加激活函数将限制输出范围。如果向最后一层添加sigmoid 激活函数，
# 那么模型只能学会预测 0 到 1 的值。这里最后一层是纯线性的，
# 所以模型可以学会预测任意范围的值。

# 利用“ K 折交叉验证”来验证你的方法
# 这种方法将可用数据划分为 K 个分区（K 通常取 4 或 5），实例化 K 个相同的模型，
# 然后将每个模型每次在 K−1 个分区上训练，并在剩下的一个分区上进行评估。
# 模型的验证分数等于这 K 个验证分数的平均值。
k = 4  # 假设是4折
num_val_samlples = len(x_train) // k  # 这里相当于切割成了k块
num_epoches = 100  # 训练100次
all_scores = []

for i in range(k):
    print(f"Peocessing fold #{i}")

    val_data = x_train[i * num_val_samlples: (i + 1) * num_val_samlples]
    val_targets = y_train[i * num_val_samlples: (i + 1) * num_val_samlples]
    partial_x_train = np.concatenate(
        [x_train[:i * num_val_samlples],
         x_train[(i + 1) * num_val_samlples:]],
        axis=0)
    partial_y_train = np.concatenate(
        [y_train[:i * num_val_samlples],
         y_train[(i + 1) * num_val_samlples:]],
        axis=0
    )

    model = build_model()
    model.fit(partial_x_train, partial_y_train, epochs=num_epoches,
              batch_size=16, verbose=0)  # verbose = 0 意味着这是静默模式
    val_mse, val_mae = model.evaluate(val_data, val_targets, verbose=0)
    all_scores.append(val_mae)

print(all_scores)
print(np.mean(all_scores))
# 每次运行模型得到的验证分数确实有很大差异，从 2.1 到 3.1 不等。
# 平均分数（2.6）是比单一分数更可靠的指标——这就是 K 折交叉验证的核心要点。
# 在这个例子中，预测房价与实际房价平均相差 2600 美元，
# 考虑到实际房价范围是 10 000 美元～ 50 000 美元，这一差别还是很大的。
