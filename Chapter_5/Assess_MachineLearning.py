# 5.2 评估机器学习模型
# 5.2.1 训练集、验证集和测试集
# 在训练数据上训练模型，在验证数据上评估模型。模型准备上线之前，在测试数据上最后测试一次，
# 测试数据应与生产数据尽可能相似。做完这些工作之后，就可以在生产环境中部署该模型。

# 原因在于开发模型时总是需要调节模型配置，比如确定层数或每层大小
# ［这些叫作模型的超参数（hyperparameter），以便与参数（权重）区分开］。
# 这个调节过程需要使用模型在验证数据上的表现作为反馈信号。
# 该过程本质上是一种学习过程：在某个参数空间中寻找良好的模型配置。
# 因此，基于模型在验证集上的表现来调节模型配置，很快会导致模型在验证集上过拟合，
# 即使你并没有在验证集上直接训练模型。

# 造成这一现象的核心原因是信息泄露（information leak）。
# 每次基于模型在验证集上的表现来调节模型超参数，都会将验证数据的一些信息泄露到模型中。
# 如果对每个参数只调节一次，那么泄露的信息很少，验证集仍然可以可靠地评估模型。
# 但如果多次重复这一过程（运行一次实验，在验证集上评估，然后据此修改模型），
# 那么会有越来越多的验证集信息泄露到模型中。
# 如果可用数据很少，那么有几种高级方法可以派上用场。

# 三种经典的评估方法：简单的留出验证、K 折交叉验证，以及带有打乱数据的重复 K 折交叉验证。
# 我们还会介绍使用基于常识的基准，以判断模型训练是否有效。

# 1. 简单的留出验证
# 留出一定比例的数据作为测试集。在剩余的数据上训练模型，然后在测试集上评估模型。
# 如前所述，为防止信息泄露，你不能基于测试集来调节模型，所以还应该保留一个验证集。

import numpy as np


def hold_out(data, get_model, test_data, num_validation_samples):
    np.random.shuffle(data)  # 通常需要打乱数据
    validation_data = data[:num_validation_samples]  # 验证集
    training_data = data[num_validation_samples:]  # 训练集
    model = get_model()
    model.fit(training_data, ...)
    validation_score = model.evaluate(validation_data, ...)
    # 在训练数据上训练模型，然后在验证数据上评估模型
    ...  # 现在可以对模型进行调节、重新训练、评估，然后再次调节
    model = get_model()
    model.fit(np.concatenate([training_data,
                              validation_data]), ...)
    test_score = model.evaluate(test_data, ...)
    # 调节好模型的超参数之后，通常的做法是在所有非测试数据上从头开始训练最终模型

# 缺点：如果可用的数据很少，那么可能验证集包含的样本就很少，无法在统计学上代表数据。
# 这个问题很容易发现：在划分数据前进行不同的随机打乱，
# 如果最终得到的模型性能差别很大，那么就存在这个问题。

# 2. K 折交叉验证：
# K 折交叉验证是指将数据划分为 K 个大小相等的分区。
# 对于每个分区 i，在剩余的 K−1 个分区上训练模型，然后在分区 i 上评估模型。
# 最终分数等于 K 个分数的平均值。
# 对于不同的训练集 − 测试集划分，如果模型的性能变化很大，那么这种方法很有用。
# 与留出验证一样，这种方法也需要独立的验证集来校准模型。

# k = 3
# num_validation_samples = len(data) // k
# np.random.shuffle(data)
# validation_scores = []
# for fold in range(k):
#  validation_data = data[num_validation_samples * fold:
#  num_validation_samples * (fold + 1)] #选择验证数据分区
#  training_data = np.concatenate(
#      data[:num_validation_samples * fold],
#      data[num_validation_samples * (fold + 1):])
#  model = get_model()  # 创建一个全新的模型实例（未训练）
#  model.fit(training_data, ...)
#  validation_score = model.evaluate(validation_data, ...)
#  validation_scores.append(validation_score)
# validation_score = np.average(validation_scores)
# model = get_model()
# model.fit(data, ...)
# test_score = model.evaluate(test_data, ...)

# 3. 带有打乱数据的重复 K 折交叉验证
# 如果可用的数据相对较少**，而你又需要尽可能精确地评估模型，那么可以使用带有打乱数据的重复 K 折交叉验证。
# 我发现这种方法在 Kaggle 竞赛中特别有用。
# 具体做法是多次使用 K 折交叉验证，每次将数据划分为 K 个分区之前都将数据打乱。
# 最终分数是每次 K 折交叉验证分数的平均值。
# 注意，这种方法一共要训练和评估 P * K 个模型（P 是重复次数），**计算代价很大。

# 5.2.2 超越常识的基准
# 本质是要超越古典概率论中对于随机事件的发生概率。
# 比如对于 MNIST 数字分类示例，一个简单的基准是验证精度大于 0.1（随机分类器）；
# 对于 IMDB 示例，基准可以是验证精度大于 0.5。
# 对于路透社示例，由于类别不均衡，因此基准约为 0.18 ～ 0.19。
# 对于一个二分类问题，如果 90% 的样本属于类别 A，10% 的样本属于类别 B，
# 那么一个总是预测类别 A 的分类器就已经达到了 0.9 的验证精度，你需要做得比这更好。

# 5.2.3 模型评估的注意事项
#  数据代表性（data representativeness）。将数据划分为训练集和测试集之前，
# 通常应该随机**打乱数据。来保证数据的充分融合，不会因为某些既有的顺序而导致错误
#  时间箭头（the arrow of time）。对于时间序列预测模型，不应该随机打乱，
# 否则会造成成**时间泄露（temporal leak）：模型将在未来数据上得到有效训练。
# 对于这种情况，应该始终确保测试集中所有数据的时间都晚于训练数据。
#  数据冗余（redundancy in your data）：
# 如果某些数据点出现了两次（这对于现实世界的数据来说十分常见），
# 那么打乱数据并划分成训练集和验证集，将导致训练集和验证集之间出现冗余。
# 从效果上看，你将在部分训练数据上评估模型，这是极其糟糕的。
# 一定要确保训练集和验证集之间没有交集**还是确保训练集与验证集的分隔，不互相影响。


