# 回归术语一览：
# 样本(sample)、输入(input): 进入的数据
# 预测(prediction)、输出(output): 直接的输出结果
# 目标(target): 真实值。对应的答应有的答案
# 预测误差(prediction error)、损失值(loss value): 预测结果与答案之间的差距
# 类别(class): 分类问题中可供选择的一组标签，真实值的集合；分类猫狗之间的：猫和狗
# 真实值(ground-truth)、标注(annotation): 分类问题中类别标注的具体实例，狗1狗2狗3, 则狗就是123图片的标签
# 二分类(binary classification): 两个互斥的分类
# 多分类(multiclass classification): 多个（也许）互斥的分类
# 多标签分类(multilabel classification): 一项分类任务，每个输入样本都可以被分配多个标签。
# 举个例子，一张图片中可能既有猫又有狗，那么应该同时被标注“猫”标签和“狗”标签。每张图片的标签个数通常是可变的。
# 标量回归(scalar regression): 目标是一个连续标量值的任务。
# 向量回归(vector regression): 目标是一组连续值（比如一个连续向量）的任务。
# 小批量(mini-batch)、批量(batch): 模型同时处理的一小部分样本（样本数通常在 8 和 128 之间）。
# 样本数通常取 2 的幂，这样便于在 GPU 上分配内存。训练时，小批量用于计算一次梯度下降，以更新模型权重。

# 在标量回归——波士顿房价预测中，已经经历了Kaggle竞赛初步学习的Titanic实例，
# 故在其中开始按照：数据分析的逻辑进行注释


#总结：
# 对于向量数据，最常见的三类机器学习任务是：二分类问题、多分类问题和标量回归问题。
#  回归问题使用的损失函数和评估指标都与分类问题不同。
#  将原始数据输入神经网络之前，通常需要对其进行数据预处理。
#  如果数据特征具有不同的取值范围，应该先进行预处理，对每个特征单独进行缩放。
#  随着训练的进行，神经网络最终会过拟合，并在前所未见的数据上得到较差的结果。
#  如果训练数据不是很多，那么可以使用只有一两个中间层的小模型，**以避免严重的过拟合**。
#  如果数据被划分为多个类别，那么中间层过小可能会造成信息瓶颈。
#  如果要处理的数据很少，那么 K 折交叉验证有助于可靠地评估模型。
