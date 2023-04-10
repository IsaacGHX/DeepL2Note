import keras
from keras.datasets import boston_housing
from keras import layers
import numpy as np

(x_train, y_train), (x_test, y_test) = (
    boston_housing.load_data())

# 数据处理：标准化
mean = x_train.mean(axis=0)
x_train -= mean
std = x_train.std(axis=0)
x_train /= std
x_test -= mean
x_test -= std


# 模型构建：样本数小，所以模型小
def build_model():
    model = keras.Sequential([
        layers.Dense(64, activation="relu"),
        layers.Dense(64, activation="relu"),
        layers.Dense(1)
    ])
    model.compile(optimizer="rmsprop", loss="mse", metrics=["mae"])
    return model


# for i in range(119, 130):
#     print(f"Epochs = #{i}: ")
model = build_model()
model.fit(x_train, y_train,
          epochs=130, batch_size=16, verbose=0)
test_score_mse, test_score_mae = model.evaluate(x_test, y_test)  # evaluate返回值输出损失值和选定的指标值
print(test_score_mae)
predictions = model.predict(x_test)
print(predictions[0])
# ?????似乎这里出现了问题数据结果及其的大

#  回归问题使用的损失函数与分类问题不同。回归常用的损失函数是均方误差（MSE）。
#  同样，回归问题使用的评估指标也与分类问题不同。
# 显然，精度的概念不再适用于回归问题。常用的回归指标是平均绝对误差（MAE）
#  如果输入数据的特征具有不同的取值范围，那么应该先进行预处理，对每个特征单独进行缩放。
#  如果可用的数据很少，那么 K 折交叉验证是评估模型的可靠方法。
#  如果可用的训练数据很少，那么最好使用中间层较少（通常只有一两个）的小模型，以避免严重的过拟合。
