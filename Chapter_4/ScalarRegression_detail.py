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
    model.compile(optimizer="sgd", loss="mse", metrics=["mae"])
    return model


k = 4
num_val_samples = len(x_train) // k
num_epochs = 500
all_mae_histories = []

for i in range(k):
    print(f"Processing fold #{i}")
    val_data = x_train[i * num_val_samples: (i + 1) * num_val_samples]
    val_targets = y_train[i * num_val_samples: (i + 1) * num_val_samples]
    partial_x_train = np.concatenate(
        [x_train[:i * num_val_samples],
         x_train[(i + 1) * num_val_samples:]],
        axis=0)
    partial_y_train = np.concatenate(
        [y_train[:i * num_val_samples],
         y_train[(i + 1) * num_val_samples:]],
        axis=0)
    model = build_model()
    history = model.fit(partial_x_train, partial_y_train,
                        validation_data=(val_data, val_targets),  # 验证数据集
                        epochs=num_epochs, batch_size=16, verbose=0)
    # 计算每轮的K折验证分数平均值
    mae_history = history.history["val_mae"]
    all_mae_histories.append(mae_history)
    average_mae_history = [
        np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)
    ]

print(average_mae_history[120:130])
import matplotlib.pyplot as plt

plt.plot(range(1, len(average_mae_history) + 1), average_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

truncated_mae_history = average_mae_history[10:]
plt.plot(range(1, len(truncated_mae_history) + 1), truncated_mae_history)
plt.xlabel("Epochs")
plt.ylabel("Validation MAE")
plt.show()

# 中可以看出，验证 MAE 在 120 ～ 140 轮（包含剔除的那 10 轮）
# 后不再显著降低，再之后就开始过拟合了

# 最终的模型
# model = build_model()
# model.fit(x_train, y_train,
#           epochs=130, batch_size=16, verbose=0)
# test_mse_score, test_mae_score = model.evaluate(x_test, y_test)
# print(test_mae_score)
# predictions = model.predict(x_test)
# print(predictions[0])

model = build_model()
model.fit(x_train, y_train,
          epochs=130, batch_size=16, verbose=0)
test_score_mse, test_score_mae = model.evaluate(x_test, y_test)  # evaluate返回值输出损失值和选定的指标值
print(test_score_mae)
predictions = model.predict(x_test)
print(predictions[0])

