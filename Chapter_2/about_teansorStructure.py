from tensorflow.python import keras
from keras.datasets import mnist
from tensorflow.python.keras import layers

# 神经网络的核心组件是层（layer）。
# 数据过滤器：将简单的层链接起来，从而实现渐进式的数据蒸馏（data distillation）。
# 深度学习模型就像是处理数据的筛子，包含一系列越来越精细的数据过滤器（也就是层）


# 含 2 个 Dense 层，它们都是密集连接（也叫全连接）的神经层。
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

model = keras.Sequential([
    layers.Dense(512, activation="relu"),  # Relu函数
    layers.Dense(10, activation="softmax")  # softmax函数
    # 第 2 层（也是最后一层）是一个 10 路 softmax 分类层，它将返回一个由 10 个概率值（总和为 1）组成的数组。
    # 每个概率值表示当前数字图像属于 10 个数字类别中某一个的概率。
])

model.compile(optimizer="rmsprop",  # 优化器（optimizer）：模型基于训练数据来自我更新的机制，其目的是提高模型性能。
              loss="sparse_categorical_crossentropy",
              # 损失函数（loss function）：模型如何衡量在训练数据上的性能，从而引导自己朝着正确的方向前进。
              metrics=["accuracy"]
              # 在训练和测试过程中需要监控的指标（metric）：本例只关心精度（accuracy），即正确分类的图像所占比例。
              )

# 准备图像数据
train_images = train_images.reshape((60000, 28 * 28))
train_images = train_images.astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28))
test_images = test_images.astype("float32") / 255

# 过调用模型的 fit 方法
model.fit(train_images, train_labels, epochs=5, batch_size=128)
# Epoch 1/5
# 60000/60000 [===========================] - 5s - loss: 0.2524 - acc: 0.9273
# Epoch 2/5
# 51328/60000 [=====================>.....] - ETA: 1s - loss: 0.1035 - acc: 0.9692

test_digits = test_images[0:10]
predictions = model.predict(test_digits)
print(predictions[0])
# 十个模型对应的效可能性
# array([1.0726176e-10, 1.6918376e-10, 6.1314843e-08, 8.4106023e-06,
#        2.9967067e-11, 3.0331331e-09, 8.3651971e-14, 9.9999106e-01,
#        2.6657624e-08, 3.8127661e-07], dtype=float32)

print(predictions[0].argmax())
# 显示预测数据

print(predictions[0][7])
# 显示“7”的预测数据的预测概率
# 0.99999106

print(test_labels[0])
# 7

test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"test_acc: {test_acc}")
# 在新数据上评估模型
# test_acc: 0.9785
# 测试精度约为 97.8%，比训练精度（98.9%）低不少。
# 训练精度和测试精度之间的这种差距是过拟合（overfit）造成的。
# 过拟合是指机器学习模型在新数据上的性能往往比在训练数据上要差


