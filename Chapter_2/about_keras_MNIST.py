from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

print(train_images.shape)  # 训练数据的60000张28*28的矩阵图像
# (60000, 28, 28)
print(len(train_labels))  # 训练数组长度就是维度就是训练样本个数
# 60000
print(train_labels)  # 训练的label，预制答案
# array([5, 0, 4, ..., 5, 6, 8], dtype=uint8)
# 这里uint8的意思是取值[0,255]

print(test_images.shape)  # 测试数据的10000张28*28的矩阵图像
# (10000, 28, 28)
print(len(test_labels))  # 测试数组长度就是维度就是测试样本个数
# 10000
print(test_labels)  # 测试的label，标准答案
# array([7, 2, 1, ..., 4, 5, 6], dtype=uint8)
