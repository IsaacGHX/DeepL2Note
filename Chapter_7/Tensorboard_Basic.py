from HEAD import *

# 7.3.4 利用 TensorBoard 进行监控和可视化
# 要想做好研究或开发出好的模型，你在实验过程中需要获得丰富且频繁的反馈，从而了解模型内部发生了什么。
# 这正是运行实验的目的：获取关于模型性能好坏的信息，并且越多越好。

# 取得进展是一个反复迭代的过程，或者说是一个循环：
# 首先，你有一个想法，并将其表述为一个实验，用于验证你的想法是否正确；
# 然后，你运行这个实验并处理生成的信息；这又激发了你的下一个想法。
# 在这个循环中，重复实验的次数越多，你的想法就会变得越来越精确、越来越强大。
# Keras 可以帮你尽快将想法转化成实验，高速 GPU 则可以帮你尽快得到实验结果。

# TensorBoard 是一个基于浏览器的应用程序，可以在本地运行。它是在训练过程中监控模型的最佳方式。
# 利用 TensorBoard，你可以做以下工作：
#  在训练过程中以可视化方式监控指标；
#  将模型架构可视化；
#  将激活函数和梯度的直方图可视化；
#  以三维形式研究嵌入。
# 果监控除模型最终损失之外的更多信息，则可以更清楚地了解模型做了什么、没做什么，
# 并且能够更快地取得进展。
#
# 要将 TensorBoard 与 Keras 模型和 fit() 方法一起使用，最简单的方式就是使用 keras.
# callbacks.TensorBoard 回调函数。
# 在最简单的情况下，只需指定让回调函数写入日志的位置即可。
(images, labels), (test_images, test_labels) = mnist.load_data()

images = images.reshape((60000, 28 * 28)).astype("float32") / 255
test_images = test_images.reshape((10000, 28 * 28)).astype("float32") / 255  # 保留一部分作为测试数据
train_images, val_images = images[10000:], images[:10000]
train_labels, val_labels = labels[10000:], labels[:10000]

model = get_mnist_model()
model.compile(optimizer="rmsprop",
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
tensorboard = keras.callbacks.TensorBoard(
    log_dir="/full_path_to_your_log_dir",
)
model.fit(train_images, train_labels,
          epochs=10,
          validation_data=(val_images, val_labels),
          callbacks=[tensorboard])


# cmd 运行：
# tensorboard --logdir /full_path_to_your_log_dir
# 然后可以访问该命令返回的 URL，以显示 TensorBoard 界面。
# 如果在 Colab 笔记本中运行脚本，则可以使用以下命令，将 TensorBoard 嵌入式实例作为笔记本的一部分运行。
# %load_ext tensorboard
# %tensorboard --logdir /full_path_to_your_log_dir
# 在 TensorBoard 界面中，你可以实时监控训练指标和评估指标的图像，如图 7-7 所示。
