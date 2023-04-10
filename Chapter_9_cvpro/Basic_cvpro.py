# 9.1 三项基本的计算机视觉任务

#  图像分类（image classification）的目的是为图像指定一个或多个标签。
# 它既可以是单标签分类（一张图像只能属于一个类别，不属于其他类别），
# 也可以是多标签分类（找出一张图像所属的所有类别）。
#  图像分割（image segmentation）的目的是将图像“分割”或“划分”成不同的区域，每个区域通常对应一个类别。
# 例如，使用软件进行视频通话时，你可以在身后设置自定义背景，它就是用图像分割模型将你的脸和身后的物体区分开，
# 并且可以达到像素级的区分效果。
#  目标检测（object detection）的目的是在图像中感兴趣的目标周围绘制矩形（称为边界框），
# 并给出每个矩形对应的类别。例如，自动驾驶汽车可以使用目标检测模型监控摄像头中的汽车、行人和交通标志。

#  语义分割（semantic segmentation）：
# 分别将每个像素划分到一个语义类别，比如“猫”。如果图像中有两只猫，那么对应的像素都会被映射到同一个“猫”类别中。
#  实例分割（instance segmentation）：
# 不仅按类别对图像像素进行分类，还要解析出单个的对象实例。对于包含两只猫的图像，
# 实例分割会将“猫 1”和“猫 2”作为两个独立的像素类别。

# Oxford-IIIT 宠物数据集，其中包含 7390 张不同品种的猫狗图片，以及每张图片的前景 − 背景分割掩码。

# 分割掩码（segmentation mask）相当于图像分割任务的标签：
# 它是与输入图像大小相同的图像，具有单一颜色通道，其中每个整数值对应输入图像中相应像素的类别。
# 本例中，分割掩码的像素值可以取以下三者之一。
#  1（表示前景）
#  2（表示背景）
#  3（表示轮廓）

from HEAD import *
from keras.preprocessing.image import load_img, img_to_array

input_dir = "D:/Downloads/images/"
target_dir = "D:/Downloads/annotations/trimaps/"

input_img_paths = sorted(
    [os.path.join(input_dir, fname)
     for fname in os.listdir(input_dir)
     if fname.endswith(".jpg")])  # 显示索引编号为 9 的输入图像

target_paths = sorted(
    [os.path.join(target_dir, fname)
     for fname in os.listdir(target_dir)
     if fname.endswith(".png") and not fname.startswith(".")])


def show_image9():
    fig, axs = plt.subplots(1, 2)
    axs[0].imshow(load_img(input_img_paths[9]))
    axs[0].set_title('Pic 0')
    axs[0].axis("off")

    # 原始标签是 1、2、3。我们减去 1，使标签的值变为 0 ～ 2，然后乘以 127，
    # 使标签变为 0（黑色）、127（灰色）、254（接近白色）

    def display_target(target_array):
        normalized_array = (target_array.astype("uint8") - 1) * 127
        axs[1].imshow(normalized_array[:, :, 0])
        axs[1].set_title('Mask 0')
        axs[1].axis("off")

    img = img_to_array(load_img(target_paths[9], color_mode="grayscale"))
    display_target(img)
    plt.show()


img_size = (200, 200)  # imput images size
num_imgs = len(input_img_paths)  # amount of samples
random.Random(1337).shuffle(input_img_paths)  # shuffle the path of jpgs, which initially sorted by class
random.Random(1337).shuffle(target_paths)  # these two


def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))


def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8") - 1  # makes the labels into 0,1,2
    return img


input_imgs = np.zeros((num_imgs,) + img_size + (3,), dtype="float32")
targets = np.zeros((num_imgs,) + img_size + (1,), dtype="uint8")
for i in range(num_imgs):
    input_imgs[i] = path_to_input_image(input_img_paths[i])
    targets[i] = path_to_target(target_paths[i])

num_val_samples = 1000  # validation groups
train_input_imgs = input_imgs[:-num_val_samples]
train_targets = targets[:-num_val_samples]
val_input_imgs = input_imgs[-num_val_samples:]
val_targets = targets[-num_val_samples:]


def get_model(img_size, num_classes):
    inputs = keras.Input(shape=img_size + (3,))
    x = layers.Rescaling(1. / 255)(inputs)
    # padding  = "same" 来避免边界填充对于特征图的影响
    x = layers.Conv2D(64, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, strides=2, activation="relu", padding="same")(x)
    x = layers.Conv2D(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2D(256, 3, strides=2, padding="same", activation="relu")(x)
    x = layers.Conv2D(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(256, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        256, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(128, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        128, 3, activation="relu", padding="same", strides=2)(x)
    x = layers.Conv2DTranspose(64, 3, activation="relu", padding="same")(x)
    x = layers.Conv2DTranspose(
        64, 3, activation="relu", padding="same", strides=2)(x)
    outputs = layers.Conv2D(num_classes, 3, activation="softmax",
                            padding="same")(x)
    model = keras.Model(inputs, outputs)
    return model


if __name__ == '__main__':
    # show_image9()
    model = get_model(img_size, num_imgs)
    # model.summary()  # cause wo gonna classified into 3 classes, thus here we get 3 softmax pathes
    model.compile(optimizer="rmsprop", loss="sparse_categorical_crossentropy")
    callbacks = [
        keras.callbacks.ModelCheckpoint("oxford_segmentation.keras",
                                        save_best_only=True)
    ]

    config = tf.compat.v1.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.compat.v1.Session(config=config)

    history = model.fit(train_input_imgs, train_targets,
                        epochs=50,
                        callbacks=callbacks,
                        batch_size=4,
                        validation_data=(val_input_imgs, val_targets))

    epochs = range(1, len(history.history["loss"]) + 1)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.figure()
    plt.plot(epochs, loss, "bo", label="Training loss")
    plt.plot(epochs, val_loss, "b", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.show()
