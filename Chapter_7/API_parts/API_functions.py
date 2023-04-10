from HEAD import *

# 7.2.2 函数式 API
# 你在现实世界中遇到的大多数 Keras 模型属于这种类型。它很强大，也很有趣，就像拼乐高积木一样。
# api层级分类：
# keras > layers > dense
# dense的返回值可以直接传给下面一层的dense作为连接
# layers是一个列表当中包含了每一层，其中需要调用哪一层就用第几个元元素 layers [i]
# 永远是dense的返回值在层之间传递！！

# 1. 简单示例
# 我们先来看一个简单的示例，即 7.2.1 节中的两层堆叠。这个例子也可以用函数式 API 来实现

# 声明一个 Input（注意，你也可以对输入对象命名，就像对其他对象一样）
# 这个 inputs 对象保存了关于模型将处理的数据的形状和数据类型的信息。
inputs = keras.Input(shape=(3,), name="my_input")
# 对于这个模型处理的批量，每个样本的形状为 (3,)。
# 每个批量的样本数量是可变的（代码中的批量大小为 None）数据批量的数据类型为 float32。
# 我们将这样的对象叫作符号张量（symbolic tensor）。
# 它不包含任何实际数据，但编码了调用模型时实际数据张量的详细信息。它代表的是未来的数据张量。

# 接下来，我们创建了一个层，并在输入上调用该层。
features = layers.Dense(64, activation="relu")(inputs)
outputs = layers.Dense(10, activation="softmax")(features)  # 通过上一层()链接
model = keras.Model(inputs=inputs, outputs=outputs)  # 同样的, outputs也是可以声明的

# 所有 Keras 层都可以在真实的数据张量与这种符号张量上调用。
# 对于后一种情况，层返回的是一个新的符号张量，其中包含更新后的形状和数据类型信息。

print(features.shape)

# 得到最终输出之后，我们在 Model 构造函数中指定输入和输出，将模型实例化。
outputs = layers.Dense(10, activation="softmax")(features)
model = keras.Model(inputs=inputs, outputs=outputs)

# model.summary()

# 2. 多输入、多输出模型
# 与上述简单模型不同，大多数深度学习模型看起来不像列表，而像图。
# 比如，模型可能有多个输入或多个输出。正是对于这种模型，函数式 API 才真正表现出色。
# 假设你要构建一个系统，按优先级对客户支持工单进行排序，并将工单转给相应的部门。
# 这个模型有 3 个输入：
#  工单标题（文本输入）
#  工单的文本正文（文本输入）
#  用户添加的标签（分类输入，假定为 one-hot 编码）
# 我们可以将文本输入编码为由 1 和 0 组成的数组，数组大小为 vocabulary_size。

# 模型还有 2 个输出：
#  工单的优先级分数，它是介于 0 和 1 之间的标量（sigmoid 输出）
#  应处理工单的部门（对所有部门做 softmax）
# 利用函数式 API，仅凭几行代码就可以构建这个模型，
vocabulary_size = 10000  # 有10000个带有标题的数据项
num_tags = 100  # 有100种标签
num_departments = 4  # 处理部门有4个

# 输入：标题、文本、标签
title = keras.Input(shape=(vocabulary_size,), name="title")
text_body = keras.Input(shape=(vocabulary_size,), name="text_body")
tags = keras.Input(shape=(num_tags,), name="tags")

# 通过拼接将输入特征组合成张量 features
features = layers.Concatenate()([title, text_body, tags])
# 利用中间层，将输入特征重组为更加丰富的表示
features = layers.Dense(64, activation="relu")(features)

# 输出层：
priority = layers.Dense(1, activation="sigmoid", name="priority")(features)
department = layers.Dense(
    num_departments, activation="softmax", name="department")(features)
# 通过指定输入和输出来创建模型
model = keras.Model(inputs=[title, text_body, tags],
                    outputs=[priority, department])

# model.summary()

# 3. 训练一个多输入、多输出模型：
# 这种模型的训练方法与序贯模型相同，都是对输入数据和输出数据组成的列表调用 fit()。
# 这些数据列表的顺序应该与传入 Model 构造函数的 inputs 的顺序相同。
num_samples = 1280

# 随便虚构的输入数据：
title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# 随便虚构的存储目标数据：
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))

model.compile(optimizer="rmsprop",
              loss=["mean_squared_error", "categorical_crossentropy"],
              metrics=[["mean_absolute_error"], ["accuracy"]])
model.fit([title_data, text_body_data, tags_data],
          [priority_data, department_data],
          epochs=1)
model.evaluate([title_data, text_body_data, tags_data],
               [priority_data, department_data])
priority_preds, department_preds = model.predict(
    [title_data, text_body_data, tags_data])
# 如果不想依赖输入顺序（比如有多个输入或输出），你也可以为 Input 对象和输出层指定名称，通过字典传递数据
model.compile(optimizer="rmsprop",
              loss={"priority": "mean_squared_error",
                    "department": "categorical_crossentropy"},
              metrics={"priority": ["mean_absolute_error"],
                       "department": ["accuracy"]})
model.fit({"title": title_data,
           "text_body": text_body_data,
           "tags": tags_data
           },
          {"priority": priority_data,
           "department": department_data
           },
          epochs=1)
model.evaluate({
    "title": title_data,
    "text_body": text_body_data,
    "tags": tags_data
},
    {
        "priority": priority_data,
        "department": department_data
    })

priority_preds, department_preds = model.predict(
    {"title": title_data,
     "text_body": text_body_data,
     "tags": tags_data
     })

# 4. 函数式 API 的强大之处：获取层的连接方式
# 函数式模型是一种图数据结构。这便于我们查看层与层之间是如何连接的，
# 并重复使用之前的图节点（层输出）作为新模型的一部分。
# 它也很适合作为大多数研究人员在思考深度神经网络时使用的“思维模型”：由层构成的图。
# 它有两个重要的用处：模型可视化与特征提取。
# 我们来可视化上述模型的连接方式（模型的拓扑结构）。你可以用 plot_model() 将函数式模型绘制成图。

tf.keras.utils.plot_model(model, "ticket_classifier.png")

# 你可以将模型每一层的输入形状和输出形状添加到这张图中，这对调试很有帮助。
tf.keras.utils.plot_model(
    model, "ticket_classifier_with_shape_info.png", show_shapes=True)

print(model.layers)
print(model.layers[3].input)

# 这样一来，我们就可以进行特征提取，重复使用模型的中间特征来创建新模型。
# 假设你想对前一个模型增加一个输出——估算某个问题工单的解决时长，这是一种难度评分。
# 实现方法是利用包含 3 个类别的分类层，这 3 个类别分别是“快速”“中等”和“困难”。
# 你无须从头开始重新创建和训练模型。你可以从前一个模型的中间特征开始（这些中间特征是可以访问的）。

features = model.layers[4].output
difficulty = layers.Dense(3, activation="softmax", name="difficulty")(features)
new_model = keras.Model(
    inputs=[title, text_body, tags],
    outputs=[priority, department, difficulty])

tf.keras.utils.plot_model(
    new_model, "updated_ticket_classifier.png", show_shapes=True)
