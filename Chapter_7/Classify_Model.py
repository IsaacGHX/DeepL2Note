from HEAD import *

# 7.2.3 模型子类化
# 最后一种构建模型的方法是最高级的方法：模型子类化，也就是将 Model 类子类化。

#  在 __init__() 方法中，定义模型将使用的层；
#  在 call() 方法中，定义模型的前向传播，重复使用之前创建的层；
#  将子类实例化，并在数据上调用，从而创建权重。

# 随便虚构的输入数据：
vocabulary_size = 10000  # 有10000个带有标题的数据项
num_tags = 100  # 有100种标签
num_departments = 4  # 处理部门有4个
num_samples = 1280

title_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
text_body_data = np.random.randint(0, 2, size=(num_samples, vocabulary_size))
tags_data = np.random.randint(0, 2, size=(num_samples, num_tags))

# 随便虚构的存储目标数据：
priority_data = np.random.random(size=(num_samples, 1))
department_data = np.random.randint(0, 2, size=(num_samples, num_departments))


# 工单管理模型
class CustomerTicketModel(keras.Model):
    def __init__(self, num_departments):
        super().__init__()  # 不要忘记调用 super()构造函数！
        # 在构造函数中定义子层
        # 64 * relu + 1 * sigmoid, 最后分类是softmax
        self.concat_layer = layers.Concatenate()
        self.mixing_layer = layers.Dense(64, activation="relu")
        self.priority_scorer = layers.Dense(1, activation="sigmoid")
        self.department_classifier = layers.Dense(
            num_departments, activation="softmax")

    def call(self, inputs):  # 在 call() 方法中定义前向传播
        title = inputs["title"]
        text_body = inputs["text_body"]
        tags = inputs["tags"]
        features = self.concat_layer([title, text_body, tags])
        features = self.mixing_layer(features)
        priority = self.priority_scorer(features)
        department = self.department_classifier(features)
        return priority, department


model = CustomerTicketModel(num_departments=4)
priority, department = model(
    {"title": title_data, "text_body": text_body_data, "tags": tags_data})

model.compile(optimizer="rmsprop",
              loss=["mean_squared_error", "categorical_crossentropy"],
              metrics=[["mean_absolute_error"], ["accuracy"]])
# 参数 loss 和 metrics 的结构必须与 call() 返回的内容完全匹配——这里是两个元素组成的列表

# 输入数据的结构必须与 call() 方法的输入完全匹配——这里是一个字典，
# 字典的键是 title、text_body 和 tags
model.fit({"title": title_data,
           "text_body": text_body_data,
           "tags": tags_data},
          [priority_data, department_data],
          epochs=1)
# 目标数据的结构必须与 call() 方法返回的内容完全匹配——这里是两个元素组成的列表
model.evaluate({"title": title_data,
                "text_body": text_body_data,
                "tags": tags_data},
               [priority_data, department_data])
priority_preds, department_preds = model.predict({"title": title_data,
                                                  "text_body": text_body_data,
                                                  "tags": tags_data})

# 函数式模型和子类化模型在本质上有很大区别。
# 函数式模型是一种数据结构——它是由层构成的图，你可以查看、检查和修改它。
# 子类化模型是一段字节码——它是带有 call() 方法的 Python 类，其中包含原始代码。
# 这是子类化工作流程具有灵活性的原因——你可以编写任何想要的功能，但它引入了新的限制。
