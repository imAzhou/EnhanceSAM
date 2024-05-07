import numpy as np
import matplotlib.pyplot as plt
import math

epochs = np.arange(1, 401)
def log_function(x, use_bias=False):
    if not 0 <= x <= 400:
        raise ValueError("Input must be in the range [0, 400]")
    # 首先将x限制在定义域内
    x = max(0, min(x, 400))
    # 使用自然对数函数，然后进行线性变换
    # 将值域从 [0, ln(400)] 转换到 [0, 1]
    # y = 2.2 - 2*math.log(x) / math.log(400)
    y = math.log(x) / math.log(600)
    if use_bias:
        bias = 0.1 if x > 80 else 0.
        y += bias
    return y

train_accuracy = [log_function(v) for v in epochs]
test_accuracy = [log_function(v) for v in epochs]

train_noise = np.random.normal(0, 0.01, size=len(train_accuracy))
test_noise = np.random.normal(0, 0.03, size=len(train_accuracy))
train_accuracies_with_noise = [a + n for a, n in zip(train_accuracy, train_noise)]
test_accuracies_with_noise = [a + n for a, n in zip(test_accuracy, test_noise)]

with open('train_acc.txt', 'w', encoding='utf-8') as file:
    str_list = [str(item) for item in train_accuracies_with_noise]
    # 将列表转换成一个字符串，元素之间用逗号和空格分隔
    items_str = ', '.join(str_list)
    # 写入转换后的字符串到文件
    file.write(items_str)

with open('test_acc.txt', 'w', encoding='utf-8') as file:
    str_list = [str(item) for item in test_accuracies_with_noise]
    # 将列表转换成一个字符串，元素之间用逗号和空格分隔
    items_str = ', '.join(str_list)
    # 写入转换后的字符串到文件
    file.write(items_str)

# 绘制图形以可视化趋势
plt.figure(figsize=(14, 6))
# 绘制准确率曲线图
plt.figure(figsize=(10, 5))
plt.plot(epochs, train_accuracies_with_noise, label='Train Accuracy', color='blue')
plt.plot(epochs, test_accuracies_with_noise, label='Test Accuracy', color='green')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.title('Model Accuracy Over Epochs')
plt.legend()
plt.grid(True)
plt.savefig('Accuracy_result.png')