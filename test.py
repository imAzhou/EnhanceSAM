import matplotlib.pyplot as plt
import numpy as np

# 示例数据
data = np.random.randn(256, 256)  # 随机生成 10x10 的二维数组

# 创建图形
plt.figure(figsize=(8, 6))

# 使用热图显示数据，使用不同的颜色映射（例如 RdBu）
plt.imshow(data, cmap='RdBu', interpolation='nearest')

# 添加颜色条
plt.colorbar()

# 添加标题
plt.title('Heatmap with Positive and Negative Values')

# 显示图形
plt.savefig('test.png')
