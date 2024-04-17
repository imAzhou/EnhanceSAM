import re
import matplotlib.pyplot as plt

base_dir = '/x22201018/codes/EnhanceSAM/logs/cls_proposal/2024_01_22_15_18_25/2024_01_31_12_18_21_point_discriminate'
log_file_path = f'{base_dir}/result.log'
loss_png_save_path = f'{base_dir}/loss.png'

# 读取 result.log 文件
with open(log_file_path, 'r') as file:
    lines = file.readlines()

# 提取 bce_loss 的值
losses = []
pattern = re.compile(r'loss : ([0-9.]+)')

for line in lines:
    match = pattern.search(line)
    if match:
        loss_value = float(match.group(1))
        losses.append(loss_value)

# 绘制折线图
iterations = range(1, len(losses) + 1)

plt.figure(figsize=(12, 6))
plt.plot(iterations, losses, label='Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.title('Loss Over Iterations')
plt.legend()
plt.grid(True)

# 保存图像
plt.savefig(loss_png_save_path)
