import nvidia.dali.fn as fn
import nvidia.dali.types as types
from nvidia.dali.pipeline import Pipeline
import torch
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor

class MyDALIPipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id):
        super(MyDALIPipeline, self).__init__(batch_size, num_threads, device_id)
        self.input = fn.readers.file(file_root='/path/to/your/image/directory/', random_shuffle=True)
        self.decode = fn.decoders.image(device="cuda:0", output_type=types.RGB)
        self.rotate = fn.rotate(self.decode, angle=90)  # 旋转90度
        self.flip = fn.flip(self.rotate, horizontal=1, vertical=0)  # 水平翻转
        self.res = fn.resize(self.flip, resize_x=224, resize_y=224)  # 缩放到224x224

    def define_graph(self):
        inputs = self.input()
        output = self.res(inputs)
        return output

# 创建 DALI Pipeline
batch_size = 32
num_threads = 4
device_id = 0
dali_pipeline = MyDALIPipeline(batch_size=batch_size, num_threads=num_threads, device_id=device_id)
dali_pipeline.build()

# 将 DALI DataLoader 转换为 PyTorch DataLoader
class DALIDataLoaderWrapper(DataLoader):
    def __iter__(self):
        while True:
            for data in super().__iter__():
                yield [torch.tensor(data[0].cpu().numpy())]

dali_loader = DALIDataLoaderWrapper(dali_pipeline, batch_size=batch_size, num_workers=num_threads)


device = torch.device('cuda:0')
for batch in dali_loader:
    inputs = batch[0].to(device)
    labels = batch[1].to(device)
    
    
