from .ss_base_dataset import SSBaseDataset

class LoveDADataset(SSBaseDataset):

       METAINFO = dict(
              classes=('background', 'building', 'road', 'water', 'barren', 'forest',
                     'agricultural'),
              palette=[[255, 255, 255], [255, 0, 0], [255, 255, 0], [0, 0, 255],
                     [159, 129, 183], [0, 255, 0], [255, 195, 128]])

       def __init__(self,**args) -> None:
              super(LoveDADataset, self).__init__(**args)
