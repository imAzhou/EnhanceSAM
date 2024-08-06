from .load_embed_dataset import LoadEmbedDataset

class CellBinaryDataset(LoadEmbedDataset):

       METAINFO = dict(
        classes=('Background','Cell',),
        palette=[[255, 255, 255], [47, 243, 15]],
        thing_classes=('Cell'),
        stuff_classes=('Background')
       )

       def __init__(self,**args) -> None:
              super(CellBinaryDataset, self).__init__(**args)
