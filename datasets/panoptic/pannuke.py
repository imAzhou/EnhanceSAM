from .load_embed_dataset import LoadEmbedDataset

class PanNukeDataset(LoadEmbedDataset):
       '''
       '''

       METAINFO = dict(
        classes=('Neoplastic', 'Inflammatory', 'Connective/Soft', 'Dead', 'Epithelial', 'Background'),
        thing_classes=('Neoplastic', 'Inflammatory', 'Connective/Soft', 'Dead', 'Epithelial'),
        stuff_classes=('Background'),
        palette=[[255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0], [255, 128, 0], [255, 255, 255]])

       def __init__(self,**args) -> None:
              super(PanNukeDataset, self).__init__(**args)



class PanNukeBinaryDataset(LoadEmbedDataset):

       METAINFO = dict(
        classes=('Background','Cell',),
        palette=[[255, 255, 255], [47, 243, 15]],
        thing_classes=('Cell'),
        stuff_classes=('Background')
       )

       def __init__(self,**args) -> None:
              super(PanNukeBinaryDataset, self).__init__(**args)
       