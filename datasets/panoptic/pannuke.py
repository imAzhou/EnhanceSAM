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


       