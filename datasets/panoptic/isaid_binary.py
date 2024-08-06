from .load_embed_dataset import LoadEmbedDataset

'''
iSAID_palette = {
        0: [0, 0, 0],
        1: [0, 0, 63],
        2: [0, 63, 63],
        3: [0, 63, 0],
        4: [0, 63, 127],
        5: [0, 63, 191],
        6: [0, 63, 255],
        7: [0, 127, 63],
        8: [0, 127, 127],
        9: [0, 0, 127],
        10: [0, 0, 191],
        11: [0, 0, 255],
        12: [0, 191, 127],
        13: [0, 127, 191],
        14: [0, 127, 255],
        15: [0, 100, 155]
    }
[{'id': 1, 'name': 'storage_tank'}, {'id': 2, 'name': 'Large_Vehicle'}, {'id': 3, 'name': 'Small_Vehicle'}, {'id': 4, 'name': 'plane'}, {'id': 5, 'name': 'ship'}, {'id': 6, 'name': 'Swimming_pool'}, {'id': 7, 'name': 'Harbor'}, {'id': 8, 'name': 'tennis_court'}, {'id': 9, 'name': 'Ground_Track_Field'}, {'id': 10, 'name': 'Soccer_ball_field'}, {'id': 11, 'name': 'baseball_diamond'}, {'id': 12, 'name': 'Bridge'}, {'id': 13, 'name': 'basketball_court'}, {'id': 14, 'name': 'Roundabout'}, {'id': 15, 'name': 'Helicopter'}]
'''

class iSAIDBinaryDataset(LoadEmbedDataset):

       METAINFO = dict(
        classes=('Background','Large_Vehicle',),
        palette=[[255, 255, 255], [179, 250, 7]],
        thing_classes=('Large_Vehicle'),
        stuff_classes=('Background')
       )

       def __init__(self,**args) -> None:
              super(iSAIDBinaryDataset, self).__init__(**args)
