data_root = '/x22201018/datasets/RemoteSensingDatasets/InriaBuildingDataset'
dataset_tag = 'inria'
image_size = (512, 512)
data_pipeline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=None),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args=None),
    # dict(type='RandomFlip', prob=0.5),
    # dict(
    #     type='RandomResize',
    #     scale=image_size,
    #     ratio_range=(0.1, 2.0),
    #     keep_ratio=True),
    # dict(
    #     type='RandomCrop',
    #     crop_size=image_size,
    #     crop_type='absolute',
    #     recompute_bbox=True,
    #     allow_negative_crop=True),
    # dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackDetInputs')
]

# val_data_pipeline = [
#     dict(
#         type='LoadImageFromFile',
#         to_float32=True,
#         backend_args=None),
#     dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
#     dict(
#         type='LoadPanopticAnnotations',
#         with_bbox=True,
#         with_mask=True,
#         with_seg=True,
#         backend_args=None),
#     dict(type='PackDetInputs')
# ]

dataset = dict(
    ann_file = 'panoptic_anns_coco.json',
    data_prefix = dict(
        img = 'img_dir/', seg = 'panoptic_seg_anns_coco/'),
    filter_cfg = dict(filter_empty_gt=True, min_size=1),
    pipeline = data_pipeline,
    backend_args = None,
    load_embed = True,
)

train_load_parts = ['train']
val_load_parts = ['val']

train_bs = 4
val_bs = 4
