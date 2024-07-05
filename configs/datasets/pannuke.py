data_root = '/x22201018/datasets/MedicalDatasets/PanNuke'
dataset_tag = 'pannuke'
data_pipline = [
    dict(
        type='LoadImageFromFile',
        to_float32=True,
        backend_args=None),
    dict(
        type='LoadPanopticAnnotations',
        with_bbox=True,
        with_mask=True,
        with_seg=True,
        backend_args=None),
    dict(type='Resize', scale=(1024, 1024), keep_ratio=True),
    dict(type='PackDetInputs')
]

dataset = dict(
    ann_file = 'panoptic_anns_coco.json',
    data_prefix = dict(
        img = 'img_dir/', seg = 'panoptic_seg_anns_coco/'),
    filter_cfg = dict(filter_empty_gt=True, min_size=32),
    pipeline = data_pipline,
    backend_args = None
)

train_load_parts = ['Part1', 'Part2']
val_load_parts = ['Part3']

train_bs = 1
val_bs = 1