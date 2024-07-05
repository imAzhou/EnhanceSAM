from process_datasets.twochannel2coco_panoptic import converter

data_root = '/x22201018/datasets/MedicalDatasets/PanNuke'
'''
python process_datasets/twochannel2coco_panoptic.py \
    --source_folder /x22201018/datasets/MedicalDatasets/PanNuke/Part3/panoptic_binary_seg_anns \
    --images_json_file /x22201018/datasets/MedicalDatasets/PanNuke/Part3/panoptic_binary_anns.json \
    --segmentations_folder /x22201018/datasets/MedicalDatasets/PanNuke/Part3/panoptic_binary_seg_anns_coco \
    --predictions_json_file /x22201018/datasets/MedicalDatasets/PanNuke/Part3/panoptic_binary_anns_coco.json
'''

for type in ['binary', 'origin']:
    for part in ['Part1', 'Part2', 'Part3']:
        prefix = 'panoptic_'
        if type == 'binary_':
            prefix += type
        source_folder = f'{data_root}/{part}/{prefix}seg_anns'
        images_json_file = f'{data_root}/{part}/{prefix}anns.json'
        segmentations_folder = f'{data_root}/{part}/{prefix}seg_anns_coco'
        predictions_json_file = f'{data_root}/{part}/{prefix}anns_coco.json'
        converter(source_folder, images_json_file, segmentations_folder, predictions_json_file)
        