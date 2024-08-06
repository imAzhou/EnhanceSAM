from process_datasets.twochannel2coco_panoptic import converter

data_root = '/x22201018/datasets/RemoteSensingDatasets/InriaBuildingDataset'


for mode in ['train_new', 'val_new']:
    save_root_dir = f'{data_root}/{mode}'

    prefix = 'panoptic_'
    
    source_folder = f'{save_root_dir}/{prefix}seg_anns'
    images_json_file = f'{save_root_dir}/{prefix}anns.json'
    segmentations_folder = f'{save_root_dir}/{prefix}seg_anns_coco'
    predictions_json_file = f'{save_root_dir}/{prefix}anns_coco.json'
    converter(source_folder, images_json_file, segmentations_folder, predictions_json_file)
        