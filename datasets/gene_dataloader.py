import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from .building_dataset import BuildingDataset
from .loveda import LoveDADataset
from .pannuke import PanNukeDataset
from .pannuke_binary import PannukeBinaryDataset

root_dir = dict(
    remote = '/x22201018/datasets/RemoteSensingDatasets/',
    medical = '/x22201018/datasets/MedicalDatasets/'
)
remote_datasets = ['whu', 'inria', 'loveda']
medical_datasets = ['pannuke', 'pannuke_binary']
dataset_config = dict(
    whu = 'WHU-Building',
    inria = 'InriaBuildingDataset',
    loveda = 'LoveDA',
    pannuke = 'PanNuke',
    pannuke_binary = 'PanNuke',
)

def gene_loader(
        *,
        data_tag,
        use_aug,
        use_embed,
        use_inner_feat = False,
        train_sample_num,
        train_bs,
        val_bs,
        train_load_parts = None,
        val_load_parts = None,
):
    assert data_tag in dataset_config.keys(), \
        f'the data_tag must be in {dataset_config.keys()}'
    
    if data_tag in remote_datasets:
        dataset_domain = 'remote'
    if data_tag in medical_datasets:
        dataset_domain = 'medical'

    data_root_dir = f'{root_dir[dataset_domain]}{dataset_config[data_tag]}'
    train_args = dict(
        data_root = data_root_dir,
        mode = 'train',
        use_embed = use_embed,
        use_inner_feat = use_inner_feat,
        use_aug = use_aug,
        train_sample_num = train_sample_num
    )
    val_args = dict(
        data_root = data_root_dir,
        mode = 'val',
        use_embed = use_embed,
        use_inner_feat = use_inner_feat
    )
    
    if data_tag in ['whu','inria']:
        dataset_cls = BuildingDataset
    if data_tag == 'loveda':
        dataset_cls = LoveDADataset
    if data_tag == 'pannuke':
        dataset_cls = PanNukeDataset
        train_args['load_parts'] = train_load_parts
        val_args['load_parts'] = val_load_parts
        del train_args['mode']
        del val_args['mode']
    if data_tag == 'pannuke_binary':
        dataset_cls = PannukeBinaryDataset
        train_args['load_parts'] = train_load_parts
        val_args['load_parts'] = val_load_parts
        del train_args['mode']
        del val_args['mode']
    
    train_dataset = dataset_cls(**train_args)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = train_bs,
        shuffle = True,
        num_workers = 16,
        drop_last = False,
        collate_fn=collate_fn
    )
    val_dataset = dataset_cls(**val_args)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = val_bs,
        shuffle = True,
        num_workers = 16,
        drop_last = False,
        collate_fn=collate_fn
    )

    metainfo = train_dataset.METAINFO
    
    return train_dataloader,val_dataloader,metainfo

def collate_fn(batch_data):
    new_batch_data = {}
    unstack_keys = ['meta_info', 'gt_boxes', 'original_size','input_size']
    all_keys = batch_data[0].keys()
    for key in all_keys:
        new_batch_data[key] = []
        for data in batch_data:
            if key not in unstack_keys:
                data[key] = torch.as_tensor(data[key])
            new_batch_data[key].append(data[key])
        if key not in unstack_keys:
            new_batch_data[key] = torch.stack(new_batch_data[key])
    return new_batch_data
