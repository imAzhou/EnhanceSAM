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
dataset_config = dict(
    whu = 'WHU-Building',
    inria = 'InriaBuildingDataset',
    loveda = 'LoveDA',
    pannuke = 'PanNuke',
    pannuke_binary = 'PanNuke',
)

def gene_loader(
        *,
        dataset_domain,
        data_tag,
        use_aug,
        use_embed,
        use_inner_feat,
        train_sample_num,
        train_bs,
        val_bs,
        train_load_parts: None,
        val_load_parts: None,
):
    assert dataset_domain in root_dir.keys(), \
        f'the dataset_domain must be in {root_dir.keys()}'
    assert data_tag in dataset_config.keys(), \
        f'the data_tag must be in {dataset_config.keys()}'

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
        drop_last = True,
    )
    val_dataset = dataset_cls(**val_args)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = val_bs,
        shuffle = True,
        num_workers = 16,
        drop_last = True,
    )

    metainfo = train_dataset.METAINFO
    
    return train_dataloader,val_dataloader,metainfo
