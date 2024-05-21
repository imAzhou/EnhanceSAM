import torch
import numpy as np
import random
from torch.utils.data import DataLoader
from .building_dataset import BuildingDataset
from .loveda import LoveDADataset
from utils import set_seed

root_dir = dict(
    zucc = '/x22201018/datasets/RemoteSensingDatasets/',
    hz = '/nfs/zly/datasets/'
)
dataset_config = dict(
    whu = 'WHU-Building',
    inria = 'InriaBuildingDataset',
    lovada = 'LoveDA'
)

def gene_loader(
        *,
        server_name,
        data_tag,
        use_aug,
        use_embed,
        train_sample_num,
        train_bs,
        val_bs,
):
    assert server_name in root_dir.keys(), \
        f'the server_name must be in {root_dir.keys()}'
    assert data_tag in dataset_config.keys(), \
        f'the data_tag must be in {dataset_config.keys()}'

    if data_tag in ['whu','inria']:
        dataset_cls = BuildingDataset
    if data_tag == 'loveda':
        dataset_cls = LoveDADataset
    
    data_root_dir = f'{root_dir[server_name]}{dataset_config[data_tag]}'
    
    train_dataset = dataset_cls(
        data_root = data_root_dir,
        mode = 'train',
        use_embed = use_embed,
        use_aug = use_aug,
        train_sample_num = train_sample_num)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = train_bs,
        shuffle = True,
        num_workers = 8,
        drop_last = True,
        # worker_init_fn=worker_init_fn
    )
    val_dataset = dataset_cls(
        data_root = data_root_dir,
        mode = 'val',
        use_embed = use_embed)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = val_bs,
        shuffle = True,
        num_workers = 0,
        drop_last = True,
        # worker_init_fn=worker_init_fn
    )

    metainfo = train_dataset.METAINFO
    
    return train_dataloader,val_dataloader,metainfo

def worker_init_fn(worker_id):
    # 获取全局随机种子
    seed = torch.initial_seed() % 2**32
    worker_seed = worker_id + seed
    set_seed(worker_seed)