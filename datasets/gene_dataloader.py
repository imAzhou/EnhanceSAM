from torch.utils.data import DataLoader
from .building_dataset import BuildingDataset
from .loveda import LoveDADataset

dataset_config = dict(
    whu = '/x22201018/datasets/RemoteSensingDatasets/WHU-Building',
    inria = '/x22201018/datasets/RemoteSensingDatasets/InriaBuildingDataset',
    lovada = '/x22201018/datasets/RemoteSensingDatasets/LoveDA'
)

def gene_loader(
        *,
        data_tag,
        use_aug,
        use_embed,
        train_sample_num,
        train_bs,
        val_bs,
):
    assert data_tag in dataset_config.keys(), \
        f'the mode must be in {dataset_config.keys()}'

    if data_tag in ['whu','inria']:
        dataset_cls = BuildingDataset
    if data_tag == 'loveda':
        dataset_cls = LoveDADataset

    train_dataset = dataset_cls(
        data_root = dataset_config[data_tag],
        mode = 'train',
        use_embed = use_embed,
        use_aug = use_aug,
        train_sample_num = train_sample_num)
    train_dataloader = DataLoader(
        train_dataset,
        batch_size = train_bs,
        shuffle = True,
        num_workers = 0,
        drop_last = True)
    val_dataset = dataset_cls(
        data_root = dataset_config[data_tag],
        mode = 'val',
        use_embed = use_embed)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size = val_bs,
        shuffle = True,
        num_workers = 0,
        drop_last = True)

    metainfo = train_dataset.METAINFO
    
    return train_dataloader,val_dataloader,metainfo
