from datasets.panoptic.create_loader import gene_loader
from mmengine.config import Config


def test_pannuke():
    # pannuke
    config_file = '/x22201018/codes/SAM/EnhanceSAM/configs/datasets/pannuke.py'
    cfg = Config.fromfile(config_file)
    dataset_config = dict(
        load_parts = ['Part1', 'Part2'],
        pure_args = cfg
    )
    dataloader_config = dict(
        batch_size = 2,
        num_workers = 16,
        seed = 1234
    )
    dataloader,metainfo = gene_loader(
        dataset_tag = 'pannuke_binary', # or pannuke_binary
        dataset_config = dataset_config,
        dataloader_config = dataloader_config
    )


    for i_batch, sampled_batch in enumerate(dataloader):
        datainfo = sampled_batch['data_samples'][0]
        print(datainfo.img_path)
        
        if i_batch > 3:
            break

test_pannuke()
