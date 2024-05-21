import os
import time
from torch.optim.lr_scheduler import LambdaLR
import torch.optim as optim
from utils.logger import create_logger
from utils import get_parameter_number

def get_logger(record_save_dir, model, args, logger_name):
    # set record files
    save_dir_date = time.strftime("%Y_%m_%d_%H_%M_%S", time.localtime())
    prefix = 'debug_' if args.debug_mode else ''
    files_save_dir = f'{record_save_dir}/{prefix}{save_dir_date}'
    os.makedirs(files_save_dir, exist_ok=True)
    pth_save_dir = f'{files_save_dir}/checkpoints'
    os.makedirs(pth_save_dir, exist_ok=True)
    # save config file
    config_file = os.path.join(files_save_dir, 'config.txt')
    config_items = []
    for key, value in args.__dict__.items():
        print(f'{key}: {value}')
        config_items.append(f'{key}: {value}\n')
    with open(config_file, 'w') as f:
        f.writelines(config_items)
    # save log file
    logger = create_logger(f'{files_save_dir}/result.log', logger_name)
    parameter_cnt = get_parameter_number(model)
    logger.info(f'total params: {parameter_cnt}')
    logger.info(f'update params:')
    for name,parameters in model.named_parameters():
        if parameters.requires_grad:
            logger.info(name)
    return logger,files_save_dir

def get_train_strategy(model, args):
    '''slow start & fast decay'''
    def lr_lambda(epoch):
        if epoch < args.warmup_epoch:
            return (epoch + 1) / args.warmup_epoch  # warm up 阶段线性增加
        else:
            return args.gamma ** (epoch-args.warmup_epoch + 1) # warm up 后每个 epoch 除以 2
    
    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr = args.base_lr, betas = (0.9, 0.999), eps = 1e-08, weight_decay=0)
    
    lr_scheduler = LambdaLR(optimizer, lr_lambda)

    return optimizer,lr_scheduler