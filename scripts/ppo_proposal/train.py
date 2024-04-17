'''

2. 处理全 0 图
3. 加入第二阶段，判断点的准确性
4. 处理 Inria 数据集
5. 数据增强用在分割阶段，以及用在强化学习的第一阶段
6. 分割阶段增加两个不同大小的 conv ，看看分割结果能不能更好
'''
import time
import os
import torch
import argparse
from datetime import datetime
from help_func.tools import set_seed,get_parameter_number
from models.ppo_proposal import Env, PPOAgent
import random
from utils.logger import create_logger
from tqdm import tqdm
# from datasets.loveda import LoveDADataset
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

dataset_name = 'inria'    # whu inria
num_points = 10
K_epochs = 10
epochs = 20
update_timestep = 10 * num_points   # 每 100 张图更新一次 ppo policy

parser.add_argument('--dataset_name', type=str,
                    default=dataset_name, help='max points select in one episode')
parser.add_argument('--num_points', type=int,
                    default=num_points, help='max points select in one episode')
parser.add_argument('--epochs', type=int,
                    default=epochs, help='total epochs for all training datasets')
parser.add_argument('--K_epochs', type=int,
                    default=K_epochs, help='update policy for K epochs in one PPO update')
parser.add_argument('--update_timestep', type=int,
                    default=update_timestep, help='update policy every n timesteps')
parser.add_argument('--eps_clip', type=float,
                    default=0.2, help='clip parameter for PPO')
parser.add_argument('--gamma', type=float,
                    default=0.99, help='discount factor')
parser.add_argument('--lr_actor', type=float,
                    default=0.0003, help='learning rate for ppo agent network')
parser.add_argument('--lr_critic', type=float,
                    default=0.001, help='learning rate for ppo agent network')
parser.add_argument('--reward_thr', type=float,
                    default=0.5, help='learning rate for ppo agent network')
parser.add_argument('--use_normalized', action='store_true', help='If activated, will apply normalization to image embeding')

parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefix is debug')

args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    if(torch.cuda.is_available()):
        torch.cuda.empty_cache()
    record_save_dir = 'logs/ppo_proposal'
    
    ppo_agent = PPOAgent(
        action_dim = 65,
        state_dim = 64*64,
        lr_actor = args.lr_actor,
        lr_critic = args.lr_critic,
        gamma = args.gamma,
        K_epochs = args.K_epochs,
        eps_clip = args.eps_clip,
        device = device
    )
    
    env = Env(
        image_size = 64,
        patch_size = 8,
        reward_thr = args.reward_thr,
        use_normalized = args.use_normalized,
        device = device )
    
    dataset_config = dict(
        whu = 'datasets/WHU-Building',
        inria = 'datasets/InriaBuildingDataset'
    )
    # load datasets
    train_dataset = BuildingDataset(
        # data_root = 'datasets/InriaBuildingDataset',
        data_root = dataset_config[args.dataset_name],
        resize_size = 1024,
        mode = 'train',
        use_embed = True,
        batch_size = 1
    )
    trainloader = DataLoader(
        train_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True)

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
        print(f'{key}: {value}\n')
        config_items.append(f'{key}: {value}\n')
    with open(config_file, 'w') as f:
        f.writelines(config_items)
    # save log file
    logger = create_logger(f'{files_save_dir}/result.log')
    parameter_cnt = get_parameter_number(ppo_agent.policy)
    logger.info(f'网络总更新参数量：{parameter_cnt}')
    logger.info(f'网络更新参数为：')
    for name,parameters in ppo_agent.policy.named_parameters():
        if parameters.requires_grad:
            logger.info(name)     
    # save result for draw line
    result_f_name = f'{files_save_dir}/result.csv'
    result_f = open(result_f_name,"w+")
    result_f.write('episode,timestep,reward\n')
    
    start_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    
    log_running_reward, time_step, right_action = 0,0,0
    # training loop  empty/total: 423/4736
    for i_episode in range(args.epochs):
        checkpoint_path = f"{pth_save_dir}/{i_episode}.pth"
        valid_data = 0
        for i_batch, sampled_batch in enumerate(tqdm(trainloader, ncols = 70)):
        # for i in tqdm(range(4736-423), ncols = 70):
            mask_te_origin = sampled_batch['mask_te_origin'][0]
            foreground_ratio = torch.sum(mask_te_origin) / (512*512)
            if args.dataset_name == 'whu' and foreground_ratio < 0.05:
                continue
            valid_data += 1
            # 如果当前图像没有真值，则跳过
            # mask = sampled_batch['mask_np_64']
            # if torch.sum(mask) == 0 and not args.process_0:
            #     continue
            # state: tensor, shape is (256, 64, 64)
            state = env.reset(sampled_batch['mask_np_64'][0], sampled_batch['img_embed'][0])
            # random_name = random.choice(['7','2','0'])
            # state = env.reset_specify(random_name)
            for t in range(1, args.num_points+1):
                # select action with policy
                action = ppo_agent.select_action(state)
                state, reward, right_flag, done = env.first_step(action, state)
                # 时间步走完了，done 也为 true
                if t == args.num_points:
                    done = True
                # saving reward and is_terminals
                ppo_agent.buffer.rewards.append(reward)
                ppo_agent.buffer.is_terminals.append(done)

                time_step +=1
                right_action += right_flag
                log_running_reward += reward

                # update PPO agent
                if time_step % args.update_timestep == 0:
                    ppo_agent.update()
                    # log average reward till last episode
                    log_avg_reward = round(log_running_reward / args.update_timestep, 4)
                    action_right_ratio = round(right_action / args.update_timestep, 4)
                    result_f.write('{},{},{}\n'.format(i_episode, time_step, action_right_ratio))
                    logger.info(f'Episode: {i_episode}, Timestep: {time_step}, reward: {log_avg_reward}, RightStep: {action_right_ratio}')
                    result_f.flush()

                    log_running_reward = 0
                    right_action = 0

                # break; if the episode is over
                if done:
                    break
        
        print("--------------------------------------------------------------------------------------------")
        print("saving model at : " + checkpoint_path)
        ppo_agent.save(checkpoint_path)
        print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
        print("--------------------------------------------------------------------------------------------")
    
    print(f'data ratio is valid/all: {valid_data}/{len(trainloader)}')
    result_f.close()
    env.close()

    # print total training time
    print("============================================================================================")
    end_time = datetime.now().replace(microsecond=0)
    print("Started training at (GMT) : ", start_time)
    print("Finished training at (GMT) : ", end_time)
    print("Total training time  : ", end_time - start_time)
    print("============================================================================================")

'''
python scripts/ppo_proposal/train.py \
    --use_normalized \
    --reward_thr 0. \
    --debug_mode
'''