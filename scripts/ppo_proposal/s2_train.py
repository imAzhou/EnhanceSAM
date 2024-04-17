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

from utils.logger import create_logger
from tqdm import tqdm
# from datasets.loveda import LoveDADataset
from datasets.whu_building_dataset import WHUBuildingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--s1_ckpt_path', type=str,
                    help='max points select in one episode')
parser.add_argument('--num_points', type=int,
                    default=3, help='max points select in one episode')
parser.add_argument('--epochs', type=int,
                    default=50, help='total epochs for all training datasets')
parser.add_argument('--K_epochs', type=int,
                    default=20, help='update policy for K epochs in one PPO update')
parser.add_argument('--update_timestep', type=int,
                    default=60, help='update policy every n timesteps')
parser.add_argument('--save_model_freq', type=int,
                    default=3000, help='save model frequency (in num timesteps)')
parser.add_argument('--eps_clip', type=float,
                    default=0.2, help='clip parameter for PPO')
parser.add_argument('--gamma', type=float,
                    default=0.99, help='discount factor')
parser.add_argument('--lr_actor', type=float,
                    default=0.0003, help='learning rate for ppo agent network')
parser.add_argument('--lr_critic', type=float,
                    default=0.001, help='learning rate for ppo agent network')
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
        action_dim = 64,
        state_dim = 64*64,
        lr_actor = args.lr_actor,
        lr_critic = args.lr_critic,
        gamma = args.gamma,
        K_epochs = args.K_epochs,
        eps_clip = args.eps_clip,
        device = device
    )
    ppo_agent.load(args.s1_ckpt_path)
    env = Env(
        image_size = 64,
        patch_size = 8, 
        device = device )

    ppo_agent_s2 = PPOAgent(
        action_dim = 65,
        state_dim = 8*8,
        lr_actor = args.lr_actor,
        lr_critic = args.lr_critic,
        gamma = args.gamma,
        K_epochs = args.K_epochs,
        eps_clip = args.eps_clip,
        device = device
    )
    
    # load datasets
    train_dataset = WHUBuildingDataset(
        data_root = 'datasets/WHU-Building',
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
    
    log_running_reward = 0
    time_step = 0

    # training loop
    for i_episode in range(args.epochs):
        checkpoint_path = f"{pth_save_dir}/{i_episode}.pth"
        for i_batch, sampled_batch in enumerate(tqdm(trainloader, ncols = 70)):
            mask = sampled_batch['mask_np_64']
            if torch.sum(mask) == 0:
                continue
            # state: tensor, shape is (256, 64, 64)
            state = env.reset(sampled_batch)
            for t in range(1, args.num_points+1):
                # select action with policy
                first_action = ppo_agent.select_action(state)
                
                x1,y1,x2,y2 = env.patch_vertex_coords[first_action]
                second_action = ppo_agent_s2.select_action(state[:,y1:y2,x1:x2])
                reward = env.second_step(first_action, second_action)

                # saving reward and is_terminals
                ppo_agent_s2.buffer.rewards.append(reward)
                ppo_agent_s2.buffer.is_terminals.append(True)

                time_step +=1
                log_running_reward += reward

                # update PPO agent
                if time_step % args.update_timestep == 0:
                    ppo_agent_s2.update()
                    ppo_agent.buffer.clear()
                    # log average reward till last episode
                    log_avg_reward = round(log_running_reward / args.update_timestep, 4)
                    result_f.write('{},{},{}\n'.format(i_episode, time_step, log_avg_reward))
                    logger.info(f'Episode: {i_episode}, Timestep: {time_step}, reward: {log_avg_reward}')
                    result_f.flush()

                    log_running_reward = 0

                # save model weights
                if time_step % args.save_model_freq == 0:
                    print("--------------------------------------------------------------------------------------------")
                    print("saving model at : " + checkpoint_path)
                    ppo_agent_s2.save(checkpoint_path)
                    print("model saved")
                    print("Elapsed Time  : ", datetime.now().replace(microsecond=0) - start_time)
                    print("--------------------------------------------------------------------------------------------")


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
python scripts/ppo_proposal/s2_train.py \
    --s1_ckpt_path logs/ppo_proposal/2024_03_16_03_40_59/checkpoints/49.pth
'''
