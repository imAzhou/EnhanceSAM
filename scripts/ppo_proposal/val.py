import time
import matplotlib.pyplot as plt
import torch
import argparse
from help_func.tools import set_seed
from models.ppo_proposal import Env, PPOAgent

from tqdm import tqdm
# from datasets.loveda import LoveDADataset
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()


parser.add_argument('--dataset_name', type=str,
                    default='whu', help='max points select in one episode')
parser.add_argument('--dir_name', type=str,
                    help='training result save dir')
parser.add_argument('--num_points', type=int,
                    default=3, help='max points select in one episode')
parser.add_argument('--epochs', type=int,
                    default=20, help='total epochs for all training datasets')
parser.add_argument('--K_epochs', type=int,
                    default=80, help='update policy for K epochs in one PPO update')
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
parser.add_argument('--debug_mode', action='store_true', help='If activated, log dirname prefis is debug')

args = parser.parse_args()

if __name__ == "__main__":
    
    set_seed(args.seed)
    device = torch.device(args.device)
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
    
    # load datasets
    dataset_config = dict(
        whu = 'datasets/WHU-Building',
        inria = 'datasets/InriaBuildingDataset'
    )
    val_dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
        resize_size = 1024,
        mode = 'val',
        use_embed = True,
        batch_size = 1
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=True,
        num_workers=4,
        drop_last=True)

    max_acc, max_avg_reward, max_epoch = 0, 0, -1
    all_results, all_avg_reward = [],[]
    pth_save_dir = f'{record_save_dir}/{args.dir_name}/checkpoints'
    for i_episode in range(args.epochs):
        checkpoint_path = f"{pth_save_dir}/{i_episode}.pth"
        print("loading network from : " + checkpoint_path)
        right_action,total_action = 0, 0
        total_reward = 0

        ppo_agent.load(checkpoint_path)
        valid_data = 0
        # empty/total: 301/1036
        for i_batch, sampled_batch in enumerate(tqdm(valloader, ncols = 70)):
            mask_te_origin = sampled_batch['mask_te_origin'][0]
            foreground_ratio = torch.sum(mask_te_origin) / (512*512)
            if args.dataset_name == 'whu' and foreground_ratio < 0.05:
                continue
            valid_data += 1
            # state: tensor, shape is (256, 64, 64)
            state = env.reset(sampled_batch['mask_np_64'][0], sampled_batch['img_embed'][0])

            for t in range(1, args.num_points+1):
                # select action with policy
                action = ppo_agent.select_action(state)
                state, reward, right_flag, done = env.first_step(action, state)
            
                right_action += right_flag
                total_action += 1
                total_reward += reward
                
                if done:
                    break

            # clear buffer
            ppo_agent.buffer.clear()

        env.close()
        avg_right_action = right_action / total_action
        avg_right_action = round(avg_right_action, 4)
        avg_test_reward = round(total_reward/total_action, 4)
        if avg_right_action > max_acc:
            max_acc = avg_right_action
            max_avg_reward = avg_test_reward
            max_epoch = i_episode
        all_results.append(avg_right_action)
        all_avg_reward.append(avg_test_reward)
        print(f"average test right step : {right_action}/{total_action}={avg_right_action}, average reward = {avg_test_reward}\n")
        print(f'data ratio is valid/all: {valid_data}/{len(valloader)}')
    
    print(f'max acc: {max_acc}, in epoch {max_epoch}, avg_eward = {max_avg_reward}')
    result_save_path = f'{record_save_dir}/{args.dir_name}/val_result.png'
    # 准备数据
    x = range(len(all_results))
    # 绘制折线图
    plt.plot(x, all_results, label='right_action')
    # plt.plot(x, all_avg_reward, label='avg_reward')
    plt.legend()
    # 设置x轴和y轴的标签
    # plt.ylabel('RightStep ratio')
    # 显示图表
    plt.savefig(result_save_path)
                


'''
python scripts/ppo_proposal/val.py \
    --use_normalized \
    --dir_name 2024_03_24_04_55_39 \
    --dataset_name inria \
    --num_points 10 \
    --reward_thr 0. \
    --epochs 20
'''
