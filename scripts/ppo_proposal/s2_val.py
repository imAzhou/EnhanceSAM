import time
import matplotlib.pyplot as plt
import torch
import argparse
from help_func.tools import set_seed
from models.ppo_proposal import Env, PPOAgent

from tqdm import tqdm
# from datasets.loveda import LoveDADataset
from datasets.whu_building_dataset import WHUBuildingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--dir_name', type=str,
                    help='training result save dir')
parser.add_argument('--s1_ckpt_path', type=str,
                    help='max points select in one episode')
parser.add_argument('--num_points', type=int,
                    default=3, help='max points select in one episode')
parser.add_argument('--epochs', type=int,
                    default=50, help='total epochs for all training datasets')
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
    val_dataset = WHUBuildingDataset(
        data_root = 'datasets/WHU-Building',
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

    all_results = []
    pth_save_dir = f'{record_save_dir}/{args.dir_name}/checkpoints'
    for i_episode in range(args.epochs):
        checkpoint_path = f"{pth_save_dir}/{i_episode}.pth"
        print("loading network from : " + checkpoint_path)
        ppo_agent_s2.load(checkpoint_path)
        right_action = 0
        total_action = 0
        for i_batch, sampled_batch in enumerate(tqdm(valloader, ncols = 70)):
            # 如果当前图像没有真值，则跳过
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
            
                right_action += reward
                total_action += 1

            # clear buffer
            ppo_agent.buffer.clear()

        env.close()
        avg_test_reward = right_action / total_action
        avg_test_reward = round(avg_test_reward, 2)
        all_results.append(avg_test_reward)
        print(f"average test right step : {right_action}/{total_action}={avg_test_reward}")
    
    result_save_path = f'{record_save_dir}/{args.dir_name}/val_result.png'
    # 准备数据
    x = range(len(all_results))
    # 绘制折线图
    plt.plot(x, all_results)
    # 设置x轴和y轴的标签
    plt.ylabel('RightStep ratio')
    # 显示图表
    plt.savefig(result_save_path)
                


'''
python scripts/ppo_proposal/s2_val.py \
    --dir_name 2024_03_18_02_04_26 \
    --s1_ckpt_path logs/ppo_proposal/2024_03_16_03_40_59/checkpoints/49.pth
'''
