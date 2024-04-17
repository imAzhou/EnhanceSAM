import os
import torch
import argparse
from help_func.tools import set_seed
from models.ppo_proposal import Env, PPOAgent
from PIL import Image, ImageDraw
from tqdm import tqdm
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

dataset_name = 'whu'    # whu inria

parser.add_argument('--dataset_name', type=str,
                    default=dataset_name, help='max points select in one episode')
parser.add_argument('--ppo_ckpt', type=str,
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

parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--device', type=str, default='cuda:0')

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
    log_dir = args.ppo_ckpt.split('/')[0]
    vis_save_dir = f'{record_save_dir}/{log_dir}/action_box_vis'
    os.makedirs(vis_save_dir, exist_ok=True)
    checkpoint_path = f'{record_save_dir}/{args.ppo_ckpt}'
    print("loading network from : " + checkpoint_path)
    ppo_agent.load(checkpoint_path)
    
    scale_ratio = 1024 // 64
    right_action,total_action = 0, 0
    total_reward = 0
    for i_batch, sampled_batch in enumerate(tqdm(valloader, ncols = 70)):
        # image_path = sampled_batch['meta_info']['img_path'][0]
        image_name = sampled_batch['meta_info']['img_name'][0]
        # image = Image.open(image_path)
        image_tensor = sampled_batch['image_tensor'][0]
        image = Image.fromarray(image_tensor.numpy().transpose(1, 2, 0))
        draw = ImageDraw.Draw(image)
        state = env.reset(sampled_batch)
        for t in range(1, args.num_points+1):
            # select action with policy
            action = ppo_agent.select_action(state)
            state, reward, right_flag, done = env.first_step(action, state)
            if action > 0:
                x1,y1,x2,y2 = [t*scale_ratio for t in env.patch_vertex_coords[action-1]]
                border_color = 'green' if right_flag == 1 else 'red'
                draw.rectangle((x1,y1,x2,y2), outline=border_color, width=5)
                draw.text((x1+5, y1-10), f'reward={reward}', fill='white')

            right_action += right_flag
            total_action += 1
            total_reward += reward      
            if done:
                break

        # clear buffer
        ppo_agent.buffer.clear()
        image.save(f'{vis_save_dir}/{image_name}.png')
        
    env.close()
    avg_right_action = round(right_action / total_action, 4)
    avg_test_reward = round(total_reward/total_action, 4)
    print(f"average test right step : {right_action}/{total_action}={avg_right_action}, average reward = {avg_test_reward}\n")
    
                

'''
python scripts/ppo_proposal/vis_action_box.py \
    --ppo_ckpt 2024_03_20_17_15_45/checkpoints/18.pth \
    --dataset_name whu
'''
