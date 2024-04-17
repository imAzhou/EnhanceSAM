import time
import os
import torch
import argparse
from help_func.build_sam import sam_model_registry
from help_func.tools import set_seed
from models.cls_proposal import ClsProposalNet
from models.ppo_proposal import Env, PPOAgent
from utils.iou_metric import BinaryIoUScore
import matplotlib.pyplot as plt
from utils.visualization import show_mask,show_points
from tqdm import tqdm
import random
from datasets.building_dataset import BuildingDataset
from torch.utils.data import DataLoader

parser = argparse.ArgumentParser()

parser.add_argument('--dataset_name', type=str,
                    default='whu', help='max points select in one episode')
parser.add_argument('--dir_name', type=str)
parser.add_argument('--num_classes', type=int,
                    default=1, help='output channel of network')
parser.add_argument('--num_points', type=int,
                    default=3, help='output channel of network')
parser.add_argument('--max_epochs', type=int,
                    default=12, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int,
                    default=1, help='batch_size per gpu')
parser.add_argument('--img_size', type=int,
                    default=1024, help='input patch size of network input')
parser.add_argument('--seed', type=int,
                    default=1234, help='random seed')
parser.add_argument('--sam_ckpt', type=str, default='checkpoints_sam/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')
parser.add_argument('--ppo_ckpt', type=str, default='checkpoints_sam/sam_vit_h_4b8939.pth',
                    help='Pretrained checkpoint')

parser.add_argument('--device', type=str, default='cuda:0')
parser.add_argument('--visual', action='store_true', help='If activated, the predict mask will be saved')
args = parser.parse_args()

if __name__ == "__main__":
    set_seed(args.seed)
    device = torch.device(args.device)
    record_save_dir = f'logs/cls_proposal_with_ppo/{args.dir_name}'

    ppo_agent = PPOAgent(
        action_dim = 65,
        state_dim = 64*64,
        mode='val',
        device = device
    )
    ppo_agent.load(args.ppo_ckpt)
    env = Env(
        image_size = 64,
        patch_size = 8,
        reward_thr = 0,
        use_normalized = False,
        device = device )
    
    # register model
    sam = sam_model_registry['vit_h'](image_size = args.img_size).to(device)
    model = ClsProposalNet(sam,
                    num_classes = args.num_classes,
                    num_points = args.num_points,
                    sam_ckpt_path = args.sam_ckpt
                ).to(device)
    model.eval()
    
    dataset_config = dict(
        whu = 'datasets/WHU-Building',
        inria = 'datasets/InriaBuildingDataset'
    )
    # load datasets
    val_dataset = BuildingDataset(
        data_root = dataset_config[args.dataset_name],
        resize_size = args.img_size,
        mode = 'val',
        use_embed = True
    )
    valloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True)
    

    all_iou = []
    max_iou,max_epoch = 0,0
    for epoch_num in range(args.max_epochs):
    # for epoch_num in range(10,11):
        avarage_iou = 0.0
        pth_load_path = f'{record_save_dir}/checkpoints/epoch_{epoch_num}.pth'
        model.load_parameters(pth_load_path)
        for i_batch, sampled_batch in enumerate(tqdm(valloader, ncols=70)):
            image_batch, label_batch = sampled_batch['image_tensor'].to(device), sampled_batch['mask_te_1024'].long().to(device)
            gt_origin_masks = sampled_batch['mask_te_origin']
            im_original_size = (sampled_batch['original_size'][0].item(),sampled_batch['original_size'][1].item())
            im_input_size = sampled_batch['input_size']

            bs_image_embedding = sampled_batch['img_embed']
            bs_input_boxes = []
            with torch.no_grad():
                for i, img_emb in enumerate(bs_image_embedding):
                    state = env.reset(sampled_batch['mask_np_64'][i], img_emb)
                    prompt_boxes = []
                    for t in range(1, args.num_points+1):
                        # select action with policy
                        action = ppo_agent.select_action(state)
                        state, reward, right_flag, done = env.first_step(action, state)
                        scale_ratio = 1024 // 64
                        if action > 0:
                            x1,y1,x2,y2 = [t*scale_ratio for t in env.patch_vertex_coords[action-1]]
                            prompt_boxes.append([x1,y1,x2,y2])
                        
                        if done:
                            # 补足 box 框的长度
                            if len(prompt_boxes) > 0:
                                while len(prompt_boxes) < args.num_points:
                                    prompt_boxes.append(random.choice(prompt_boxes))
                            else:
                                prompt_boxes = [[-1,-1,-1,-1] for _ in range(args.num_points)]
                            break

                    # clear buffer
                    ppo_agent.buffer.clear()
                    bs_input_boxes.append(prompt_boxes)
                
            bs_input_boxes = torch.tensor(bs_input_boxes, device=device)
            outputs = model.forward_with_ppo(bs_image_embedding.to(device), bs_input_boxes)
            
            # shape: [input_h, input_w]
            pred_logits = model.postprocess_masks(
                outputs['pred_logits'],
                input_size=im_input_size,
                original_size=im_original_size
            ).squeeze(1)
            pred_logits, _ = torch.max(pred_logits, dim=0)
            # pred_sem_seg.shape: ( H, W)
            pred_mask = (pred_logits>0).detach().cpu()

            iou_score = BinaryIoUScore(pred_mask, gt_origin_masks).item()
            avarage_iou += iou_score

            if args.visual and i_batch % 50 == 0:
                pred_save_dir = f'{record_save_dir}/pred_vis'
                os.makedirs(pred_save_dir, exist_ok=True)
                fig = plt.figure(figsize=(12,6))
                ax_0 = fig.add_subplot(121)
                ax_0.imshow(image_batch[0].permute(1,2,0).cpu().numpy())
                show_mask(label_batch.cpu(), ax_0)
                ax_0.set_title('GT mask')
        
                ax_1 = fig.add_subplot(122)
                ax_1.imshow(image_batch[0].permute(1,2,0).cpu().numpy())
                sam_size_pred_mask = (outputs['pred_logits']>0).detach().cpu()
                show_mask(sam_size_pred_mask.numpy(), ax_1)
                # 当前图有前景区域，坐标都是非负值
                if torch.sum(outputs['points'] < 0) == 0:
                    coords_torch = (outputs['points'][0]).cpu()
                    labels_torch = torch.ones(len(coords_torch))
                    show_points(coords_torch, labels_torch, ax_1)
                title = f'pred mask: mIoU {iou_score:.3f}'
                ax_1.set_title(title)

                plt.tight_layout()
                image_name = sampled_batch['meta_info']['img_name'][0]
                plt.savefig(f'{pred_save_dir}/{image_name}')
                plt.close()
        avarage_iou /= len(valloader)
        print(f'epoch: {epoch_num}, miou: {avarage_iou}')
        all_iou.append(f'epoch: {epoch_num}, miou: {avarage_iou}\n')
        if avarage_iou > max_iou:
            max_iou = avarage_iou
            max_epoch = epoch_num
    # save result file
    config_file = os.path.join(record_save_dir, 'pred_result.txt')
    with open(config_file, 'w') as f:
        f.writelines(all_iou)
        f.write(f'\nmax_iou: {max_iou}, max_epoch: {max_epoch}\n')

'''
python scripts/cls_proposal/use_ppo_proposal_val.py \
    --dir_name 2024_03_25_07_23_59 \
    --ppo_ckpt logs/ppo_proposal/2024_03_20_10_51_02/checkpoints/34.pth \
    --num_points 3 \
    --max_epochs 12 \
    --dataset_name inria 
    
    --device cuda:1 \
    --weight_decay 0.01
'''
