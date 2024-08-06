# model
sam_ckpts = {
    'vit_b': '/x22201018/codes/SAM/checkpoints_sam/sam_vit_b_01ec64.pth',
    'vit_l': '/x22201018/codes/SAM/checkpoints_sam/sam_vit_l_0b3195.pth',
    'vit_h': '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth',
}

sam_type = 'vit_h'
sam_ckpt = sam_ckpts[sam_type]
semantic_module_depth = 2
use_inner_feat = True
num_classes = 1
num_mask_tokens = 1
use_boundary_head = True

# strategy
max_epochs = 20
train_prompt_type = None    # all_bboxes, random_bbox
val_prompt_type = None
# loss_type = 'ce_dice','focal_dice','loss_masks'
search_loss_type = ['loss_masks']
# base_lr = 0.01, 0.005, 0.001, 0.0005, 0.001
search_lr = [0.003, 0.001]
warmup_epoch = 5
gamma = 0.9
dice_param = 0.5
save_each_epoch = False
