# model
sam_ckpt = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
semantic_module_depth = 2
use_inner_feat = True
num_classes = 3
num_mask_tokens = 3

# strategy
max_epochs = 30
train_prompt_type = None    # all_bboxes, random_bbox
val_prompt_type = None
# loss_type = 'ce_dice','focal_dice','loss_masks'
search_loss_type = ['ce_dice']
# base_lr = 0.01, 0.005, 0.001, 0.0005, 0.001
search_lr = [0.003, 0.001, 0.0005, 0.0001]
warmup_epoch = 5
gamma = 0.9
dice_param = 0.8
save_each_epoch = False

use_cls_predict = False
split_self_attn = False # only can be true when use_cls_predict = True
