import torch
from models.cls_proposal_v2 import ClsProposalNet

device = torch.device('cuda:0')

pth_load_path = 'logs/cls_proposal_v2/sota-whu/checkpoints/epoch_15.pth'
pth_load_path_new = 'logs/cls_proposal_v2/sota-whu/checkpoints/epoch_15_new.pth'
state_dict = torch.load(pth_load_path)
maskdecoder_param = state_dict['mask_decoder']
rename_state_dict = {}
for key,value in maskdecoder_param.items():
    if 'local_conv' in key:
        new_name = key.replace("local_conv", "semantic_module.local_conv")
        rename_state_dict[new_name] = value
    else:
        rename_state_dict[key] = value
merged_dict = {
    'mask_decoder': rename_state_dict
}
torch.save(merged_dict, pth_load_path_new)

model = ClsProposalNet(
            num_classes = 1,
            num_points = [1,0],
            useModule = 'conv'
        ).to(device)
model.eval()
# state_dict = torch.load(pth_load_path_new)
model.load_parameters(pth_load_path_new)

'''
local_conv
semantic_module.local_conv
'''