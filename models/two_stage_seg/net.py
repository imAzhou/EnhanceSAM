import torch
import torch.nn as nn
from utils import one_hot_encoder, get_prompt
from models.sam.build_sam import sam_model_registry
from .mask_decoder import MaskDecoder
from .transformer import TwoWayTransformer
import torch.nn.functional as F
import numpy as np
from mmdet.structures import DetDataSample

class TwoStageNet(nn.Module):

    def __init__(self, 
                 num_classes = 1,
                 num_mask_tokens = 4,
                 sm_depth = 1,
                 use_inner_feat = False,
                 use_embed = False,
                 split_self_attn = False,
                 use_cls_predict = False,
                 sam_ckpt = None,
                 device = None,
                ):
        '''
        Args:
            - num_classes: determined mask_tokens num,
                -1: sam origin mask decoder, 1: binary segmantation, >1: multi class segmentation.
            - sm_depth: the depth of semantic module in transformer, 0 means don't use this module.
        '''
        super(TwoStageNet, self).__init__()

        self.use_inner_feat = use_inner_feat
        self.use_embed = use_embed
        self.use_cls_predict = use_cls_predict
        self.num_classes = num_classes
        self.num_mask_tokens = num_mask_tokens
        self.device = device
        
        sam = sam_model_registry['vit_h'](checkpoint = sam_ckpt)
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.preprocess = sam.preprocess
        self.postprocess_masks = sam.postprocess_masks

        self.mask_decoder = MaskDecoder(
            num_mask_tokens = num_mask_tokens,
            num_classes = num_classes,
            transformer = TwoWayTransformer(
                sm_depth = sm_depth,
                split_self_attn = split_self_attn,
                num_mask_tokens = num_mask_tokens,
            ),
            use_inner_feat = use_inner_feat,
            use_cls_predict = use_cls_predict,
        )

        self.load_sam_parameters(sam.mask_decoder.state_dict())
        self.freeze_parameters()

    def load_sam_parameters(self, sam_mask_decoder_params: dict):
        
        self_mask_decoder_parmas = self.mask_decoder.state_dict()
        if self.num_classes < 0:
            self_mask_decoder_parmas.update(sam_mask_decoder_params)
        else:
            load_dict = {}
            load_params_from_sam = [
                'transformer', 'output_upscaling', 
            ]
            for name, param in sam_mask_decoder_params.items():
                for key in load_params_from_sam:
                    if key in name:
                        load_dict[name] = param
                        break
            self_mask_decoder_parmas.update(load_dict)

        print('='*10 + ' load parameters from sam ' + '='*10)
        print(self.mask_decoder.load_state_dict(self_mask_decoder_parmas, strict = False))
        # use_weight = sam_mask_decoder_params['mask_tokens.weight'][2].unsqueeze(0)
        # repeat_num = self.num_classes
        # cls_token_init_weight = torch.repeat_interleave(use_weight, repeat_num, dim=0)
        # self.mask_decoder.output_tokens.weight = nn.Parameter(cls_token_init_weight)
        print('='*59)

    def freeze_parameters(self):

        update_param = [
            'semantic_module',
            'process_inner_layers',
            'merge_inner',
            'upscaling_inter_feat',
            'mask_tokens',
            'output_upscaling',
            'output_hypernetworks_mlps',
            'cls_token',
            'cls_prediction_head',
            # 'transformer'
        ]

        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.prompt_encoder.named_parameters():
            param.requires_grad = False
        
        # freeze transformer
        for name, param in self.mask_decoder.named_parameters():
            need_update = False
            for key in update_param:
                if key in name:
                    need_update = True
                    break
            param.requires_grad = need_update
            
    def save_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        merged_dict = {
            'mask_decoder': self.mask_decoder.state_dict()
        }
        torch.save(merged_dict, filename)

    def load_parameters(self, filename: str) -> None:
        assert filename.endswith(".pt") or filename.endswith('.pth')
        state_dict = torch.load(filename)
        print('='*10 + ' load parameters for mask decoder ' + '='*10)
        print(self.mask_decoder.load_state_dict(state_dict['mask_decoder'], strict = False))
        print('='*59)

    def forward_coarse(self, sampled_batch, prompt_type):
        
        bs_image_embedding,inter_feature = self.gene_img_embed(sampled_batch)

        type_to_repeat = dict(
            max_bbox = 2,
            random_bbox = 2,
            random_point = 2,
            max_bbox_center_point = 2,
            max_bbox_with_point = 3,
            all_bboxes = 0
        )
        bs_sparse_embedding = []
        bs_prompt = []
        bs = len(sampled_batch['inputs'])
        input_size,origin_size = None,None
        for idx in range(bs):
            datainfo: DetDataSample = sampled_batch['data_samples'][idx]
            input_size,origin_size = datainfo.img_shape,datainfo.ori_shape
            boxes, points = None, None
            if prompt_type is not None:
                # all_gtboxes.shape: (k, 4)
                all_gtboxes = datainfo.gt_instances.bboxes.tensor.numpy()
                coord_ratio = 1024 // datainfo.img_shape[0]
                boxes, points, _ = get_prompt(prompt_type, None, all_gtboxes, self.device, coord_ratio)

            sparse_embeddings, dense_embeddings = self.prompt_encoder(
                points = points,
                boxes = boxes,
                masks = None,
            )
            if sparse_embeddings.shape[1] == 0 and prompt_type is not None:
                repeat_num = type_to_repeat[prompt_type]
                sparse_embeddings = self.prompt_encoder.not_a_point_embed.weight.unsqueeze(0)
                sparse_embeddings = torch.repeat_interleave(sparse_embeddings, repeat_num, dim=1)
            bs_sparse_embedding.append(sparse_embeddings)
        
            bs_prompt.append(dict(
                points = points,
                boxes = boxes,
            ))
        bs_sparse_embedding = torch.cat(bs_sparse_embedding, dim=0)
        
        image_pe = self.prompt_encoder.get_dense_pe()
        # low_res_masks.shape: (bs, num_cls, 256, 256)
        decoder_outputs = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = bs_sparse_embedding,
            dense_prompt_embeddings = dense_embeddings,
            inter_feature = inter_feature,
            multimask_output = True
        )
        logits_1024 = F.interpolate(
            decoder_outputs['logits_256'],
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        logits_origin = self.postprocess_masks(logits_1024, input_size, origin_size)
        decoder_outputs['logits_origin'] = logits_origin
        decoder_outputs['prompts'] = bs_prompt
        
        return decoder_outputs
    
    def forward_fine(self, sampled_batch, salient_points):
        '''
        Args:
            - sampled_batch: list(DetDataSample)
            - salient_points: np.array, shape is (bs, k, 2), 2 means x,y, scale in 1024
            - train_level: str, 'inst' or 'img'
        '''
        
        bs_image_embedding,inter_feature = self.gene_img_embed(sampled_batch)

        coords_torch = torch.as_tensor(salient_points, dtype=torch.float, device=self.device)
        bs,k,_ = coords_torch.shape
        labels_torch = torch.ones((bs,k), dtype=torch.int, device=self.device)
        points = (coords_torch, labels_torch)

        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points = points,
            boxes = None,
            masks = None,
        )
        image_pe = self.prompt_encoder.get_dense_pe()
        # low_res_masks.shape: (bs, mask_tokens, 256, 256)
        decoder_outputs = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings = dense_embeddings,
            inter_feature = inter_feature,
            multimask_output = self.num_classes < 0
        )
        if self.num_classes < 0:
            # choice smallest level mask
            decoder_outputs['logits_256'] = decoder_outputs['logits_256'][:, 0, ::].unsqueeze(1)
            decoder_outputs['iou_pred'] = decoder_outputs['iou_pred'][:, 0].unsqueeze(1)

        logits_1024 = F.interpolate(decoder_outputs['logits_256'], (1024, 1024), mode="bilinear", align_corners=False)
        data_sample = sampled_batch['data_samples'][0]
        input_size,origin_size = data_sample.img_shape,data_sample.ori_shape
        logits_origin = self.postprocess_masks(logits_1024, input_size, origin_size)
        decoder_outputs['logits_origin'] = logits_origin
        return decoder_outputs


    def forward_multi_class(self, sampled_batch, prompt_type, mask_256_logits=None):
        mask_512 = sampled_batch['mask_512'].to(self.device)
        gt_boxes,coord_ratio = None,1
        # one_hot_gt_mask_512.shape: (1, num_cls, h, w)
        one_hot_gt_mask_512 = one_hot_encoder(self.num_classes, mask_512)

        boxes,points,masks = None, None, None
        # 若当前图像没有任何前景包围框，则它应该输出全零图
        target_cls = (torch.ones((1,), dtype=torch.uint8) * self.ignore_idx).to(self.device)
        target_masks = torch.zeros_like(mask_512).to(self.device)

        all_gtboxes, all_gtboxes_clsid, all_gtboxes_gtmasks = [],[],[]
        if 'gt_boxes' in sampled_batch.keys():
            gt_boxes = sampled_batch['gt_boxes'][0]
            coord_ratio = sampled_batch['coord_ratio'][0].item()
            for cls_id, boxes in gt_boxes.items():
                cls_id = int(cls_id)
                choice_boxes, _, sample_idx = get_prompt(prompt_type, None, np.array(boxes), self.device, coord_ratio)
                all_gtboxes.extend(choice_boxes)
                all_gtboxes_clsid.extend([cls_id]*len(choice_boxes))
                gtmasks = torch.repeat_interleave(one_hot_gt_mask_512[:,cls_id, ::], len(choice_boxes), dim=0)
                all_gtboxes_gtmasks.extend(gtmasks)
            
        if len(all_gtboxes) > 0:
            target_cls = torch.as_tensor(all_gtboxes_clsid, device=self.device)
            target_masks = torch.stack(all_gtboxes_gtmasks).to(self.device)
            boxes = torch.stack(all_gtboxes).to(self.device)

        bs_image_embedding,inter_feature = self.gene_img_embed(sampled_batch)
        sparse_embeddings, dense_embeddings = self.prompt_encoder(
            points = points,
            boxes = boxes,
            masks = masks,   # masks.shape: (k_num_boxes or 1, 1, 256, 256)
        )
        image_pe = self.prompt_encoder.get_dense_pe()
        # low_logits.shape: (k_num_boxes or 1, num_cls or 1, 256, 256)
        # cls_logits.shape: (k_num_boxes or 1, num_cls)
        low_logits,cls_logits = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = sparse_embeddings,
            dense_prompt_embeddings = dense_embeddings,
            inter_feature = inter_feature
        )
        logits_512 = F.interpolate(
            low_logits,
            (512, 512),
            mode="bilinear",
            align_corners=False,
        )
        if self.use_multi_mlps:
            # cls_pred.shape: (k_num_boxes or 1, )
            cls_pred = torch.argmax(cls_logits, dim=1)
            cls_low_logits = [logits_512[i, cls_pred[i], ::] for i in range(len(cls_pred))]
            cls_low_logits = torch.stack(cls_low_logits,dim=0).unsqueeze(1)
        else:
            cls_low_logits = logits_512

        outputs = dict(
            pred_mask_logits = cls_low_logits, target_masks = target_masks,
            pred_cls_logits = cls_logits, target_cls = target_cls,
            prompts = dict(
                points = points,
                boxes = boxes,
            )
        )
    
        return outputs

    def gene_img_embed(self, sampled_batch: dict):
        
        if self.use_embed:
            bs_image_embedding = torch.stack(sampled_batch['img_embed']).to(self.device)
            if self.use_inner_feat:
                inter_feature = torch.stack(sampled_batch['inter_feat']).to(self.device)
            else:
                inter_feature = None
        else:
            with torch.no_grad():
                # input_images.shape: [bs, 3, 1024, 1024]
                input_images = self.preprocess(sampled_batch['input']).to(self.device)
                # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
                if self.use_inner_feat:
                    bs_image_embedding,inter_feature = self.image_encoder(input_images, need_inter=True)
                    bs_image_embedding,inter_feature = bs_image_embedding.detach(),inter_feature.detach()
                else:
                    bs_image_embedding = self.image_encoder(input_images, need_inter=False).detach()
                    inter_feature = None

        return bs_image_embedding,inter_feature
    