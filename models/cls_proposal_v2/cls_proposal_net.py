import torch
import torch.nn as nn
from utils import gene_point_embed
from models.sam.build_sam import sam_model_registry
from .mask_decoder import MaskDecoder
from .cls_transformer import TwoWayTransformer
import torch.nn.functional as F


class ClsProposalNet(nn.Module):

    def __init__(self, 
                 num_points = [0,0],
                 num_classes = 1,
                 useModule = None,
                 sam_ckpt = '/x22201018/codes/SAM/checkpoints_sam/sam_vit_h_4b8939.pth'
                ):
        super(ClsProposalNet, self).__init__()
        self.num_points = num_points
        self.num_classes = num_classes
        sam = sam_model_registry['vit_h'](checkpoint = sam_ckpt)
        self.image_encoder = sam.image_encoder
        self.prompt_encoder = sam.prompt_encoder
        self.preprocess = sam.preprocess
        
        transformer_dim = sam.mask_decoder.transformer_dim
        self.mask_decoder = MaskDecoder(
            num_classes_outputs = num_classes,
            transformer = TwoWayTransformer(
                depth=2,
                embedding_dim = transformer_dim,
                mlp_dim = 2048,
                num_heads = 8,
                useModule = useModule,
            ),
            transformer_dim = transformer_dim,
            encoder_embed_dim = 1280
        )

        self.except_keys = [
            'cls_tokens', 
            'cls_mlps',
            'process_inter_feat',
            'upscaling_inter_feat',
            'transformer.layers.0.semantic_module',
            'transformer.layers.1.semantic_module',
        ]

        self.load_sam_parameters(sam.mask_decoder.state_dict())
        self.freeze_parameters()

    def load_sam_parameters(self, sam_mask_decoder_params: dict):
        decoder_state_dict = {}
        discard_keys = [
            'output_hypernetworks_mlps',
        ]
        for key,value in sam_mask_decoder_params.items():
            key_first = key.split('.')[0]
            if key_first not in [*self.except_keys, *discard_keys]:
                decoder_state_dict[key] = value

        print('='*10 + ' load parameters from sam ' + '='*10)
        print(self.mask_decoder.load_state_dict(decoder_state_dict, strict = False))
        use_weight = sam_mask_decoder_params['mask_tokens.weight'][2].unsqueeze(0)
        cls_token_init_weight = torch.repeat_interleave(use_weight, self.num_classes, dim=0)
        self.mask_decoder.cls_tokens.weight = nn.Parameter(cls_token_init_weight)
        print('='*59)

    def freeze_parameters(self):
        for name, param in self.image_encoder.named_parameters():
            param.requires_grad = False
        for name, param in self.prompt_encoder.named_parameters():
            param.requires_grad = False
        
        self.except_keys.append('output_upscaling')
        # freeze transformer
        for name, param in self.mask_decoder.named_parameters():
            need_update = False
            for key in self.except_keys:
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
        

    def forward(self, image_batch: torch.Tensor, label_mask: torch.Tensor):
        device = image_batch.device
        with torch.no_grad():
            # input_images.shape: [bs, 3, 1024, 1024]
            input_images = self.preprocess(image_batch.cpu()).to(device)
            # bs_image_embedding.shape: [bs, c=256, h=64, w=64]
            bs_image_embedding,inter_feature = self.image_encoder(input_images, need_inter=True)
            bs_image_embedding,inter_feature = bs_image_embedding.detach(),inter_feature.detach()
        
        sparse,dense,coords = self.gene_prompt_embed(bs_image_embedding,label_mask)

        image_pe = self.prompt_encoder.get_dense_pe()
        low_res_masks = self.mask_decoder(
            image_embeddings = bs_image_embedding,
            image_pe = image_pe,
            sparse_prompt_embeddings = sparse,
            dense_prompt_embeddings = dense,
            inter_feature = inter_feature
        )
        input_sam_size_masks = F.interpolate(
            low_res_masks,
            (1024, 1024),
            mode="bilinear",
            align_corners=False,
        )
        outputs = {
            'pred_mask_512': low_res_masks,
            'pred_mask_1024': input_sam_size_masks,
            'points': coords,
        }

        return outputs
    
    def gene_prompt_embed(self, bs_image_embedding, label_mask):
        device = bs_image_embedding.device
        bs_sparse_embedding = []
        bs_points_coord = []
        for single_label in label_mask:
            sparse_embedding,single_img_points = gene_point_embed(self, single_label.cpu(), self.num_points, device)
            
            # sparse_embedding,_ = gene_center_p_random_n_embed(self, label_mask.cpu())
            # there is no positive point
            if sparse_embedding.shape[1] == 0:
                points_coord = torch.ones((1, sum(self.num_points), 2), device=device) * -1
                sparse_embedding = self.prompt_encoder.not_a_point_embed.weight.unsqueeze(0)
                # when prompt box is none, prompt encoder will concat more point embedding
                sparse_embedding = torch.repeat_interleave(sparse_embedding, sum(self.num_points)+1, dim=1)
            else:
                points_coord, points_label = single_img_points

            bs_sparse_embedding.append(sparse_embedding)
            bs_points_coord.append(points_coord)
        bs_sparse_embedding = torch.cat(bs_sparse_embedding, dim=0)
        bs_points_coord = torch.cat(bs_points_coord, dim=0)
        b,c,h,w = bs_image_embedding.shape
        p = self.prompt_encoder
        dense_embeddings = p.no_mask_embed.weight.reshape(1, -1, 1, 1).expand(
            b, -1, h, w
        )

        return bs_sparse_embedding,dense_embeddings,bs_points_coord
