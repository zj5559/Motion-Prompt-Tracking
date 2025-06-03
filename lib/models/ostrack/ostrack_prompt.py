"""
Basic OSTrack model.
"""
import math
import os
from typing import List

import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones

from lib.models.layers.head import build_box_head
from lib.models.ostrack.vit import vit_base_patch16_224
from lib.models.ostrack.vit_ce import vit_large_patch16_224_ce, vit_base_patch16_224_ce
from lib.utils.box_ops import box_xyxy_to_cxcywh
from lib.models.ostrack.prompt import build_promptDecoder_traj,build_promptEncoder_traj
import torch.nn.functional as F

class OSTrack_prompt(nn.Module):
    """ This is the base class for OSTrack """

    def __init__(self, transformer, box_head,promptEnc,promptDec, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()
        self.backbone = transformer
        self.box_head = box_head
        self.promptEnc=promptEnc
        self.promptDec=promptDec

        self.aux_loss = aux_loss
        self.head_type = head_type
        # if head_type == "CORNER" or head_type == "CENTER":
        self.feat_sz_s = int(box_head.feat_sz)
        self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                prompt_bbox=None,
                ce_template_mask=None,
                ce_keep_rate=None,
                return_last_attn=False,
                training=True
                ):
        if len(prompt_bbox.shape)==3:
            prompt_bbox=prompt_bbox.unsqueeze(1)
        x, aux_dict = self.backbone(z=template, x=search,
                                    ce_template_mask=ce_template_mask,
                                    ce_keep_rate=ce_keep_rate,
                                    return_last_attn=return_last_attn, )

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        enc_opt = feat_last[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.promptEnc is not None and prompt_bbox is not None:
            bs, Nq, traj_len, _ = prompt_bbox.shape
            prompt_bbox = prompt_bbox.view(-1, traj_len, 4)
            prompt_feat, traj_pe = self.promptEnc(prompt_bbox)
            x_pe = self.promptEnc.get_dense_pe((self.feat_sz_s, self.feat_sz_s))
            src0 = opt_feat.unsqueeze(1).repeat(1, Nq, 1, 1, 1).view(-1, C, self.feat_sz_s, self.feat_sz_s)
            iou_hs, src, iou_pred = self.promptDec(src0, x_pe.expand(bs*Nq, -1, -1, -1), prompt_feat, traj_pe)
            src = src.transpose(1, 2).view(-1, C, self.feat_sz_s, self.feat_sz_s)
            if 'iou_pred' in iou_pred:
                w_value = iou_pred['iou_pred']
                corner_values=iou_pred['iou_pred']
                opt_feat_new = src
            else:
                w_value = iou_pred['weight_pred']
                corner_values=iou_pred['token_pred']
                attn_w = w_value.unsqueeze(2).unsqueeze(2).expand(-1, C, self.feat_sz_s, self.feat_sz_s)
                #TODO: implement non-linear attn_w
                opt_feat_new = (torch.ones_like(attn_w) - attn_w) * src0 + attn_w * src
            # if isinstance(iou_pred,dict):
            #     attn_w=iou_pred['iou_pred'].unsqueeze(2).unsqueeze(2).expand(-1,C, self.feat_sz_s, self.feat_sz_s)
            #     iou_pred_loss=iou_pred['iou_pred_loss']
            #     w_value=iou_pred['iou_pred']
            # else:
            #     iou_pred_loss = None
            #     w_value = iou_pred
            #     if iou_pred.shape[1]==C:
            #         attn_w = iou_pred.unsqueeze(2).unsqueeze(2).expand(-1, -1, self.feat_sz_s, self.feat_sz_s)
            #     else:
            #         attn_w=iou_pred.unsqueeze(2).unsqueeze(2).expand(-1,C, self.feat_sz_s, self.feat_sz_s)

            #
            out_prompt = self.forward_head(opt_feat_new, None)
            if training:
                with torch.no_grad():
                    out_vit = self.forward_head(opt_feat, None)
                out = {'pred_boxes_vit': out_vit['pred_boxes'],
                       'pred_boxes_prompt': out_prompt['pred_boxes'],
                       'score_map_vit': out_vit['score_map'],
                       'score_map_prompt': out_prompt['score_map'],
                       'size_map': out_prompt['size_map'],
                       'offset_map': out_prompt['offset_map'],
                       'conf_pred': w_value,
                       'token_feats':corner_values.reshape(bs,-1)
                       }
            else:
                # corner_values=corner_values.reshape(bs,-1)
                # outputs_coord = box_xyxy_to_cxcywh(corner_values)
                # outputs_coord_new = outputs_coord.view(bs, 1, 4)
                # out_prompt1 = {'pred_boxes': outputs_coord_new
                #        }

                out = {'pred_boxes_prompt': out_prompt['pred_boxes'],
                       'score_map_prompt': out_prompt['score_map'],
                       'size_map': out_prompt['size_map'],
                       'offset_map': out_prompt['offset_map'],
                       'conf_pred': w_value,
                       'token_feats': w_value.reshape(bs,-1)
                       }

            # out['conf_pred']=w_value
            # out.update(aux_dict)
            # out['backbone_feat'] = x

        else:
            out = self.forward_head(opt_feat, None)
            out.update(aux_dict)
            out['backbone_feat'] = x
        # out['feat_img']=feat_ori
        # out['feat_pt']=opt_feat.detach().cpu().numpy()
        return out

    def forward_head(self, opt_feat, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        bs, _, _, _ = opt_feat.size()
        Nq = 1
        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError




def build_ostrack_traj(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''
    print('current_path:',pretrained)

    if cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224':
        backbone = vit_base_patch16_224(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE)
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_base_patch16_224_ce':
        backbone = vit_base_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                           ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                           ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                           )
        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    elif cfg.MODEL.BACKBONE.TYPE == 'vit_large_patch16_224_ce':
        backbone = vit_large_patch16_224_ce(pretrained, drop_path_rate=cfg.TRAIN.DROP_PATH_RATE,
                                            ce_loc=cfg.MODEL.BACKBONE.CE_LOC,
                                            ce_keep_ratio=cfg.MODEL.BACKBONE.CE_KEEP_RATIO,
                                            )

        hidden_dim = backbone.embed_dim
        patch_start_index = 1

    else:
        raise NotImplementedError

    backbone.finetune_track(cfg=cfg, patch_start_index=patch_start_index)

    box_head = build_box_head(cfg, hidden_dim)
    if cfg.PROMPT.USE_PROMPT:
        promptEnc=build_promptEncoder_traj(cfg)
        promptDec = build_promptDecoder_traj(cfg)
    else:
        promptEnc = None
        promptDec = None

    model = OSTrack_prompt(
        backbone,
        box_head,
        promptEnc,
        promptDec,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
        checkpoint = torch.load(pretrained, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
        print('Load pretrained model from: ' + cfg.MODEL.PRETRAIN_FILE)

    return model
