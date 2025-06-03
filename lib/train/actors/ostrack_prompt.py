from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search
from ...utils.heapmap_utils import generate_heatmap
from ...utils.ce_utils import generate_mask_cond, adjust_keep_rate


class OSTrackActor_prompt(BaseActor):
    """ Actor for training OSTrack models """

    def __init__(self, net, objective, loss_weight, settings, cfg=None):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.cfg = cfg

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data)

        # compute losses
        loss, status = self.compute_losses(out_dict, data)

        return loss, status

    def forward_pass(self, data):
        # currently only support 1 template and 1 search region
        assert len(data['template_images']) == 1
        assert len(data['search_images']) == 1

        template_list = []
        for i in range(self.settings.num_template):
            template_img_i = data['template_images'][i].view(-1,
                                                             *data['template_images'].shape[2:])  # (batch, 3, 128, 128)
            # template_att_i = data['template_att'][i].view(-1, *data['template_att'].shape[2:])  # (batch, 128, 128)
            template_list.append(template_img_i)

        search_img = data['search_images'][0].view(-1, *data['search_images'].shape[2:])  # (batch, 3, 320, 320)
        # search_att = data['search_att'][0].view(-1, *data['search_att'].shape[2:])  # (batch, 320, 320)

        box_mask_z = None
        ce_keep_rate = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            box_mask_z = generate_mask_cond(self.cfg, template_list[0].shape[0], template_list[0].device,
                                            data['template_anno'][0])

            ce_start_epoch = self.cfg.TRAIN.CE_START_EPOCH
            ce_warm_epoch = self.cfg.TRAIN.CE_WARM_EPOCH
            ce_keep_rate = adjust_keep_rate(data['epoch'], warmup_epochs=ce_start_epoch,
                                                total_epochs=ce_start_epoch + ce_warm_epoch,
                                                ITERS_PER_EPOCH=1,
                                                base_keep_rate=self.cfg.MODEL.BACKBONE.CE_KEEP_RATIO[0])

        if len(template_list) == 1:
            template_list = template_list[0]
        if self.cfg.PROMPT.USE_PROMPT:
            pre_traj = data['search_trackdata'].permute(1, 0, 2,3)
            pre_traj = box_xywh_to_xyxy(pre_traj)
            out_dict = self.net(template=template_list,
                                search=search_img,
                                prompt_bbox=pre_traj,
                                ce_template_mask=box_mask_z,
                                ce_keep_rate=ce_keep_rate,
                                return_last_attn=False)
        else:
            out_dict = self.net(template=template_list,
                            search=search_img,
                            ce_template_mask=box_mask_z,
                            ce_keep_rate=ce_keep_rate,
                            return_last_attn=False)

        return out_dict

    def compute_gt_feat(self,gt_dict):
        gt_bbox=gt_dict['search_anno']
        gt_bbox = box_xywh_to_xyxy(gt_bbox).permute(1,0,2)
        box_embeds=self.net.promptEnc._embed_boxes(gt_bbox)
        return box_embeds.detach()


    def compute_losses(self, pred_dict, gt_dict, return_status=True):

        # gt gaussian map
        gt_bbox = gt_dict['search_anno'][-1]  # (Ns, batch, 4) (x1,y1,w,h) -> (batch, 4)
        gt_gaussian_maps = generate_heatmap(gt_dict['search_anno'], self.cfg.DATA.SEARCH.SIZE, self.cfg.MODEL.BACKBONE.STRIDE)
        gt_gaussian_maps = gt_gaussian_maps[-1].unsqueeze(1)

        pred_boxes_prompt = pred_dict['pred_boxes_prompt']
        pred_boxes_vit = pred_dict['pred_boxes_vit']
        if torch.isnan(pred_boxes_prompt).any() or torch.isnan(pred_boxes_vit).any():
            print('here')
            raise ValueError("Network outputs is NAN! Stop Training")

        # Get boxes
        num_queries = pred_boxes_prompt.size(1)
        pred_boxes_vec_prompt = box_cxcywh_to_xyxy(pred_boxes_prompt).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        pred_boxes_vec_vit = box_cxcywh_to_xyxy(pred_boxes_vit).view(-1, 4)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0,
                                                                                                           max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss_prompt, iou_prompt = self.objective['giou'](pred_boxes_vec_prompt, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss_prompt, iou_prompt = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        try:
            giou_loss_vit, iou_vit = self.objective['giou'](pred_boxes_vec_vit, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss_vit, iou_vit = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        try:
            giou_loss_union, iou_union = self.objective['giou'](pred_boxes_vec_vit, pred_boxes_vec_prompt)  # (BN,4) (BN,4)
        except:
            giou_loss_union, iou_union = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()

        # compute l1 loss
        l1_loss_prompt = self.objective['l1'](pred_boxes_vec_prompt, gt_boxes_vec,reduction='none').mean(dim=1)  # (BN,4) (BN,4)
        # compute location loss
        if 'score_map_prompt' in pred_dict:
            location_loss_prompt = self.objective['focal'](pred_dict['score_map_prompt'], gt_gaussian_maps)
            # location_loss_vit = self.objective['focal'](pred_dict['score_map_vit'], gt_gaussian_maps)
        else:
            location_loss_prompt = torch.tensor(0.0, device=l1_loss_prompt.device)
            # location_loss_vit = torch.tensor(0.0, device=l1_loss_prompt.device)

        # compute max_idx_dist(pred_score,gt_score)
        dist_prompt, idx_acc_prompt = self.cal_score_dist(pred_dict['score_map_prompt'], gt_gaussian_maps)
        dist_vit, idx_acc_vit = self.cal_score_dist(pred_dict['score_map_vit'], gt_gaussian_maps)

        # prompt iou loss
        if self.cfg.PROMPT.token_loss:
            # gt_cls_tokens = self.compute_gt_feat(gt_dict)
            token_bbox=pred_dict['token_feats']
            try:
                giou_loss_token, iou_token = self.objective['giou'](token_bbox, gt_boxes_vec)
            except:
                giou_loss_token, iou_token = torch.zeros(token_bbox.shape[0]).cuda(), torch.zeros(token_bbox.shape[0]).cuda()
            l1_loss_token = self.objective['l1'](token_bbox, gt_boxes_vec)
            traj_iou_loss = (self.loss_weight['giou'] * giou_loss_token).mean() + self.loss_weight['l1'] * l1_loss_token
            if self.cfg.PROMPT.DEC_TYPE=='rep_token_weight':
                gt_conf_label = iou_token.detach()
                prompt_conf_label = pred_dict['conf_pred'].squeeze()
                iou_loss= self.objective['prompt'](prompt_conf_label, gt_conf_label)
                traj_iou_loss+=iou_loss
                r_ca, r_cb, r_ba = self.corr(gt_dict['prompt_iou'].squeeze().detach().cpu().numpy(),
                                             prompt_conf_label.squeeze().detach().cpu().numpy(),
                                             gt_conf_label.squeeze().cpu().numpy())
            else:
                r_ca=0
                r_cb = 0
                r_ba = 0
        elif 'conf_pred' in pred_dict:
            # iou_prompt[gt_dict['search_visible'].squeeze().eq(0)]=1.0#0.5
            # iou_vit[gt_dict['search_visible'].squeeze().eq(0)] = 0.5
            if self.cfg.PROMPT.WEIGHT_LABEL_TYPE == 'traj_iou':
                gt_conf_label = gt_dict['prompt_iou'].squeeze()
            elif self.cfg.PROMPT.WEIGHT_LABEL_TYPE == 'pred_iou':
                gt_conf_label = iou_prompt.detach()
            elif self.cfg.PROMPT.WEIGHT_LABEL_TYPE == 'fusion':
                gt_conf_label = gt_dict['prompt_iou'].squeeze() * iou_prompt.detach()

            prompt_conf_label = pred_dict['conf_pred'].squeeze()
            if self.objective['prompt'] is not None:
                # print('shape:',prompt_conf_label.shape,gt_conf_label.shape)
                # print('shape:',prompt_conf_label.shape,gt_conf_label.shape)
                traj_iou_loss = self.objective['prompt'](prompt_conf_label, gt_conf_label)
            else:
                traj_iou_loss = torch.tensor(0.0, device=l1_loss_prompt.device)

            r_ca, r_cb, r_ba=self.corr(gt_dict['prompt_iou'].squeeze().detach().cpu().numpy(),prompt_conf_label.squeeze().detach().cpu().numpy(),
                                       iou_prompt.squeeze().detach().cpu().numpy())
        else:
            traj_iou_loss = torch.tensor(0.0, device=l1_loss_prompt.device)

        # weighted sum
        giou_loss_prompt[((gt_dict['prompt_iou'].squeeze() < self.cfg.PROMPT.traj_filter) | (
            gt_dict['search_visible'].squeeze().eq(0)))] = 0
        l1_loss_prompt[((gt_dict['prompt_iou'].squeeze() < self.cfg.PROMPT.traj_filter) | (
            gt_dict['search_visible'].squeeze().eq(0)))] = 0
        location_loss_prompt[((gt_dict['prompt_iou'].squeeze() < self.cfg.PROMPT.traj_filter) | (
            gt_dict['search_visible'].squeeze().eq(0)))] = 0

        loss = (self.loss_weight['giou'] * giou_loss_prompt + self.loss_weight['l1'] * l1_loss_prompt + \
                self.loss_weight['focal'] * location_loss_prompt).mean()
        loss += self.loss_weight['prompt'] * traj_iou_loss
        if return_status:
            # status for log
            status = {"Loss/total": loss.item(),
                      "Loss/giou_prompt": giou_loss_prompt.mean().item(),
                      "Loss/l1_prompt": l1_loss_prompt.mean().item(),
                      "Loss/location_prompt": location_loss_prompt.mean().item(),
                      "Loss/prompt_loss": traj_iou_loss.item(),
                      "gt/iou_traj": gt_dict['prompt_iou'].detach().mean().item(),
                      "gt/valid_traj": ((gt_dict['prompt_iou'].squeeze() >= self.cfg.PROMPT.traj_filter).sum() / gt_dict['prompt_iou'].squeeze().shape[0]).item(),
                      "gt/use_cutmix": (gt_dict['use_cutmix'].sum() / len(gt_dict['use_cutmix'])).item(),
                      "IoU/iou_prompt": iou_prompt.detach().mean().item(),
                      "IoU/iou_vit": iou_vit.detach().mean().item(),
                      "IoU/iou_union": iou_union.detach().mean().item(),
                      "IoU/iou_gap": (iou_prompt-iou_vit).detach().mean().item()}
            dist_vit = dist_vit.view(-1, num_queries)
            dist_prompt = dist_prompt.view(-1, num_queries)
            status['score_dist/dist_vit'] = dist_vit[:, 0].mean().item()
            status['score_dist/dist_prompt'] = dist_prompt[:, 0].mean().item()
            status['acc/score_idx_acc_vit'] = (idx_acc_vit.sum() / len(idx_acc_vit)).item()
            status['acc/score_idx_acc_prompt_all'] = (idx_acc_prompt.sum() / len(idx_acc_prompt)).item()


            if self.cfg.PROMPT.token_loss:
                status['IoU/iou_token'] = iou_token.detach().mean().item()
            status['corr/predIOU_trajIOU'] = r_ca
            status['corr/predIOU_predScore'] = r_cb
            status['corr/trajIOU_predScore'] = r_ba
            return loss, status
        else:
            return loss
    def cal_score_dist(self,pred_score,gt_score):
        _, idx_prompt = torch.max(pred_score.flatten(1), dim=1, keepdim=True)
        feat_sz = pred_score.shape[-1]
        idx_y_prompt = idx_prompt // feat_sz
        idx_x_prompt = idx_prompt % feat_sz

        _, idx_gt = torch.max(gt_score.flatten(1), dim=1, keepdim=True)
        idx_y_gt = idx_gt // feat_sz
        idx_x_gt = idx_gt % feat_sz

        dist=F.pairwise_distance(torch.cat((idx_y_prompt,idx_x_prompt),dim=1),
                                 torch.cat((idx_y_gt,idx_x_gt),dim=1),
                                 p=2)
        acc=(idx_prompt==idx_gt).squeeze()
        return dist,acc
    def corr(self,traj_iou,pred_score,pred_iou):
        import pandas as pd
        df = pd.DataFrame([traj_iou, pred_score, pred_iou], index=['a', 'b', 'c']).T
        r_ca = df.c.corr(df.a)#traj better, pred_box better
        r_cb = df.c.corr(df.b)#pred_box better, pred_score better
        r_ba = df.b.corr(df.a)#traj better, pred_score better
        return r_ca,r_cb,r_ba
