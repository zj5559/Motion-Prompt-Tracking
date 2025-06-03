import math

from lib.models.ostrack import build_ostrack,build_ostrack_traj
from lib.test.tracker.basetracker import BaseTracker
import torch

from lib.test.tracker.vis_utils import gen_visualization
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target,transform_image_to_crop
from lib.utils.box_ops import box_xywh_to_xyxy,box_xyxy_to_cxcywh
# for debug
import cv2
import os
import numpy as np

from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

from lib.test.tracker.vis_utils import vis_attn_maps, vis_feat_maps,vis_attn_maps_prompt,vis_attn_maps_prompt_early, vis_feat_maps_prompt,vis_traj
def cal_iou(bboxes1, bboxes2):
    #[x1,y1,x2,y2]
    #occ rate of bboxes1
    int_ymin = torch.maximum(bboxes1[0], bboxes2[0])
    int_xmin = torch.maximum(bboxes1[1], bboxes2[1])
    int_ymax = torch.minimum(bboxes1[2]+bboxes1[0], bboxes2[2]+bboxes2[0])
    int_xmax = torch.minimum(bboxes1[3]+bboxes1[1], bboxes2[3]+bboxes2[1])

    int_h = torch.maximum(int_ymax - int_ymin, torch.tensor(0.0))
    int_w = torch.maximum(int_xmax - int_xmin, torch.tensor(0.0))

    int_vol = torch.multiply(int_h, int_w)
    vol1 = torch.multiply(bboxes1[2], bboxes1[3])
    vol2 = torch.multiply(bboxes2[2], bboxes2[3])
    iou = (int_vol + 1e-8) / (vol1 + vol2 - int_vol + 1e-8)

    return iou
class OSTrack(BaseTracker):
    def __init__(self, params, dataset_name):
        super(OSTrack, self).__init__(params)
        self.cfg = params.cfg
        if self.cfg.TEST.USE_PROMPT:
            network = build_ostrack_traj(params.cfg, training=False)
        else:
            network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)

        self.network = network.cuda()
        self.network.eval()
        self.preprocessor = Preprocessor()
        self.state = None

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()

        # for debug
        self.debug = params.debug
        self.yaml_name = params.yaml_name
        self.use_visdom = 0
        self.frame_id = 0
        self.vis_traj=1
        self.vis_attn = 1
        self.return_attention = self.vis_attn
        self.vis_feat = 1
        self.save_detail=1
        self.save_vec=[]
        self.fuse_img=True
        self.save_dir = os.path.join("/media/zj/ssd/models/traj_prompt/vis/prompt_sparse5",self.yaml_name)
        if self.debug:
            if self.use_visdom:
                self._init_visdom(None, 1)
        # for save boxes from all queries
        self.save_all_boxes = params.save_all_boxes
        self.z_dict1 = {}

    def initialize(self, image, info: dict,seqname=None,seq_len=100):
        # forward the template once
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, info['init_bbox'], self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(info['init_bbox'], resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
        self.crop_sz = torch.Tensor([self.params.search_size, self.params.search_size])
        # save states
        self.state = info['init_bbox']
        self.frame_id = 0
        self.seq_name = seqname if seqname is not None else 'vis_seq'
        self.seq_len=seq_len
        self.save_dir = os.path.join(self.save_dir, self.seq_name)

        if self.cfg.TEST.USE_PROMPT:
            self.traj = [torch.Tensor(self.state) for i in range(self.cfg.PROMPT.TRACKLEN)]

        if self.save_all_boxes:
            '''save all predicted boxes'''
            all_boxes_save = info['init_bbox'] * self.cfg.MODEL.NUM_OBJECT_QUERIES
            return {"all_boxes": all_boxes_save}

    def track(self, image, info: dict = None):
        H, W, _ = image.shape
        self.frame_id += 1
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, self.state, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        if self.debug:
            if self.vis_attn:
                if self.cfg.TEST.USE_PROMPT:
                    self.add_attn_hook_prompt()
                    # self.add_attn_cam_prompt()
                else:
                    self.add_attn_hook()
            if self.vis_feat:
                if self.cfg.TEST.USE_PROMPT:
                    self.add_feat_hook_prompt()
                else:
                    self.add_feat_hook()

        with torch.no_grad():
            x_dict = search
            if self.cfg.TEST.USE_PROMPT:
                box_traj_crop = [
                    transform_image_to_crop(a_gt, torch.Tensor(self.state), resize_factor, self.crop_sz,
                                            normalize=True) for a_gt in self.traj]
                if self.debug:
                    gt = transform_image_to_crop(torch.Tensor(info['gt_bbox']), torch.Tensor(self.state), resize_factor,
                                                 self.crop_sz,
                                                 normalize=False)
                prompt_bbox = torch.stack(box_traj_crop, dim=0)
                prompt_bbox = prompt_bbox.to(x_dict.tensors.device)
                prompt_bbox = box_xywh_to_xyxy(prompt_bbox).unsqueeze(0)
                if self.debug:
                    out_dict = self.network.forward(
                        template=self.z_dict1.tensors, search=x_dict.tensors, prompt_bbox=prompt_bbox,
                        ce_template_mask=self.box_mask_z, return_last_attn=self.return_attention, training=True)
                else:
                    out_dict = self.network.forward(
                        template=self.z_dict1.tensors, search=x_dict.tensors, prompt_bbox=prompt_bbox,
                        ce_template_mask=self.box_mask_z,return_last_attn=self.return_attention,training=False)
            else:
                # merge the template and the search
                # run the transformer
                out_dict = self.network.forward(
                    template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z,
                return_last_attn=self.return_attention)
        if self.cfg.TEST.USE_PROMPT:
            if self.cfg.TEST.POST:
                pred_score_map = out_dict['score_map_prompt']
                response = self.output_window * pred_score_map
                pred_boxes_ori = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
            else:
                pred_boxes_ori=out_dict['pred_boxes_prompt']
        else:
            # add hann windows
            pred_score_map = out_dict['score_map']
            if self.cfg.TEST.POST:
                response = self.output_window * pred_score_map
            else:
                response =pred_score_map
            pred_boxes_ori = self.network.box_head.cal_bbox(response, out_dict['size_map'], out_dict['offset_map'])
        pred_boxes = pred_boxes_ori.view(-1, 4)
        # Baseline: Take the mean of all pred boxes as the final result
        pred_box = (pred_boxes.mean(
            dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]


        # for debug
        if self.debug and self.frame_id % 1 == 0:
            if not self.use_visdom:
                image_BGR = cv2.cvtColor(x_patch_arr, cv2.COLOR_RGB2BGR)
                if self.vis_traj and self.cfg.TEST.USE_PROMPT:
                    save_path = os.path.join(self.save_dir, 'traj')
                    if not os.path.exists(save_path):
                        os.makedirs(save_path)
                    vis_traj(image_BGR.copy(),box_traj_crop,save_path,self.frame_id)
                cx, cy, w, h = (pred_boxes_ori.squeeze() * self.params.search_size).tolist()
                cv2.rectangle(image_BGR, (int(cx-0.5*w), int(cy-0.5*h)), (int(cx+0.5*w), int(cy+0.5*h)), color=(0, 0, 255), thickness=2)

                x1, y1, w, h = gt.tolist()
                cv2.rectangle(image_BGR, (int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=(0, 255, 255),thickness=1)
                save_path = os.path.join(self.save_dir, 'bbox', "%04d.jpg" % self.frame_id)
                if not os.path.exists(os.path.join(self.save_dir,'bbox')):
                    os.makedirs(os.path.join(self.save_dir,'bbox'))
                cv2.imwrite(save_path, image_BGR)


                if self.vis_attn:
                    if self.cfg.TEST.USE_PROMPT:
                        vis_attn_maps_prompt(image_BGR, self.enc_attn_weights0, self.save_dir,
                                      self.frame_id, type=0, fuse_img=self.fuse_img)
                        vis_attn_maps_prompt(image_BGR, self.enc_attn_weights1, self.save_dir,
                                             self.frame_id, type=1, fuse_img=self.fuse_img)
                        vis_attn_maps_prompt(image_BGR, self.enc_attn_weights2, self.save_dir,
                                             self.frame_id, type=2, fuse_img=self.fuse_img)
                    else:
                        vis_attn_maps(x_patch_arr,self.enc_attn_weights, 8, os.path.join(self.save_dir,'attn'), self.frame_id, last_only=True,fuse_img=self.fuse_img)
                if self.vis_feat:
                    if self.cfg.TEST.USE_PROMPT:
                        vis_feat_maps_prompt(image_BGR, self.backbone_feature[-1], self.prompt_feature[-1],out_dict['score_map_vit'],out_dict['score_map_prompt'], 8,
                                      os.path.join(self.save_dir, 'feat'), self.frame_id, self.yaml_name,
                                      fuse_img=self.fuse_img)
                    else:
                        vis_feat_maps(x_patch_arr,self.backbone_feature[-1], self.box_head_score[-1], 8, os.path.join(self.save_dir,'feat'), self.frame_id, self.yaml_name,fuse_img=self.fuse_img)
                if self.save_detail:
                    gt_box = info['gt_bbox']
                    pred_boxes_vit = out_dict['pred_boxes_vit'].view(-1, 4)
                    pred_boxes_vit = (pred_boxes_vit.mean(
                        dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                    pred_boxes_vit = np.array(
                        clip_box(self.map_box_back(pred_boxes_vit, resize_factor), H, W, margin=10))

                    pred_boxes_prompt = out_dict['pred_boxes_prompt'].view(-1, 4)
                    pred_boxes_prompt = (pred_boxes_prompt.mean(
                        dim=0) * self.params.search_size / resize_factor).tolist()  # (cx, cy, w, h) [0,1]
                    pred_boxes_prompt = np.array(
                        clip_box(self.map_box_back(pred_boxes_prompt, resize_factor), H, W, margin=10))
                    iou_vit = cal_iou(torch.tensor(pred_boxes_vit), torch.tensor(gt_box))
                    iou_prompt = cal_iou(torch.tensor(pred_boxes_prompt), torch.tensor(gt_box))
                    w_attn=out_dict['conf_pred'].squeeze().detach().cpu()
                    print(self.seq_len,[self.frame_id,iou_prompt.item(),iou_vit.item(),w_attn.item()])
                    self.save_vec.append([self.frame_id,iou_prompt,iou_vit,w_attn])
                    if self.frame_id==self.seq_len-1:
                        save_vec=np.array(self.save_vec)
                        np.savetxt(os.path.join(self.save_dir,'iou.txt'), save_vec, delimiter='\t', fmt='%.2f')

            else:
                self.visdom.register((image, info['gt_bbox'].tolist(), self.state), 'Tracking', 1, 'Tracking')

                self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
                self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
                self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
                self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

                if 'removed_indexes_s' in out_dict and out_dict['removed_indexes_s']:
                    removed_indexes_s = out_dict['removed_indexes_s']
                    removed_indexes_s = [removed_indexes_s_i.cpu().numpy() for removed_indexes_s_i in removed_indexes_s]
                    masked_search = gen_visualization(x_patch_arr, removed_indexes_s)
                    self.visdom.register(torch.from_numpy(masked_search).permute(2, 0, 1), 'image', 1, 'masked_search')

                while self.pause_mode:
                    if self.step:
                        self.step = False
                        break
        # get the final box result
        self.state = clip_box(self.map_box_back(pred_box, resize_factor), H, W, margin=10)
        if self.cfg.TEST.USE_PROMPT and self.frame_id%self.cfg.TEST.SPARSE==0:
            self.traj.pop(0)
            self.traj.append(torch.Tensor(self.state))
        if self.debug:
            if self.vis_attn:
                self.remove_attn_hook()
            if self.vis_feat:
                self.remove_feat_hook()
        if self.save_all_boxes:
            '''save all predictions'''
            all_boxes = self.map_box_back_batch(pred_boxes * self.params.search_size / resize_factor, resize_factor)
            all_boxes_save = all_boxes.view(-1).tolist()  # (4N, )
            return {"target_bbox": self.state,
                    "all_boxes": all_boxes_save}
        else:
            return {"target_bbox": self.state}
    def add_attn_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        self.attn_hooks = []
        for i in range(12):
            self.attn_hooks.append(self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            ))
        self.enc_attn_weights = enc_attn_weights
    def add_feat_hook(self):
        backbone_feature, box_head_score = [], []
        self.backbone_feature_hooks = []
        self.box_head_score_hooks = []
        self.prompt_feature_hooks=[]
        self.backbone_feature_hooks.append(self.network.backbone.register_forward_hook(
            lambda self, input, output: backbone_feature.append(output[0])
        ))
        self.box_head_score_hooks.append(self.network.box_head.register_forward_hook(
            lambda self, input, output: box_head_score.append(output[0])
        ))
        self.backbone_feature = backbone_feature
        self.box_head_score = box_head_score

    def add_attn_hook_prompt(self):
        enc_attn_weights1, enc_attn_weights2, enc_attn_weights0 = [], [], []
        self.attn_hooks0 = []
        self.attn_hooks1 = []
        self.attn_hooks2 = []

        self.attn_hooks0.append(
            self.network.promptDec.transformer.layers[-1].self_attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights0.append(output[1])
            ))
        self.enc_attn_weights0 = enc_attn_weights0

        self.attn_hooks1.append(
            self.network.promptDec.transformer.layers[-1].cross_attn_token_to_image.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights1.append(output[1])
            ))
        self.enc_attn_weights1 = enc_attn_weights1

        self.attn_hooks2.append(self.network.promptDec.transformer.layers[-1].cross_attn_image_to_token.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights2.append(output[1])
            ))
        self.enc_attn_weights2 = enc_attn_weights2
    def add_attn_hook_prompt_early(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        self.attn_hooks = []
        self.attn_hooks.append(self.network.promptDec.transformer.layers[-1].cross_attn_token_to_image.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            ))

        self.enc_attn_weights = enc_attn_weights
    # def add_attn_cam_prompt(self):
    #     target_layers = [self.network.promptDec.transformer.layers[-1].cross_attn_image_to_token[1]]
    #     self.cam_attn = EigenCAM(self.network, target_layers, use_cuda=True)
    #
    # def vis_attn_cam(self,tensor,img,save_path):
    #     if not os.path.exists(save_path):
    #         os.makedirs(save_path)
    #     grayscale_cam = self.cam_attn(tensor)[0, :, :]
    #     cam_image = show_cam_on_image(img, grayscale_cam, use_rgb=True)
    #     cv2.imwrite(os.path.join(save_path, f'frame{self.frame_id}_attn.jpg'), cam_image)


    def add_feat_hook_prompt(self):
        backbone_feature, prompt_feature,box_head_score = [], [], []
        self.backbone_feature_hooks = []
        self.prompt_feature_hooks = []
        self.box_head_score_hooks = []
        self.backbone_feature_hooks.append(self.network.backbone.register_forward_hook(
            lambda self, input, output: backbone_feature.append(output[0])
        ))
        self.prompt_feature_hooks.append(self.network.promptDec.register_forward_hook(
            lambda self, input, output: prompt_feature.append(output[1])
        ))
        self.box_head_score_hooks.append(self.network.box_head.register_forward_hook(
            lambda self, input, output: box_head_score.append(output[0])
        ))
        self.backbone_feature = backbone_feature
        self.prompt_feature = prompt_feature
        self.box_head_score = box_head_score
    def remove_attn_hook(self):
        for hook in self.attn_hooks0:
            hook.remove()
        self.enc_attn_weights0 = []

        for hook in self.attn_hooks1:
            hook.remove()
        self.enc_attn_weights1 = []

        for hook in self.attn_hooks2:
            hook.remove()
        self.enc_attn_weights2 = []

    def remove_feat_hook(self):
        for hook in self.backbone_feature_hooks:
            hook.remove()
        for hook in self.box_head_score_hooks:
            hook.remove()
        for hook in self.prompt_feature_hooks:
            hook.remove()
        self.backbone_feature = []
        self.box_head_score = []
        self.prompt_feature=[]
    def map_box_back(self, pred_box: list, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return [cx_real - 0.5 * w, cy_real - 0.5 * h, w, h]

    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float):
        cx_prev, cy_prev = self.state[0] + 0.5 * self.state[2], self.state[1] + 0.5 * self.state[3]
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        half_side = 0.5 * self.params.search_size / resize_factor
        cx_real = cx + (cx_prev - half_side)
        cy_real = cy + (cy_prev - half_side)
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)

    def add_hook(self):
        conv_features, enc_attn_weights, dec_attn_weights = [], [], []

        for i in range(12):
            self.network.backbone.blocks[i].attn.register_forward_hook(
                # lambda self, input, output: enc_attn_weights.append(output[1])
                lambda self, input, output: enc_attn_weights.append(output[1])
            )

        self.enc_attn_weights = enc_attn_weights


def get_tracker_class():
    return OSTrack
