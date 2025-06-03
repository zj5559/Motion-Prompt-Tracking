import torch
import torchvision.transforms as transforms
from lib.utils import TensorDict
import lib.train.data.processing_utils as prutils
import torch.nn.functional as F
import random
import numpy as np
import cv2,math


def stack_tensors(x):
    if isinstance(x, (list, tuple)) and isinstance(x[0], torch.Tensor):
        return torch.stack(x)
    return x
def cal_occ(bboxes1, bboxes2):
    #[x1,y1,x2,y2]
    #occ rate of bboxes1
    int_ymin = np.maximum(bboxes1[0], bboxes2[0])
    int_xmin = np.maximum(bboxes1[1], bboxes2[1])
    int_ymax = np.minimum(bboxes1[2], bboxes2[2])
    int_xmax = np.minimum(bboxes1[3], bboxes2[3])

    int_h = np.maximum(int_ymax - int_ymin, 0.)
    int_w = np.maximum(int_xmax - int_xmin, 0.)

    # 计算IOU
    int_vol = np.multiply(int_h, int_w)
    vol1 = np.multiply(bboxes1[2] - bboxes1[0], bboxes1[3] - bboxes1[1])
    occ = (int_vol + 1e-8) / (vol1 + 1e-8)
    return occ
def cal_iou(bboxes1, bboxes2):
    #[x1,y1,w,h]
    #occ rate of bboxes1
    int_ymin = torch.maximum(bboxes1[0], bboxes2[0])
    int_xmin = torch.maximum(bboxes1[1], bboxes2[1])
    int_ymax = torch.minimum(bboxes1[2]+bboxes1[0], bboxes2[2]+bboxes2[0])
    int_xmax = torch.minimum(bboxes1[3]+bboxes1[1], bboxes2[3]+bboxes2[1])

    int_h = torch.maximum(int_ymax - int_ymin, torch.tensor(0.0))
    int_w = torch.maximum(int_xmax - int_xmin, torch.tensor(0.0))

    # 计算IOU
    int_vol = torch.multiply(int_h, int_w)
    vol1 = torch.multiply(bboxes1[2], bboxes1[3])
    vol2 = torch.multiply(bboxes2[2], bboxes2[3])
    iou = (int_vol + 1e-8) / (vol1 + vol2 - int_vol + 1e-8)

    return iou
def cal_iou_batch(bboxes1, bboxes2):
    #[x1,y1,w,h]
    #occ rate of bboxes1
    int_ymin = torch.maximum(bboxes1[:,0], bboxes2[:,0])
    int_xmin = torch.maximum(bboxes1[:,1], bboxes2[:,1])
    int_ymax = torch.minimum(bboxes1[:,2]+bboxes1[:,0], bboxes2[:,2]+bboxes2[:,0])
    int_xmax = torch.minimum(bboxes1[:,3]+bboxes1[:,1], bboxes2[:,3]+bboxes2[:,1])

    int_h = torch.maximum(int_ymax - int_ymin, torch.tensor(0.0))
    int_w = torch.maximum(int_xmax - int_xmin, torch.tensor(0.0))

    # 计算IOU
    int_vol = torch.multiply(int_h, int_w)
    vol1 = torch.multiply(bboxes1[:,2], bboxes1[:,3])
    vol2 = torch.multiply(bboxes2[:,2], bboxes2[:,3])
    iou = (int_vol + 1e-8) / (vol1 + vol2 - int_vol + 1e-8)

    return iou

class BaseProcessing:
    """ Base class for Processing. Processing class is used to process the data returned by a dataset, before passing it
     through the network. For example, it can be used to crop a search region around the object, apply various data
     augmentations, etc."""
    def __init__(self, transform=transforms.ToTensor(), template_transform=None, search_transform=None, joint_transform=None):
        """
        args:
            transform       - The set of transformations to be applied on the images. Used only if template_transform or
                                search_transform is None.
            template_transform - The set of transformations to be applied on the template images. If None, the 'transform'
                                argument is used instead.
            search_transform  - The set of transformations to be applied on the search images. If None, the 'transform'
                                argument is used instead.
            joint_transform - The set of transformations to be applied 'jointly' on the template and search images.  For
                                example, it can be used to convert both template and search images to grayscale.
        """
        self.transform = {'template': transform if template_transform is None else template_transform,
                          'search':  transform if search_transform is None else search_transform,
                          'joint': joint_transform}

    def __call__(self, data: TensorDict):
        raise NotImplementedError


class STARKProcessing(BaseProcessing):
    """ The processing class used for training LittleBoy. The images are processed in the following way.
    First, the target bounding box is jittered by adding some noise. Next, a square region (called search region )
    centered at the jittered target center, and of area search_area_factor^2 times the area of the jittered box is
    cropped from the image. The reason for jittering the target box is to avoid learning the bias that the target is
    always at the center of the search region. The search region is then resized to a fixed size given by the
    argument output_sz.

    """

    def __init__(self, search_area_factor, output_sz, center_jitter_factor, scale_jitter_factor,
                 bbox_center_jitter_factor, bbox_scale_jitter_factor,traj_jitter_prob,traj_filter,
                 tracklen=30, cutmix_prob=0, cutmix_occ=0.3,
                 mode='pair', settings=None, *args, **kwargs):
        """
        args:
            search_area_factor - The size of the search region  relative to the target size.
            output_sz - An integer, denoting the size to which the search region is resized. The search region is always
                        square.
            center_jitter_factor - A dict containing the amount of jittering to be applied to the target center before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            scale_jitter_factor - A dict containing the amount of jittering to be applied to the target size before
                                    extracting the search region. See _get_jittered_box for how the jittering is done.
            mode - Either 'pair' or 'sequence'. If mode='sequence', then output has an extra dimension for frames
        """
        super().__init__(*args, **kwargs)
        self.search_area_factor = search_area_factor
        self.output_sz = output_sz
        self.center_jitter_factor = center_jitter_factor
        self.scale_jitter_factor = scale_jitter_factor
        self.mode = mode
        self.settings = settings

        self.bbox_center_jitter_factor = bbox_center_jitter_factor
        self.bbox_scale_jitter_factor = bbox_scale_jitter_factor

        self.tracklen = tracklen
        self.cutmix_prob = cutmix_prob
        self.cutmix_occ = cutmix_occ
        self.traj_jitter_prob=traj_jitter_prob
        self.traj_filter=traj_filter

    def _get_jittered_box(self, box, mode):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """

        jittered_size = box[2:4] * torch.exp(torch.randn(2) * self.scale_jitter_factor[mode])
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(self.center_jitter_factor[mode]).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)

        return torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
    def _get_jittered_box_prompt_single(self, box):
        """ Jitter the input box
        args:
            box - input bounding box
            mode - string 'template' or 'search' indicating template or search data

        returns:
            torch.Tensor - jittered box
        """
        scale_jitter_factor=self.bbox_scale_jitter_factor
        center_jitter_factor=self.bbox_center_jitter_factor
        jittered_size = box[2:4] * torch.exp(torch.randn(2) * scale_jitter_factor)
        max_offset = (jittered_size.prod().sqrt() * torch.tensor(center_jitter_factor).float())
        jittered_center = box[0:2] + 0.5 * box[2:4] + max_offset * (torch.rand(2) - 0.5)
        jitter_box=torch.cat((jittered_center - 0.5 * jittered_size, jittered_size), dim=0)
        # iou = cal_iou(box, jitter_box)
        # print('iou(jit,prompt):',iou)
        # jitter_box[2:] += jitter_box[:2]
        return jitter_box
    def select_pos_rand(self, ori_pos,content_box,occ_thres):
        x1,y1,x2,y2=ori_pos

        x1=int(max(content_box[0],x1))
        y1 = int(max(content_box[1], y1))
        x2 = int(min(content_box[2], x2))
        y2 = int(min(content_box[3], y2))
        w,h=x2-x1,y2-y1
        if content_box[2]-w<content_box[0] or content_box[3]-h<content_box[1]:
            cancel=True
            # print('error1',content_box,w,h)
            return [0,0,0,0],cancel
        xx1 = random.randint(int(content_box[0]), int(content_box[2]-w))
        yy1 = random.randint(int(content_box[1]), int(content_box[3]-h))
        xx2 = xx1+w
        yy2 = yy1+h
        occ_rate = cal_occ([x1,y1,x2,y2], [xx1,yy1,xx2,yy2])
        cancel=False
        ct=0
        while occ_rate>occ_thres:
            if ct>10:
                cancel=True
                break
            xx1 = random.randint(int(content_box[0]), int(content_box[2] - w))
            yy1 = random.randint(int(content_box[1]), int(content_box[3] - h))
            xx2 = xx1 + w
            yy2 = yy1 + h
            occ_rate = cal_occ([x1,y1,x2,y2], [xx1, yy1, xx2, yy2])
            ct+=1
        return [xx1, yy1, xx2, yy2],cancel,occ_rate
    def cutmix_new(self,data,crops,boxes,output_sz,content_box):
        crops = list(crops)
        use_cutmix=False
        occ_rate=0
        for idx in range(len(crops)):
            x1,y1,x2,y2=round(boxes[idx][0].item()*output_sz),round(boxes[idx][1].item()*output_sz),\
                    round((boxes[idx][0].item()+boxes[idx][2].item())*output_sz),\
                        round((boxes[idx][1].item()+boxes[idx][3].item())*output_sz)
            x1=max(x1,0)
            y1=max(y1,0)
            x2=min(x2,output_sz)
            y2 = min(y2, output_sz)
            w=x2-x1
            h=y2-y1
            if w<1 or h<1:
                # print('cutmix: extra out of range')
                continue
            # obj=crops[idx][y1:y2,x1:x2,:].copy()
            bbox_obj,cancel,occ_rate = self.select_pos_rand([x1,y1,x2,y2], content_box[idx],occ_thres=self.cutmix_occ)
            if cancel:
                # print('cutmix: cannot find proper place')
                continue
            # crops[idx][bbox_obj[1]:bbox_obj[3], bbox_obj[0]:bbox_obj[2], :]=obj
            im = data['cutmix_frames'][idx]
            target_bb=data['cutmix_anno'][idx]
            x, y, w, h=target_bb
            h_img, w_img, _ = im.shape
            w = min(x + w, w_img)
            h = min(y + h, h_img)
            x = max(0, x)
            y = max(0, y)
            w -= x
            h -= y
            if w < 1 or h < 1:
                continue
            x, y, w, h = int(x), int(y), math.ceil(w), math.ceil(h)
            obj2=im[y:y + h, x:x + w, :].copy()
            obj2 = cv2.resize(obj2, (bbox_obj[2]-bbox_obj[0],bbox_obj[3]-bbox_obj[1]))
            crops[idx][bbox_obj[1]:bbox_obj[3], bbox_obj[0]:bbox_obj[2], :] = obj2
            use_cutmix=True
        crops = tuple(crops)
        return crops,use_cutmix,occ_rate
    def __call__(self, data: TensorDict):
        """
        args:
            data - The input data, should contain the following fields:
                'template_images', search_images', 'template_anno', 'search_anno'
        returns:
            TensorDict - output data block with following fields:
                'template_images', 'search_images', 'template_anno', 'search_anno', 'test_proposals', 'proposal_iou'
        """
        if 'search_trackdata' in data.keys():
            jitter_traj = False
            traj_ori=data['search_trackdata'].clone()
            traj_len = len(traj_ori)
            if data['search_visible'][-1] and random.random() < self.traj_jitter_prob:
                jitter_traj = True
                traj_jitter = traj_ori
                jitter_idx = random.sample(list(np.arange(traj_len)), random.randint(0, traj_len//2))
                # jitter_idx.append(traj_len - 1)
                jitter_idx.sort()
                for idx in jitter_idx:
                    traj_jitter[idx] = self._get_jittered_box_prompt_single(traj_jitter[idx])
                data['search_trackdata']=traj_jitter
            prompt_iou = cal_iou_batch(data['search_trackdata'], data['search_trackgt'])
            prompt_iou[data['search_trackvisible'].eq(0)]=self.traj_filter
            weight = torch.tensor(np.arange(traj_len)).float()
            weight1 = torch.softmax(weight, dim=0)
            prompt_iou_mean=(weight1*prompt_iou).sum()
            # prompt_iou_mean = prompt_iou.mean()
            # print(prompt_iou.mean(),prompt_iou_mean)
            num_search = len(data['search_anno'])
            data['search_anno'] = data['search_anno'] + [data['search_trackdata'][i] for i in range(len(data['search_trackdata']))]

        # Apply joint transforms
        if self.transform['joint'] is not None:
            data['template_images'], data['template_anno'], data['template_masks'] = self.transform['joint'](
                image=data['template_images'], bbox=data['template_anno'], mask=data['template_masks'])
            data['search_images'], data['search_anno'], data['search_masks'] = self.transform['joint'](
                image=data['search_images'], bbox=data['search_anno'], mask=data['search_masks'], new_roll=False)
        if 'search_trackdata' in data.keys():
            data['search_trackdata'] = data['search_anno'][num_search:]
            data['search_anno'] = data['search_anno'][:num_search]
            if len(data['search_trackdata'])<self.tracklen:
                pre_list=[data['search_trackdata'][0] for i in range(self.tracklen-len(data['search_trackdata']))]
                data['search_trackdata']=pre_list+data['search_trackdata']

        for s in ['template', 'search']:
            assert self.mode == 'sequence' or len(data[s + '_images']) == 1, \
                "In pair mode, num train/test frames must be 1"

            # Add a uniform noise to the center pos
            jittered_anno = [self._get_jittered_box(a, s) for a in data[s + '_anno']]

            # 2021.1.9 Check whether data is valid. Avoid too small bounding boxes
            w, h = torch.stack(jittered_anno, dim=0)[:, 2], torch.stack(jittered_anno, dim=0)[:, 3]

            crop_sz = torch.ceil(torch.sqrt(w * h) * self.search_area_factor[s])
            if (crop_sz < 1).any():
                data['valid'] = False
                # print("Too small box is found. Replace it with new data.")
                return data
            if 'search_trackdata' in data.keys() and (s == 'search'):

                crops, boxes, boxes_traj, att_mask, mask_crops = prutils.motion_jittered_center_crop(data[s + '_images'],jittered_anno,
                                                                                                    data[s + '_anno'],data['search_trackdata'],
                                                                                                    self.search_area_factor[s],
                                                                                                    self.output_sz[s],masks=data[s + '_masks'])
            else:
                crops, boxes, att_mask, mask_crops = prutils.jittered_center_crop(data[s + '_images'], jittered_anno,
                                                                              data[s + '_anno'], self.search_area_factor[s],
                                                                              self.output_sz[s], masks=data[s + '_masks'])
            w, h = torch.stack(boxes, dim=0)[:, 2] * self.output_sz[s], torch.stack(boxes, dim=0)[:, 3] * \
                   self.output_sz[s]
            if (w < 1).any() or (h < 1).any():
                data['valid'] = False
                # print("L2: Too small box is found. Replace it with new data.")
                return data
            use_cutmix = False
            if s == 'search' and data['search_visible'][-1] and random.random() < self.cutmix_prob:
                # if self.prompt_type=='track':
                #     iou = cal_iou(data['search_anno'][-1], data['search_trackdata'][-1])
                # else:
                #     #for traj
                #     iou=prompt_iou.mean()
                if prompt_iou_mean >= self.traj_filter:
                    content_box = prutils.sample_target_content_bbox(data[s + '_images'], jittered_anno,
                                                                     self.search_area_factor[s], self.output_sz[s])
                    crops, use_cutmix, occ_rate = self.cutmix_new(data, crops, boxes, self.output_sz[s], content_box)
                vis = False
                if use_cutmix and vis:
                    import cv2
                    import os
                    bbox_search = [int(boxes[0][0] * self.output_sz[s]), int(boxes[0][1] * self.output_sz[s]), \
                                   int((boxes[0][0] + boxes[0][2]) * self.output_sz[s] - 1),
                                   int((boxes[0][1] + boxes[0][3]) * self.output_sz[s] - 1)]
                    bbox_prompt = [int(boxes_traj[-1][0] * self.output_sz[s]),
                                   int(boxes_traj[-1][1] * self.output_sz[s]), \
                                   int((boxes_traj[-1][0] + boxes_traj[-1][2]) * self.output_sz[s] - 1),
                                   int((boxes_traj[-1][1] + boxes_traj[-1][3]) * self.output_sz[s] - 1)]
                    im = crops[0]
                    im = cv2.rectangle(crops[0], (bbox_search[0], bbox_search[1]), (bbox_search[2], bbox_search[3]),
                                       (0, 255, 0), 1)  # g
                    im = cv2.rectangle(im, (int(content_box[0][0]), int(content_box[0][1])),
                                       (int(content_box[0][2] - 1), int(content_box[0][3] - 1)),
                                       (255, 0, 255), 2)  # g
                    im = cv2.rectangle(im, (bbox_prompt[0], bbox_prompt[1]), (bbox_prompt[2], bbox_prompt[3]),
                                       (0, 255, 255), 2)  # g
                    im = cv2.putText(im, '%.2f' % (prompt_iou[-1].item()), (20, 20), cv2.FONT_HERSHEY_SIMPLEX,
                                     0.8, (0, 255, 255), 1)
                    path_ori = '/media/zj/ssd/models/traj_prompt/vis/cutmix'
                    if not os.path.exists(path_ori):
                        os.makedirs(path_ori)
                    if len(os.listdir(path_ori)) < 500:
                        a = random.randint(0, 10000)
                        path = os.path.join(path_ori, '%d_x.jpg' % (a))
                        im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        cv2.imwrite(path, im)
            # Apply transforms
            if 'search_trackdata' in data.keys() and (s == 'search'):
                num_search = len(boxes)
                boxes = boxes + boxes_traj
                data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                    image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=True)
                data['search_trackdata'] = data['search_anno'][num_search:]
                data['search_anno'] = data['search_anno'][:num_search]
            else:
                data[s + '_images'], data[s + '_anno'], data[s + '_att'], data[s + '_masks'] = self.transform[s](
                    image=crops, bbox=boxes, att=att_mask, mask=mask_crops, joint=False)

            # 2021.1.9 Check whether elements in data[s + '_att'] is all 1
            # Note that type of data[s + '_att'] is tuple, type of ele is torch.tensor
            for ele in data[s + '_att']:
                if (ele == 1).all():
                    data['valid'] = False
                    # print("Values of original attention mask are all one. Replace it with new data.")
                    return data
            # 2021.1.10 more strict conditions: require the donwsampled masks not to be all 1
            for ele in data[s + '_att']:
                feat_size = self.output_sz[s] // 16  # 16 is the backbone stride
                # (1,1,128,128) (1,1,256,256) --> (1,1,8,8) (1,1,16,16)
                mask_down = F.interpolate(ele[None, None].float(), size=feat_size).to(torch.bool)[0]
                if (mask_down == 1).all():
                    data['valid'] = False
                    # print("Values of down-sampled attention mask are all one. "
                    #       "Replace it with new data.")
                    return data

        data['prompt_bbox'] = [data['search_trackdata'][-1]]  # useless
        traj_ori = torch.stack(data['search_trackdata'], 0)
        data['prompt_iou'] = [prompt_iou_mean]
        data['search_trackdata'] = [traj_ori]
        data['use_cutmix'] = use_cutmix
        data['jitter_traj'] = jitter_traj
        data['valid'] = True
        if 'cutmix_frames' in data:
            data.pop('cutmix_frames')
            data.pop('cutmix_anno')
        # if we use copy-and-paste augmentation
        if data["template_masks"] is None or data["search_masks"] is None:
            data["template_masks"] = torch.zeros((1, self.output_sz["template"], self.output_sz["template"]))
            data["search_masks"] = torch.zeros((1, self.output_sz["search"], self.output_sz["search"]))
        # Prepare output
        if self.mode == 'sequence':
            data = data.apply(stack_tensors)
        else:
            data = data.apply(lambda x: x[0] if isinstance(x, list) else x)

        return data
