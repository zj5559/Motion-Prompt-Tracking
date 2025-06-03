import numpy as np


############## used for visulize eliminated tokens #################
def get_keep_indices(decisions):
    keep_indices = []
    for i in range(3):
        if i == 0:
            keep_indices.append(decisions[i])
        else:
            keep_indices.append(keep_indices[-1][decisions[i]])
    return keep_indices


def gen_masked_tokens(tokens, indices, alpha=0.2):
    # indices = [i for i in range(196) if i not in indices]
    indices = indices[0].astype(int)
    tokens = tokens.copy()
    tokens[indices] = alpha * tokens[indices] + (1 - alpha) * 255
    return tokens


def recover_image(tokens, H, W, Hp, Wp, patch_size):
    # image: (C, 196, 16, 16)
    image = tokens.reshape(Hp, Wp, patch_size, patch_size, 3).swapaxes(1, 2).reshape(H, W, 3)
    return image


def pad_img(img):
    height, width, channels = img.shape
    im_bg = np.ones((height, width + 8, channels)) * 255
    im_bg[0:height, 0:width, :] = img
    return im_bg


def gen_visualization(image, mask_indices, patch_size=16):
    # image [224, 224, 3]
    # mask_indices, list of masked token indices

    # mask mask_indices need to cat
    # mask_indices = mask_indices[::-1]
    num_stages = len(mask_indices)
    for i in range(1, num_stages):
        mask_indices[i] = np.concatenate([mask_indices[i-1], mask_indices[i]], axis=1)

    # keep_indices = get_keep_indices(decisions)
    image = np.asarray(image)
    H, W, C = image.shape
    Hp, Wp = H // patch_size, W // patch_size
    image_tokens = image.reshape(Hp, patch_size, Wp, patch_size, 3).swapaxes(1, 2).reshape(Hp * Wp, patch_size, patch_size, 3)

    stages = [
        recover_image(gen_masked_tokens(image_tokens, mask_indices[i]), H, W, Hp, Wp, patch_size)
        for i in range(num_stages)
    ]
    imgs = [image] + stages
    imgs = [pad_img(img) for img in imgs]
    viz = np.concatenate(imgs, axis=1)
    return viz


import torch
import os
import matplotlib.pyplot as plt
import matplotlib
import math
import cv2
def norm01(data):
    a=(data-np.min(data))/(np.max(data)-np.min(data))
    a=np.uint8(255 * a)
    return a
def vis_attn_maps(img,attns, template_feat_size, save_path, frame_id, last_only=False,fuse_img=False):
    """
    attn: [bs, nh, lens_t+lens_s, lens_t+lens_s] * encoder_layers (e.g. 12)
    if feed forward twice, encoder_layers will be 24
    """
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list("", ["#04686b", "#fcaf7c"])  # plasma
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    attn = attns[0]
    bs, hn, _, _ = attn.shape
    lens_t = template_feat_size * template_feat_size
    lens_s = attn.shape[-1] - lens_t
    search_feat_size = int(math.sqrt(lens_s))
    sz=img.shape[0]

    assert search_feat_size ** 2 == lens_s, "search_feat_size ** 2 must be equal to lens_s"
    # Default: CE_TEMPLATE_RANGE = 'CTR_POINT'
    if template_feat_size == 8:
        index = slice(3, 5)
    elif template_feat_size == 12:
        index = slice(5, 7)
    elif template_feat_size == 7:
        index = slice(3, 4)
    else:
        raise NotImplementedError


    for block_num, attn in enumerate(attns):
        if last_only:
            if block_num not in [11, 23]:
                continue
        if os.path.exists(os.path.join(save_path, f'frame{frame_id}_block{block_num + 1}_attn_weight.png')):
            print(f"-1")
            return
            # if block_num < len(attns)-1:
            #     continue
        attn_t = attn[:, :, :lens_t, lens_t:]
        box_mask_z = torch.zeros([bs, template_feat_size, template_feat_size], device=attn.device)
        box_mask_z[:, index, index] = 1
        box_mask_z = box_mask_z.flatten(1).to(torch.bool)
        box_mask_z = box_mask_z.unsqueeze(1).unsqueeze(-1).expand(-1, attn_t.shape[1], -1, attn_t.shape[-1])
        attn_t = attn_t[box_mask_z]
        attn_t = attn_t.view(bs, hn, -1, lens_s)
        attn_t = attn_t.mean(dim=2).mean(dim=1)  # bs, len_s
        attn_t_plot = attn_t.squeeze(dim=0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        map = cv2.resize(attn_t_plot, (sz, sz))
        map = norm01(map)
        # map = np.uint8(255 * map)
        map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
        if fuse_img:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            map = map * 0.7 + img
        cv2.imwrite(os.path.join(save_path, f'frame{frame_id}_block{block_num+1}_attn_weight.jpg'), map)
        # fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=300)
        # fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
        # ax = fig.add_subplot(111)
        # ax.imshow(attn_t_plot, cmap='plasma', interpolation='nearest')
        # ax.axis('off')
        # plt.savefig(os.path.join(save_path, f'frame{frame_id}_block{block_num+1}_attn_weight.png'))
        # plt.close()

def vis_feat_maps(img,backbone_out, head_score, template_feat_size, save_path, frame_id, yaml_name,fuse_img=False):
    """
    backbone_out: (B,Hz*Wz+Hx*Wx,C)
    head_score: (B,1,Hx,Wx)
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    if 'prompt' in yaml_name:
        plt_save = os.path.join(save_path, f'frame{frame_id}_prompt_head_score.png')
    else:
        plt_save = os.path.join(save_path, f'frame{frame_id}_ori_head_score.png')
    if os.path.exists(plt_save):
        print(f"-1")
        return
    bs, _, c = backbone_out.shape
    lens_t = template_feat_size * template_feat_size
    lens_s = backbone_out.shape[1] - lens_t
    search_feat_size = int(math.sqrt(lens_s))
    sz=img.shape[0]
    assert search_feat_size ** 2 == lens_s, "search_feat_size ** 2 must be equal to lens_s"
    backbone_out_z = backbone_out[:, :lens_t].transpose(-1, -2).reshape(bs, -1, template_feat_size, template_feat_size)
    backbone_out_x = backbone_out[:, -lens_s:].transpose(-1, -2).reshape(bs, -1, search_feat_size, search_feat_size)
    backbone_out_z = backbone_out_z.mean(dim=1).squeeze().cpu().numpy()
    backbone_out_x = backbone_out_x.mean(dim=1).squeeze().cpu().numpy()
    head_score_np = head_score.squeeze().cpu().numpy()

    map = visual_map(head_score_np, sz, img, fuse_img)
    cv2.imwrite(os.path.join(save_path, f'frame{frame_id}_score.jpg'), map)
    map = visual_map(backbone_out_x, sz, img, fuse_img)
    cv2.imwrite(os.path.join(save_path, f'frame{frame_id}_oriFeat.jpg'), map)

def pltshow(pred_map,save_path, image=None):
    plt.figure(2)
    pred_frame = plt.gca()
    from PIL import Image
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        search = Image.fromarray(image)
        pred_map = cv2.resize(pred_map, (image.shape[0], image.shape[1]))
        plt.imshow(search)
        plt.imshow(pred_map, alpha=0.5,cmap='jet', interpolation='nearest')
    else:
        plt.imshow(pred_map, alpha=1.0, cmap='jet', interpolation='nearest')
    # plt.colorbar()
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False)
    pred_frame.spines['bottom'].set_visible(False)
    pred_frame.spines['left'].set_visible(False)
    pred_frame.spines['right'].set_visible(False)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    # plt.show()
    plt.close(2)
def pltshow_all(pred_map,pred_map2,save_path, image=None):
    plt.figure(2)
    pred_frame = plt.gca()
    from PIL import Image
    if image is not None:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        search = Image.fromarray(image)
        pred_map = cv2.resize(pred_map, (image.shape[0], image.shape[1]))
        pred_map2 = cv2.resize(pred_map2, (image.shape[0], image.shape[1]))
        plt.imshow(search)
        plt.imshow(pred_map, alpha=0.7,cmap='jet', interpolation='nearest')
        plt.imshow(pred_map2, alpha=0.7, cmap='jet', interpolation='nearest')
    else:
        plt.imshow(pred_map, alpha=1.0, cmap='jet', interpolation='nearest')
    # plt.colorbar()
    pred_frame.axes.get_yaxis().set_visible(False)
    pred_frame.axes.get_xaxis().set_visible(False)
    pred_frame.spines['top'].set_visible(False)
    pred_frame.spines['bottom'].set_visible(False)
    pred_frame.spines['left'].set_visible(False)
    pred_frame.spines['right'].set_visible(False)
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=150)
    # plt.show()
    plt.close(2)
def vis_attn_maps_prompt_early(img,attns, save_path, frame_id, type=[0],fuse_img=False):
    """
    attn: [bs, nh, lens_t+lens_s, lens_t+lens_s] * encoder_layers (e.g. 12)
    if feed forward twice, encoder_layers will be 24
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    attn = attns[0]
    bs, hn, token_num,sz  = attn.shape#[1,8,61,256]
    search_feat_size = int(math.sqrt(sz))
    sz=img.shape[0]

    if 0 in type:
        # plot [1,61] average 256 pixels, represent which time step is more salient
        attn_t = attn.mean(dim=[1,2])
        attn_t_plot = attn_t.squeeze(0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        pltshow(attn_t_plot,os.path.join(save_path, f'frame{frame_id}_attn_avg.jpg'), img)
    if 1 in type:
        attn_t = attn.mean(dim=[1])[:,0,:]
        attn_t_plot = attn_t.squeeze(0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        pltshow(attn_t_plot, os.path.join(save_path, f'frame{frame_id}_attn_cls_tl.jpg'), img)

        attn_t = attn.mean(dim=[1])[:, 1, :]
        attn_t_plot = attn_t.squeeze(0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        pltshow(attn_t_plot, os.path.join(save_path, f'frame{frame_id}_attn_cls_br.jpg'), img)
    if 2 in type:
        attn_t = attn.mean(dim=[1])[:,-1,:]
        attn_t_plot = attn_t.squeeze(0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        pltshow(attn_t_plot, os.path.join(save_path, f'frame{frame_id}_attn_last.jpg'), img)

def vis_attn_maps_prompt(img,attns, save_path, frame_id, type=0,fuse_img=False):
    """
    attn: [bs, nh, lens_t+lens_s, lens_t+lens_s] * encoder_layers (e.g. 12)
    if feed forward twice, encoder_layers will be 24
    """

    attn = attns[0]
    if type==0:
        bs, hn, token_num, token_num = attn.shape  # [1,8,62,62]
    elif type==1:
        bs, hn, token_num, sz = attn.shape  # [1,8,62,256]
        search_feat_size = int(math.sqrt(sz))
    elif type==2:
        bs, hn, sz, token_num = attn.shape  # [1,8,256,2]
        search_feat_size = int(math.sqrt(sz))
    elif type==3:
        bs, hn, token_num, sz = attn.shape  # [1,8,256,2]
        search_feat_size = int(math.sqrt(sz))
    sz=img.shape[0]

    if type==0:
        # plot temporal attn
        if not os.path.exists(os.path.join(save_path,'attn0')):
            os.makedirs(os.path.join(save_path,'attn0'))
        attn_t = attn.mean(dim=[1]).squeeze()
        attn_t_plot = attn_t.cpu().numpy()
        pltshow(attn_t_plot, os.path.join(save_path,'attn0', f'frame{frame_id}_attn0.jpg'))
    elif type==1:
        # plot attn1
        if not os.path.exists(os.path.join(save_path,'attn1')):
            os.makedirs(os.path.join(save_path,'attn1'))
        attn_t = attn.mean(dim=[1])[:, 0, :]
        attn_t_plot = attn_t.squeeze(0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        pltshow(attn_t_plot, os.path.join(save_path,'attn1', f'frame{frame_id}_attn1_w.jpg'), img)

        attn_t = attn.mean(dim=[1])[:, 1, :]
        attn_t_plot = attn_t.squeeze(0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        pltshow(attn_t_plot, os.path.join(save_path,'attn1', f'frame{frame_id}_attn1_tl.jpg'), img)

        attn_t = attn.mean(dim=[1])[:, 2, :]
        attn_t_plot = attn_t.squeeze(0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        pltshow(attn_t_plot, os.path.join(save_path, 'attn1', f'frame{frame_id}_attn1_br.jpg'), img)
    elif type == 3:
        # plot attn1
        if not os.path.exists(os.path.join(save_path, 'attn3')):
            os.makedirs(os.path.join(save_path, 'attn3'))
        attn_t1 = attn.mean(dim=[1])[:, 1, :]
        attn_t_plot1 = attn_t1.squeeze(0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        attn_t2 = attn.mean(dim=[1])[:, 2, :]
        attn_t_plot2 = attn_t2.squeeze(0).reshape((search_feat_size, search_feat_size)).cpu().numpy()
        pltshow_all(attn_t_plot1,attn_t_plot1, os.path.join(save_path, 'attn3', f'frame{frame_id}_attn1.jpg'), img)

    elif type==2:
        # plot attn2
        if not os.path.exists(os.path.join(save_path,'attn2')):
            os.makedirs(os.path.join(save_path,'attn2'))
        attn_t = attn.mean(dim=[1,3])
        attn_t_plot = attn_t.squeeze().reshape((search_feat_size, search_feat_size)).cpu().numpy()
        pltshow(attn_t_plot, os.path.join(save_path,'attn2', f'frame{frame_id}_attn2_avg.jpg'), img)

        # attn_t_plot = attn_t[:, :, 1].squeeze().reshape((search_feat_size, search_feat_size)).cpu().numpy()
        # pltshow(attn_t_plot, os.path.join(save_path,'attn2', f'frame{frame_id}_attn2_br.jpg'), img)

def visual_map(map,sz,img,fuse_img):
    map = cv2.resize(map, (sz, sz))
    map = norm01(map)
    map = cv2.applyColorMap(map, cv2.COLORMAP_JET)
    if fuse_img:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        map = map * 0.7 + img
    return map
def vis_feat_maps_prompt(img,backbone_out,vit_score, prompt_score,template_feat_size, save_path, frame_id, yaml_name,fuse_img=False):
    """
    backbone_out: (B,Hz*Wz+Hx*Wx,C)
    head_score: (B,1,Hx,Wx)
    """
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    # if 'prompt' in yaml_name:
    #     plt_save = os.path.join(save_path, f'frame{frame_id}_prompt_head_score.png')
    # else:
    #     plt_save = os.path.join(save_path, f'frame{frame_id}_ori_head_score.png')
    bs, _, c = backbone_out.shape
    lens_t = template_feat_size * template_feat_size
    lens_s = backbone_out.shape[1] - lens_t
    search_feat_size = int(math.sqrt(lens_s))
    sz=img.shape[0]
    assert search_feat_size ** 2 == lens_s, "search_feat_size ** 2 must be equal to lens_s"
    # backbone_out_z = backbone_out[:, :lens_t].transpose(-1, -2).reshape(bs, -1, template_feat_size, template_feat_size)
    # backbone_out_x = backbone_out[:, -lens_s:].transpose(-1, -2).reshape(bs, -1, search_feat_size, search_feat_size)
    # backbone_out_z = backbone_out_z.mean(dim=1).squeeze().cpu().numpy()
    # backbone_out_x = backbone_out_x.mean(dim=1).squeeze().cpu().numpy()
    # backbone_out_x = backbone_out_x[:,10,:,:].squeeze().cpu().numpy()
    # pltshow(backbone_out_x, os.path.join(save_path, f'frame{frame_id}_feat_vit.jpg'), img)
    # prompt_out=prompt_out.transpose(-1, -2).reshape(bs, -1, search_feat_size, search_feat_size).mean(dim=1).squeeze().cpu().numpy()
    # prompt_out = prompt_out.transpose(-1, -2).reshape(bs, -1, search_feat_size, search_feat_size)[:,10,:,:].squeeze().cpu().numpy()
    # pltshow(prompt_out, os.path.join(save_path, f'frame{frame_id}_feat_prompt.jpg'), img)

    map = prompt_score.squeeze().cpu().numpy()
    pltshow(map, os.path.join(save_path, f'frame{frame_id}_score_prompt.jpg'), img)
    map = vit_score.squeeze().cpu().numpy()
    pltshow(map, os.path.join(save_path, f'frame{frame_id}_score_vit.jpg'), img)



    # fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    # fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    # ax = fig.add_subplot(111)
    # ax.imshow(backbone_out_z, cmap='plasma', interpolation='nearest')
    # ax.axis('off')
    # plt.savefig(os.path.join(save_path, f'frame{frame_id}_backbone_out_z.png'))
    #
    # fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=160)
    # fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    # ax = fig.add_subplot(111)
    # ax.imshow(backbone_out_x, cmap='plasma', interpolation='nearest')
    # ax.axis('off')
    # plt.savefig(os.path.join(save_path, f'frame{frame_id}_backbone_out_x.png'))


    # fig = plt.figure(constrained_layout=False, figsize=(5, 5), dpi=300)
    # fig.subplots_adjust(left=0.0, bottom=0.0, right=1.0, top=1.0)
    # ax = fig.add_subplot(111)
    # ax.imshow(head_score_np, cmap='plasma', interpolation='nearest')
    # ax.axis('off')
    # plt.savefig(plt_save)
    # plt.close()

def vis_traj(img,traj,save_path,frame_id):
    num=len(traj)
    start_color=(255,0,0)#red
    end_color=(0,255,0)#green
    sz=img.shape[0]
    for i in range(num):
        color=tuple((start_color[j]*(num-i)+end_color[j]*i)/num for j in range(3))
        x1,y1,w,h=traj[i]*sz
        img=cv2.rectangle(img,(int(x1), int(y1)), (int(x1 + w), int(y1 + h)), color=color, thickness=1)
    return img
