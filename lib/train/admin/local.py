class EnvironmentSettings:
    def __init__(self):
        self.workspace_dir = '/media/zj/ssd/models/traj_prompt'    # Base directory for saving network checkpoints.
        self.tensorboard_dir = '/media/zj/ssd/models/traj_prompt/output/tensorboard'    # Directory for tensorboard files.
        self.pretrained_networks = '/home/zj/tracking/traj_prompt_23/MPT_code/pretrained_models'
        self.lasot_dir = '/media/zj/T9/Dataset/tracking/lasot/images'
        self.got10k_dir = '/media/zj/T9/Dataset/tracking/got10k/train'
        self.got10k_val_dir = '/media/zj/Samsung_T5/Datasets/GOT-10k/val'
        self.lasot_lmdb_dir = ''
        self.got10k_lmdb_dir = ''
        self.trackingnet_dir = '/media/zj/4T/Dataset/TrackingNet'
        self.trackingnet_lmdb_dir = ''
        self.coco_dir = '/media/zj/4T/Dataset/COCO'
        self.coco_lmdb_dir = ''
        self.lvis_dir = ''
        self.sbd_dir = ''
        self.imagenet_dir = ''
        self.imagenet_lmdb_dir = ''
        self.imagenetdet_dir = ''
        self.ecssd_dir = ''
        self.hkuis_dir = ''
        self.msra10k_dir = ''
        self.davis_dir = ''
        self.youtubevos_dir = ''
        self.lasot_mask_dir = '/mimer/NOBACKUP/groups/alvis_cvl/datasets/VOT/alpha_masks/lasot'
        self.got10k_mask_dir = '/mimer/NOBACKUP/groups/alvis_cvl/datasets/VOT/alpha_masks/got10k/train'
        self.got10k_val_mask_dir = '/mimer/NOBACKUP/groups/alvis_cvl/datasets/VOT/alpha_masks/got10k/val'
        self.trackingnet_mask_dir = '/mimer/NOBACKUP/groups/alvis_cvl/datasets/VOT/alpha_masks/trackingnet'
