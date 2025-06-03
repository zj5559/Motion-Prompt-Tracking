from lib.test.evaluation.environment import EnvSettings

def local_env_settings():
    settings = EnvSettings()

    # Set your local paths here.

    settings.davis_dir = ''
    settings.got10k_lmdb_path = ''
    settings.got10k_path = '/media/zj/4T/Dataset/GOT-10k/GOT-10k'
    settings.got_packed_results_path = ''
    settings.got_reports_path = '/home/zj/tracking/traj_prompt_23/MPT_code/output'
    settings.itb_path = ''
    settings.lasot_extension_subset_path = '/media/zj/T9/Dataset/tracking/LaSOT_extension_subset/'
    settings.lasot_lmdb_path = ''
    settings.lasot_path = '/media/zj/4T/Dataset/LaSOT/dataset/images'
    settings.network_path = '/media/zj/ssd/models/traj_prompt/output/test/networks'    # Where tracking networks are stored.
    settings.nfs_path = ''
    settings.otb_path = ''
    settings.prj_dir = '/home/zj/tracking/traj_prompt_23/MPT_code'
    settings.result_plot_path = '/home/zj/tracking/traj_prompt_23/MPT_code/output/test/result_plots'
    settings.results_path = '/home/zj/tracking/traj_prompt_23/MPT_code/output/test/tracking_results'    # Where to store tracking results
    settings.save_dir = '/media/zj/ssd/models/traj_prompt/output'
    settings.segmentation_path = ''
    settings.tc128_path = ''
    settings.tn_packed_results_path = ''
    settings.tnl2k_path = '/media/zj/T9/Dataset/tracking/lang-data/tnl2k/TEST_TAR'
    settings.tlp_path = ''
    settings.trackingnet_path = '/media/zj/4T/Dataset/TrackingNet/'
    settings.uav_path = '/media/zj/4T/Dataset/UAV123/Dataset_UAV123/UAV123'
    settings.vot18_path = '/mimer/NOBACKUP/groups/alvis_cvl/jie/codes/OSTrack-main/data/vot2018'
    settings.vot22_path = '/mimer/NOBACKUP/groups/alvis_cvl/jie/codes/OSTrack-main/data/vot2022'
    settings.vot_path = '/mimer/NOBACKUP/groups/alvis_cvl/jie/codes/OSTrack-main/data/VOT2019'
    settings.youtubevos_dir = ''

    return settings

