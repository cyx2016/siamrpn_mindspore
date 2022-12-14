""" unique configs """
import numpy as np


class Config:
    """
    Config setup
    """
    # dataset related
    exemplar_size = 127                    # exemplar size
    instance_size = 255                    # instance size 271
    context_amount = 0.5                   # context amount
    sample_type = 'uniform'

    # training related
    exem_stretch = False
    ohem_pos = False
    ohem_neg = False
    ohem_reg = False
    fix_former_3_layers = True
    scale_range = (0.001, 0.7)
    ratio_range = (0.1, 10)
    pairs_per_video_per_epoch = 2          # pairs per video
    train_ratio = 0.99                     # training ratio of VID dataset
    frame_range_vid = 100                  # frame range of choosing the instance
    frame_range_ytb = 1
    train_batch_size = 32                  # training batch size  32
    valid_batch_size = 1                   # validation batch size  8
    train_num_workers = 4                  # number of workers of train dataloader
    valid_num_workers = 4                  # number of workers of validation dataloader
    clip = 10                              # grad clip

    start_lr = 3e-2
    end_lr = 1e-5
    epoch = 50
    lr = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    gamma = np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[1] / \
        np.logspace(np.log10(start_lr), np.log10(end_lr), num=epoch)[0]
    # decay rate of LR_Schedular
    step_size = 1                          # step size of LR_Schedular
    momentum = 0.9                         # momentum of SGD
    weight_decay = 0.0005                  # weight decay of optimizator

    seed = 6666                            # seed to sample training videos
    max_translate = 12                     # max translation of random shift
    scale_resize = 0.15                    # scale step of instance image
    total_stride = 8                       # total stride of backbone
    valid_scope = int((instance_size - exemplar_size) / total_stride / 2)
    anchor_scales = np.array([8,])
    anchor_ratios = np.array([0.33, 0.5, 1, 2, 3])
    anchor_num = len(anchor_scales) * len(anchor_ratios)
    anchor_base_size = 8
    pos_threshold = 0.6
    neg_threshold = 0.3
    num_pos = 16
    num_neg = 48
    lamb = 5
    save_interval = 1
    show_interval = 100
    show_topK = 3
    # tracking related
    gray_ratio = 0.25
    blur_ratio = 0.15
    score_size = int((instance_size - exemplar_size) / 8 + 1)
    penalty_k = 0.22
    window_influence = 0.40
    lr_box = 0.30
    min_scale = 0.1
    max_scale = 10
    sdk_pipeline_name = b"im_siamRPN"
    STREAM_NAME = "im_siamRPN"
    INFER_TIMEOUT = 100000


config = Config()
