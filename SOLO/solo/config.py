# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

from detectron2.config import CfgNode as CN


def add_solo_config(cfg):
    """
    Add config for Solo.
    """
    cfg.MODEL.DEVICE = 'cpu'
    cfg.MODEL.INS_SEG_HEAD = CN()
    cfg.MODEL.INS_SEG_HEAD.NAME = "SemSegFPNHead"
    cfg.MODEL.INS_SEG_HEAD.IN_FEATURES = ["p2", "p3", "p4", "p5"]
    # Label in the semantic segmentation ground truth that is ignored, i.e., no loss is calculated for
    # the correposnding pixel.
    cfg.MODEL.INS_SEG_HEAD.IGNORE_VALUE = 255
    # Number of classes in the semantic segmentation head
    cfg.MODEL.INS_SEG_HEAD.NUM_CLASSES = 20
    # Number of channels in the 3x3 convs inside semantic-FPN heads.
    cfg.MODEL.INS_SEG_HEAD.CONVS_DIM = 128
    # Outputs from semantic-FPN heads are up-scaled to the COMMON_STRIDE stride.
    cfg.MODEL.INS_SEG_HEAD.COMMON_STRIDE = 4
    # Normalization method for the convolution layers. Options: "" (no norm), "GN".
    cfg.MODEL.INS_SEG_HEAD.NORM = "GN"
    cfg.MODEL.INS_SEG_HEAD.LOSS_WEIGHT = 1.0
