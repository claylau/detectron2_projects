# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import numpy as np
import cv2
from scipy.optimize import linear_sum_assignment
from typing import Dict
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, cat
from detectron2.structures import ImageList, Instances, BitMasks, BoxMode, Boxes

from detectron2.modeling.backbone import build_backbone
from detectron2.modeling.postprocessing import sem_seg_postprocess
from detectron2.modeling.meta_arch.build import META_ARCH_REGISTRY

__all__ = ["Solo"]



@META_ARCH_REGISTRY.register()
class Solo(nn.Module):
    """
    Implemention Solo.
    """

    def __init__(self, cfg):
        super().__init__()

        self.device = torch.device(cfg.MODEL.DEVICE)
        self.instance_loss_weight = 1.0
        # build_resnet_fpn_backbone()
        self.backbone = build_backbone(cfg)

        self.sem_seg_head = SemHead(cfg, self.backbone.output_shape())
        self.ins_seg_head = InsHead(cfg, self.backbone.output_shape())

        pixel_mean = torch.Tensor(cfg.MODEL.PIXEL_MEAN).to(self.device).view(-1, 1, 1)
        pixel_std = torch.Tensor(cfg.MODEL.PIXEL_STD).to(self.device).view(-1, 1, 1)
        self.normalizer = lambda x: (x - pixel_mean) / pixel_std

        self.to(self.device)

    def forward(self, batched_inputs):
        """
        Args:
            batched_inputs: a list, batched outputs of :class:`DatasetMapper` .
                Each item in the list contains the inputs for one image.

        For now, each item in the list is a dict that contains:
            image: Tensor, image in (C, H, W) format.
            sem_seg: semantic segmentation ground truth
            Other information that's included in the original dicts, such as:
                "height", "width" (int): the output resolution of the model, used in inference.
                    See :meth:`postprocess` for details.

        Returns:
            list[dict]: Each dict is the output for one input image.
                The dict contains one key "sem_seg" whose value is a
                Tensor of the output resolution that represents the
                per-pixel segmentation prediction.
        """
        images = [x["image"].to(self.device) for x in batched_inputs]
        images = [self.normalizer(x) for x in images]
        # 某些backbone需要保证图像可以被某个整数整除size_divisibility
        images = ImageList.from_tensors(images, self.backbone.size_divisibility)

        # dict(), eg. features["p2"].shape = N, C, H, W
        features = self.backbone(images.tensor)

        if "instances" in batched_inputs[0]:
            gt_instances = [x["instances"].to(self.device) for x in batched_inputs]
            # generate gt_sem_seg from instance annotation for sem_seg.
            gt_sem_seg = []
            for instances_per_image in gt_instances:
                if len(instances_per_image) == 0:
                    continue
                # A tensor of shape (I)
                gt_classes_per_image = instances_per_image.gt_classes.to(dtype=torch.int64)
                gt_masks_per_image = instances_per_image.gt_masks.tensor
                # A tensor of shape (I, H, M), I=#instances in the image; H,W=input image size

                # ignore overlap thing
                ignore = gt_masks_per_image.sum(0) > 1.0
                gt_sem_seg_per_image = gt_masks_per_image * gt_classes_per_image.view(-1, 1, 1)
                gt_sem_seg_per_image.sum(0)[ignore] = self.sem_seg_head.ignore_value
                gt_sem_seg.append(gt_sem_seg_per_image)
            
            gt_sem_seg = ImageList.from_tensors(
                gt_sem_seg, self.backbone.size_divisibility, self.sem_seg_head.ignore_value
            ).tensor
        else:
            gt_instances = None
            gt_sem_seg = None        
        
        sem_seg_results, sem_seg_losses = self.sem_seg_head(features, gt_sem_seg)
        ins_seg_results, ins_score_results, ins_seg_losses = self.ins_seg_head(features, gt_instances)

        if self.training:
            losses = {}
            losses.update(sem_seg_losses)
            losses.update(ins_seg_losses)
            return losses

        processed_results = []
        for sem_seg, ins_seg, ins_score, image_size in zip(
            sem_seg_results, ins_seg_results, ins_score_results, images.image_sizes):
            result = self.postprocess(sem_seg, ins_seg, ins_score, image_size)
            processed_results.append({"instances": result})
        return processed_results
    
    def postprocess(self, sem_seg, ins_seg, ins_score, image_size):
        """
        Args:
            sem_seg (Tensor:[C, H, W]): semantic segmentation of one image.
            ins_seg (Tensor:[B, H, W]): instance segmentation of one image.
            ins_score (Tensor:[B]): instance segmentation confidence of one image.
            image_size (tuple): image size that network is taking as input.
        Return:
            result (Instances): instance result of one image for COCO eavl.
        """
        sem_seg = sem_seg[:, :image_size[0], :image_size[1]]
        ins_seg = ins_seg[:, :image_size[0], :image_size[1]]
        sem_seg = torch.argmax(sem_seg, dim=0)
        ins_seg = (ins_seg >= 0.5).to(dtype=torch.int)
        filter_inds = torch.arange(ins_score.size(0))[ins_score >= 0.5]
        pred_classes = []
        pred_masks = []
        pred_boxes = []
        for i in filter_inds:
            ins_and_sem = ins_seg[i] * sem_seg
            sem_class = ins_and_sem.view(-1).bincount()
            sem_class = torch.argsort(sem_class, descending=True)
            # semantic class
            pred_classes.append(sem_class[0])
            # instance mask
            ins_seg[i][ins_and_sem != sem_class[0]] = 0
            pred_masks.append(ins_seg[i])
            # instance box from mask
            gray_mask = (ins_seg[i]*255).numpy().astype(np.uint8)
            thresh = cv2.Canny(gray_mask, 128, 256)
            _, contours = cv2.findContours(thresh,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
            try:
                x, y, w, h = cv2.boundingRect(contours[0])
                pred_boxes.append(
                    BoxMode.convert([x,y,w,h], BoxMode.XYWH_ABS, BoxMode.XYXY_ABS))
            except:
                pred_boxes.append([0, 0, 0, 0])
        
        result = Instances(image_size)
        result.pred_boxes = Boxes(torch.Tensor(pred_boxes).to(dtype=torch.float))
        result.scores = ins_score[filter_inds]
        result.pred_classes = torch.Tensor(pred_classes)
        result.pred_masks = BitMasks(torch.stack(pred_masks))
        return result


class SemHead(nn.Module):
    """
    Semantic FPN head.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.SEM_SEG_HEAD.IN_FEATURES
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value     = cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE
        num_classes           = cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.SEM_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.SEM_SEG_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.SEM_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.SEM_SEG_HEAD.LOSS_WEIGHT
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        x = self.predictor(x)
        x = F.interpolate(x, scale_factor=self.common_stride, mode="bilinear", align_corners=False)

        if self.training:
            losses = {}
            losses["loss_sem_seg"] = (
                F.cross_entropy(x, targets, reduction="mean", ignore_index=self.ignore_value)
                * self.loss_weight
            )
            return [], losses
        else:
            return x, {}


class InsHead(nn.Module):
    """
    Semantic FPN head.
    """
    def __init__(self, cfg, input_shape: Dict[str, ShapeSpec]):
        super().__init__()

        # fmt: off
        self.in_features      = cfg.MODEL.INS_SEG_HEAD.IN_FEATURES
        feature_strides       = {k: v.stride for k, v in input_shape.items()}
        feature_channels      = {k: v.channels for k, v in input_shape.items()}
        self.ignore_value     = cfg.MODEL.INS_SEG_HEAD.IGNORE_VALUE
        self.num_classes      = cfg.MODEL.INS_SEG_HEAD.NUM_CLASSES
        conv_dims             = cfg.MODEL.INS_SEG_HEAD.CONVS_DIM
        self.common_stride    = cfg.MODEL.INS_SEG_HEAD.COMMON_STRIDE
        norm                  = cfg.MODEL.INS_SEG_HEAD.NORM
        self.loss_weight      = cfg.MODEL.INS_SEG_HEAD.LOSS_WEIGHT
        # fmt: on

        self.scale_heads = []
        for in_feature in self.in_features:
            head_ops = []
            head_length = max(
                1, int(np.log2(feature_strides[in_feature]) - np.log2(self.common_stride))
            )
            for k in range(head_length):
                norm_module = nn.GroupNorm(32, conv_dims) if norm == "GN" else None
                conv = Conv2d(
                    feature_channels[in_feature] if k == 0 else conv_dims,
                    conv_dims,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    bias=not norm,
                    norm=norm_module,
                    activation=F.relu,
                )
                weight_init.c2_msra_fill(conv)
                head_ops.append(conv)
                if feature_strides[in_feature] != self.common_stride:
                    head_ops.append(
                        nn.Upsample(scale_factor=2, mode="bilinear", align_corners=False)
                    )
            self.scale_heads.append(nn.Sequential(*head_ops))
            self.add_module(in_feature, self.scale_heads[-1])
        self.predictor = Conv2d(conv_dims, self.num_classes, kernel_size=1, stride=1, padding=0)
        weight_init.c2_msra_fill(self.predictor)
        self.score = nn.Sequential(
            nn.Conv2d(self.num_classes, self.num_classes, 
                kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_classes, self.num_classes,
                kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_classes, self.num_classes, 
                kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(self.num_classes, self.num_classes,
                kernel_size=3, stride=2, padding=0),
            nn.ReLU(inplace=True))
        for l in self.score:
            if isinstance(l, nn.Conv2d):
                weight_init.c2_msra_fill(l)

    def forward(self, features, targets=None):
        for i, f in enumerate(self.in_features):
            if i == 0:
                x = self.scale_heads[i](features[f])
            else:
                x = x + self.scale_heads[i](features[f])
        pred_mask_logits = self.predictor(x)

        # score_input = torch.cat([x, pred_mask_logits], dim=1)
        pred_scores = self.score(pred_mask_logits)
        pred_scores = F.max_pool2d(pred_scores, 
            kernel_size=pred_scores.size()[2:]).squeeze(-1).squeeze(-1)
        pred_scores = torch.sigmoid(pred_scores)

        pred_mask_logits = F.interpolate(pred_mask_logits,
            scale_factor=self.common_stride, mode="bilinear", align_corners=False)
        pred_mask_logits = F.softmax(pred_mask_logits, dim=1)
        
        if self.training:
            ins_losses = self.losses(pred_mask_logits, pred_scores, targets)
            return [], [], ins_losses
        else:
            return pred_mask_logits, pred_scores, {}
    
    def losses(self, pred_mask_logits, pred_scores, gt_instances):
        """
        Compute the mask prediction loss.

        Args:
            pred_mask_logits (Tensor): A tensor of shape (N, B, H, W) 
                where B is the total number of predicted masks in one images.
                The values are logits.
            pred_scores (Tensor): A tensor of shape (N, B)
                the confidences of each mask.
            gt_instances (list[Instances]): A list of N Instances, where N is the number of images
                in the batch. The ground-truth labels (class, box, mask,
                ...) associated with each instance are stored in fields.

        Returns:
            losses (dict): A dict{loss_key: loss_value} containing the loss.
        """
        ins_seg_loss = torch.zeros(1)
        ins_cls_loss = torch.zeros(1)
        for i, instances_per_image in enumerate(gt_instances):
            if len(instances_per_image) == 0:
                continue
            gt_masks_per_image = instances_per_image.gt_masks.to(device=pred_mask_logits.device).tensor
            # A tensor of shape (I, Hi, Wi), I=#instances in the image
            _, hi, wi = gt_masks_per_image.size()
            pred_masks_per_image = pred_mask_logits[i, :, :hi, :wi]
            # pred_masks_per_image is associated with gt_masks_per_image
            pred_order, gt_score = self.instances_association(pred_masks_per_image, gt_masks_per_image)
            pred_masks_per_image = pred_masks_per_image[i, pred_order]
            pred_scores_per_image = pred_scores[i, pred_order]

            # confidence loss: smooth L1
            cls_loss = F.smooth_l1_loss(pred_scores_per_image, 
                torch.from_numpy(gt_score).to(pred_scores_per_image.device))
            ins_cls_loss += cls_loss

            # segmentation loss: dice loss
            intersection = (pred_masks_per_image * gt_masks_per_image).sum()
            seg_loss = 1 - 2*(intersection + 1) / \
                (pred_masks_per_image.sum() + gt_masks_per_image.sum())
            ins_seg_loss += seg_loss

        return {"loss_ins_seg": ins_seg_loss*self.loss_weight, "loss_ins_cls": ins_cls_loss}

    def instances_association(self, pred_masks, gt_masks):
        """
        Instance Association Layer.
        Args:
            pred_masks (Tensor shape [B, H, W]):
            gt_masks (BitMask shape [I, H, W]):
        Return:
            pred_order (List): pred_mask index according to gt_mask
        """
        # calculation the cost that the i-th predicted instance 
        # mask is assigned to the j-th ground-truth instance.
        num_gt, num_pred = gt_masks.size(0), pred_masks.size(0)
        
        gt_masks = gt_masks.unsqueeze(0).expand(num_pred, -1, -1, -1)
        pred_masks = pred_masks.unsqueeze(1).expand(-1, num_gt, -1,-1)

        tp = torch.mean(gt_masks * pred_masks, dim=(-1, -2))
        fp = torch.mean(pred_masks, dim=(-1, -2)) - tp
        fn = torch.mean(gt_masks, dim=(-1, -2)) - tp
        # associate matrix: shape of (num_pred, num_gt) to minimize.
        associate_matrix = -tp / (tp + fp + fn + 1e-6)

        # Hungarian algorithm to solve optimal association problem.
        # paired (r[array],c[array]) r is index of pred, c is gt.
        r, c = linear_sum_assignment(associate_matrix)
        score = -associate_matrix[r, c]
        pred_order = r[c]

        return pred_order, score
