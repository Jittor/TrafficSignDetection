# coding=utf-8
import jittor as jt 
from jittor import nn 
import numpy as np
from .resnet import Resnet101,Resnet50
from .rpn import RegionProposalNetwork,AnchorTargetCreator,ProposalTargetCreator
from .roi_head import RoIHead
from utils.box_ops import loc2bbox

# from https://github.com/chenyuntc/simple-faster-rcnn-pytorch/
def _smooth_l1_loss(x, t, in_weight, sigma):
    sigma2 = sigma ** 2
    diff = in_weight * (x - t)
    abs_diff = diff.abs()
    flag = (abs_diff < (1. / sigma2)).float()
    y = (flag * (sigma2 / 2.) * (diff ** 2) + (1 - flag) * (abs_diff - 0.5 / sigma2))
    return y.sum()

# from https://github.com/chenyuntc/simple-faster-rcnn-pytorch/
def _fast_rcnn_loc_loss(pred_loc, gt_loc, gt_label, sigma):
    in_weight = jt.zeros(gt_loc.shape)
    # Localization loss is calculated only for positive rois.
    in_weight[(gt_label > 0).view(-1, 1).expand_as(in_weight)] = 1
    loc_loss = _smooth_l1_loss(pred_loc, gt_loc, in_weight.detach(), sigma)
    # Normalize by total number of negtive and positive rois.
    # ignore gt_label==-1 for rpn_loss
    loc_loss /= ((gt_label >= 0).sum().float()) 
    return loc_loss


class FasterRCNN(nn.Module):
    
    def __init__(self,n_class,backbone_name='resnet50'):
        super(FasterRCNN,self).__init__()
        if backbone_name=='resnet101':
            self.backbone = Resnet101(pretrained=True)
        elif backbone_name == 'resnet50':
            self.backbone = Resnet50(pretrained=True)
        else:
            assert False, f'{backbone_name} is not supported'

        self.n_class = n_class

        self.rpn = RegionProposalNetwork(in_channels=self.backbone.out_channels, 
                                        mid_channels=512, 
                                        ratios=[0.5, 1, 2],
                                        anchor_scales=[8, 16, 32], 
                                        feat_stride=self.backbone.feat_stride,
                                        nms_thresh=0.7,
                                        n_train_pre_nms=12000,
                                        n_train_post_nms=2000,
                                        n_test_pre_nms=6000,
                                        n_test_post_nms=300,
                                        min_size=16,)

        self.anchor_target_creator = AnchorTargetCreator(n_sample=256,
                                                         pos_iou_thresh=0.7, 
                                                         neg_iou_thresh=0.3,
                                                         pos_ratio=0.5)

        self.proposal_target_creator = ProposalTargetCreator(n_sample=128,
                                                             pos_ratio=0.25, 
                                                             pos_iou_thresh=0.5,
                                                             neg_iou_thresh_hi=0.5, 
                                                             neg_iou_thresh_lo=0.0)
        
        self.head = RoIHead(in_channels=self.backbone.out_channels,
                            n_class=n_class,
                            roi_size=7,
                            spatial_scale=1.0/self.backbone.feat_stride,
                            sampling_ratio=0)
        
        self.rpn_sigma = 3.
        self.roi_sigma = 1.
        
    def execute(self,images,boxes=None,labels=None):
        # w,h
        img_size = (images.shape[-1],images.shape[-2])
        features = self.backbone(images)
        if self.is_training():
            return self._forward_train(features,img_size,boxes,labels)
        else:
            return self._forward_test(features,img_size)
    
    def _forward_train(self,features,img_size,boxes,labels):
        N = features.shape[0]
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size)
        
        sample_rois = []
        gt_roi_locs = []
        gt_roi_labels = []
        sample_roi_indexs = []
        gt_rpn_locs = []
        gt_rpn_labels = []
        for i in range(N):
            index = jt.where(roi_indices == i)[0]
            roi = rois[index,:]
            box = boxes[i]
            label = labels[i]
            sample_roi, gt_roi_loc, gt_roi_label = self.proposal_target_creator(roi,box,label)
            sample_roi_index = i*jt.ones((sample_roi.shape[0],))
            
            sample_rois.append(sample_roi)
            gt_roi_labels.append(gt_roi_label)
            gt_roi_locs.append(gt_roi_loc)
            sample_roi_indexs.append(sample_roi_index)
            
            gt_rpn_loc, gt_rpn_label = self.anchor_target_creator(box,anchor,img_size)
            gt_rpn_locs.append(gt_rpn_loc)
            gt_rpn_labels.append(gt_rpn_label)
            
        sample_roi_indexs = jt.contrib.concat(sample_roi_indexs,dim=0)
        sample_rois = jt.contrib.concat(sample_rois,dim=0)
        roi_cls_loc, roi_score = self.head(features,sample_rois,sample_roi_indexs)
        
        # ------------------ RPN losses -------------------#
        rpn_locs = rpn_locs.reshape(-1,4)
        rpn_scores = rpn_scores.reshape(-1,2)
        gt_rpn_labels = jt.contrib.concat(gt_rpn_labels,dim=0)
        gt_rpn_locs = jt.contrib.concat(gt_rpn_locs,dim=0)
        rpn_loc_loss = _fast_rcnn_loc_loss(rpn_locs,gt_rpn_locs,gt_rpn_labels,self.rpn_sigma)
        rpn_cls_loss = nn.cross_entropy_loss(rpn_scores[gt_rpn_labels>=0,:],gt_rpn_labels[gt_rpn_labels>=0])
        
        # ------------------ ROI losses (fast rcnn loss) -------------------#
        gt_roi_locs = jt.contrib.concat(gt_roi_locs,dim=0)
        gt_roi_labels = jt.contrib.concat(gt_roi_labels,dim=0)
        n_sample = roi_cls_loc.shape[0]
        roi_cls_loc = roi_cls_loc.view(n_sample, np.prod(roi_cls_loc.shape[1:]).item()//4, 4)
        roi_loc = roi_cls_loc[jt.arange(0, n_sample).int32(), gt_roi_labels]
        roi_loc_loss = _fast_rcnn_loc_loss(roi_loc,gt_roi_locs,gt_roi_labels,self.roi_sigma)
        roi_cls_loss = nn.cross_entropy_loss(roi_score, gt_roi_labels)

        losses = [rpn_loc_loss, rpn_cls_loss, roi_loc_loss, roi_cls_loss]
        losses = losses + [sum(losses)]
        return losses
    
    def _forward_test(self,features,img_size):
        rpn_locs, rpn_scores, rois, roi_indices, anchor = self.rpn(features, img_size)
        roi_cls_locs, roi_scores = self.head(features, rois, roi_indices)
        return rpn_locs, rpn_scores,roi_cls_locs, roi_scores, rois, roi_indices
        
    def predict(self, images,score_thresh=0.7,nms_thresh = 0.3):
        N = images.shape[0]
        img_size = (images.shape[-1],images.shape[-2])
        rpn_locs, rpn_scores,roi_cls_locs, roi_scores, rois, roi_indices = self.execute(images)
        roi_cls_locs = roi_cls_locs.reshape(roi_cls_locs.shape[0],-1,4)
        probs = nn.softmax(roi_scores,dim=-1)
        rois = rois.unsqueeze(1).repeat(1,self.n_class,1)
        cls_bbox = loc2bbox(rois.reshape(-1,4),roi_cls_locs.reshape(-1,4))
        cls_bbox[:,0::2] = jt.clamp(cls_bbox[:,0::2],min_v=0,max_v=img_size[0])
        cls_bbox[:,1::2] = jt.clamp(cls_bbox[:,1::2],min_v=0,max_v=img_size[1])
        
        cls_bbox = cls_bbox.reshape(roi_cls_locs.shape)
        
        results = []
        for i in range(N):
            index = jt.where(roi_indices==i)[0]
            score = probs[index,:]
            bbox = cls_bbox[index,:,:]
            boxes = []
            scores = []
            labels = []
            for j in range(1,self.n_class):
                bbox_j = bbox[:,j,:]
                score_j = score[:,j]
                mask = jt.where(score_j>score_thresh)[0]
                bbox_j = bbox_j[mask,:]
                score_j = score_j[mask]
                dets = jt.contrib.concat([bbox_j,score_j.unsqueeze(1)],dim=1)
                keep = jt.nms(dets,nms_thresh)
                bbox_j = bbox_j[keep]
                score_j = score_j[keep]
                label_j = jt.ones_like(score_j).int32()*j
                boxes.append(bbox_j)
                scores.append(score_j)
                labels.append(label_j)
            
            boxes = jt.contrib.concat(boxes,dim=0)
            scores = jt.contrib.concat(scores,dim=0)
            labels = jt.contrib.concat(labels,dim=0)
            results.append((boxes,scores,labels))

        return results
    
    
    
    
        