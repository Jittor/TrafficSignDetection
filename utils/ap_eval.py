#coding=utf-8
import numpy as np 

def calculate_ious(gt_boxes,box):

    in_w = np.minimum(gt_boxes[:,2],box[2]) - np.maximum(gt_boxes[:,0],box[0])
    in_h = np.minimum(gt_boxes[:,3],box[3]) - np.maximum(gt_boxes[:,1],box[1])

    in_w = np.maximum(in_w,0)
    in_h = np.maximum(in_h,0)
    
    inter = in_w*in_h 

    area1 = (gt_boxes[:,2]-gt_boxes[:,0])*(gt_boxes[:,3]-gt_boxes[:,1])
    area2 = (box[2]-box[0])*(box[3]-box[1])
    union = area1+area2-inter
    ious = inter / union
    jmax = np.argmax(ious)
    maxiou = ious[jmax]
    return maxiou,jmax

def calculate_voc_ap(prec,rec,use_07_metric):
    if use_07_metric:
        # 11 point metric
        # http://host.robots.ox.ac.uk/pascal/VOC/pubs/everingham10.pdf (page 313)
        
        ap = 0.
        for t in np.arange(0., 1.1, 0.1):
            if np.sum(rec >= t) == 0:
                p = 0
            else:
                p = np.max(prec[rec >= t])
            ap = ap + p / 11.
    else:
        # correct AP calculation (from VOC 2010 challenge)
        # http://host.robots.ox.ac.uk/pascal/VOC/voc2012/devkit_doc.pdf (page 12)
        
        # first append sentinel values at the end
        mrec = np.concatenate(([0.], rec, [1.]))
        mpre = np.concatenate(([0.], prec, [0.]))

        # compute the precision envelope
        for i in range(mpre.size - 1, 0, -1):
            mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

        # to calculate area under PR curve, look for points
        # where X axis (recall) changes value
        i = np.where(mrec[1:] != mrec[:-1])[0]

        # and sum (\Delta recall) * prec
        ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    
    return ap

def calculate_prec_recall(result,label,iou_thresh):
    GTBB = {}
    BB = []
    BB_ids = []
    BB_scores = []
    n_gt_bboxes = 0
    for img_id,pred_boxes,pred_labels,pred_scores,gt_boxes,gt_labels in result:
        gts = gt_boxes[gt_labels==label,:]
        n_gt_bboxes+=gts.shape[0]
        GTBB[img_id]={
            'bboxes':gts,
            'detected':[False for i in range(gts.shape[0])]
        }
        pred = pred_boxes[pred_labels==label,:]
        scores = pred_scores[pred_labels==label]
        ids = [img_id for i in range(scores.shape[0])]
        BB.append(pred)
        BB_ids.extend(ids)
        BB_scores.append(scores)
    
    if n_gt_bboxes==0:
        return None,None 

    if len(BB_ids)>0:
        BB = np.concatenate(BB,axis=0)
        BB_scores = np.concatenate(BB_scores,axis=0)
        indexes = np.argsort(-BB_scores)
        BB = BB[indexes,:]
        BB_ids = np.array(BB_ids)[indexes]
    else:
        return np.array([0.]),np.array([0.])

    n_pred = len(BB_ids)
    tp = np.zeros(n_pred)
    fp = np.zeros(n_pred)

    for d in range(n_pred):
        bb = BB[d,:]
        gt_boxes = GTBB[BB_ids[d]]['bboxes']

        ioumax = 0.0
        if gt_boxes.shape[0]>0:
            ioumax,jmax = calculate_ious(gt_boxes,bb)
        if ioumax>iou_thresh:
            if not GTBB[BB_ids[d]]['detected'][jmax]:
                tp[d]=1.
                GTBB[BB_ids[d]]['detected'][jmax] = True 
            else:
                fp[d]=1.
        else:
            fp[d] = 1.

    fp = np.cumsum(fp)
    tp = np.cumsum(tp)

    rec = tp / float(n_gt_bboxes)
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)

    return prec,rec


def calculate_VOC_mAP(result,classnames,iou_thresh=0.5,use_07_metric=True):
    '''
    INPUTS:
        result: the result list of detections, 
        it's like [(img_id,pred_boxes,pred_labels,pred_scores,gt_boxes,gt_labels),...,(...)]
        In Labels, 0 is background

        classnames: the class we used to calculate mAP, and "background" is not in it.
        
        iou_thresh: A bounding box reported by an algorithm is considered
        correct if its area intersection over union with a ground 
        truth bounding box is beyond 50%.

        use_07_metric: True means we use voc 07 challenge metric, False means use 10 metric
    
    OUTPUT:
        mAP: mean Average Precision
    '''
    
    aps = []
    all_aps = []
    for i,classname in enumerate(classnames):
        # background is not in classnames and it's 0
        label = i+1
        prec,rec = calculate_prec_recall(result,label,iou_thresh)
        if prec is None:
            all_aps.append(None)
            continue
        ap = calculate_voc_ap(prec,rec,use_07_metric)
        all_aps.append(ap)
        aps.append(ap)

    if len(aps)>0:
        mAP = np.mean(aps)
    else:
        mAP = 0.
    return mAP,all_aps 


    