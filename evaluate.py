#coding=utf-8
import os 
import glob 
import json
import jittor as jt 
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from dataset.data import build_testdataset
from utils.ap_eval import calculate_VOC_mAP
from model.faster_rcnn import FasterRCNN
from utils.visualize import save_visualize_image


CLASSNAMES = ['i1', 'i10', 'i11', 'i12', 'i13', 'i14', 'i15', 'i2', 'i3', 'i4', 'i5', 'il100', 'il110', 'il50', 'il60', 'il70', 'il80', 'il90', 'ip', 'p1', 'p10', 'p11', 'p12', 'p13', 'p14', 'p15', 'p16', 'p17', 'p18', 'p19', 'p2', 'p20', 'p21', 'p23', 'p24', 'p25', 'p26', 'p27', 'p28', 'p3', 'p4', 'p5', 'p6', 'p7', 'p8', 'p9', 'pa10', 'pa12', 'pa13', 'pa14', 'pa8', 'pb', 'pc', 'pg', 'ph2', 'ph2.1', 'ph2.2', 'ph2.4', 'ph2.5', 'ph2.8', 'ph2.9', 'ph3', 'ph3.2', 'ph3.5', 'ph3.8', 'ph4', 'ph4.2', 'ph4.3', 'ph4.5', 'ph4.8', 'ph5', 'ph5.3', 'ph5.5', 'pl10', 'pl100', 'pl110', 'pl120', 'pl15', 'pl20', 'pl25', 'pl30', 'pl35', 'pl40', 'pl5', 'pl50', 'pl60', 'pl65', 'pl70', 'pl80', 'pl90', 'pm10', 'pm13', 'pm15', 'pm1.5', 'pm2', 'pm20', 'pm25', 'pm30', 'pm35', 'pm40', 'pm46', 'pm5', 'pm50', 'pm55', 'pm8', 'pn', 'pne', 'pr10', 'pr100', 'pr20', 'pr30', 'pr40', 'pr45', 'pr50', 'pr60', 'pr70', 'pr80', 'ps', 'pw2.5', 'pw3', 'pw3.2', 'pw3.5', 'pw4', 'pw4.2', 'pw4.5', 'w1', 'w10', 'w12', 'w13', 'w16', 'w18', 'w20', 'w21', 'w22', 'w24', 'w28', 'w3', 'w30', 'w31', 'w32', 'w34', 'w35', 'w37', 'w38', 'w41', 'w42', 'w43', 'w44', 'w45', 'w46', 'w47', 'w48', 'w49', 'w5', 'w50', 'w55', 'w56', 'w57', 'w58', 'w59', 'w60', 'w62', 'w63', 'w66', 'w8', 'i6', 'i7', 'i8', 'i9', 'p29', 'w29', 'w33', 'w36', 'w39', 'w4', 'w40', 'w51', 'w52', 'w53', 'w54', 'w6', 'w61', 'w64', 'w65', 'w67', 'w7', 'w9', 'pd', 'pe', 'pnl', 'w11', 'w14', 'w15', 'w17', 'w19', 'w2', 'w23', 'w25', 'w26', 'w27', 'pm2.5', 'ph4.4', 'ph3.3', 'ph2.6', 'i4l', 'i2r', 'im', 'wc', 'pcr', 'pcl', 'pss', 'pbp', 'p1n', 'pbm', 'pt', 'pn-2', 'pclr', 'pcs', 'pcd', 'iz', 'pmb', 'pdd', 'pctl', 'ph1.8', 'pnlc', 'pmblr', 'phclr', 'phcs', 'pmr']

def run_images(img_dir,classnames,checkpoint_path,save_path):
    test_dataset = build_testdataset(img_dir=img_dir)
    
    faster_rcnn = FasterRCNN(n_class=len(classnames)+1)
    faster_rcnn.load(checkpoint_path)  
    faster_rcnn.eval()
    
    results = {}
    for batch_idx,(images,image_sizes,img_ids) in tqdm(enumerate(test_dataset)):
        result = faster_rcnn.predict(images,score_thresh=0.01)
        for img_id,(pred_boxes,pred_scores,pred_labels) in zip(img_ids,result):
            objects = []
            for box,label,score in zip(pred_boxes.numpy(),pred_labels.numpy(),pred_scores.numpy()):
                bbox = {"xmin":float(box[0]),"ymin":float(box[1]),"xmax":float(box[2]),"ymax":float(box[3])}
                category = classnames[label-1]
                score = float(score)
                objects.append({"bbox":bbox,"category":category,"score":score})
            results[img_id+".jpg"]=objects
    
    os.makedirs(save_path,exist_ok=True)
    json.dump(results,open(os.path.join(save_path,"test_result.json"),"w"))

def build_comparison(detection_f,gt_f,classnames=None):
    detections = json.load(open(detection_f))
    gt_annos = json.load(open(gt_f))
    gts = gt_annos["imgs"]
    if classnames is None:
        classnames = gts["types"]

    img_ids = set(gts.keys())
    img_ids.update(detections.keys())
    
    results = []
    for img_id in img_ids:
        if not img_id in gts or len([o for o in gts[img_id]['objects'] if o["category"] in classnames])==0:
            gt_boxes = np.zeros((0,4))
            gt_labels = np.zeros((0,))
        else:
            gt = gts[img_id]
            objects = [o for o in gt['objects'] if o["category"] in classnames]
            gt_boxes = [[o['bbox']['xmin'],o['bbox']['ymin'],o['bbox']['xmax'],o['bbox']['ymax']] for o in objects]
            gt_labels = [classnames.index(o['category'])+1 for o in objects]
            gt_boxes = np.array(gt_boxes)
            gt_labels = np.array(gt_labels)

            w = gt_boxes[:,2]-gt_boxes[:,0]
            h = gt_boxes[:,3]-gt_boxes[:,1]
            use_range = (w>=16) & (h>=16)

            gt_boxes = gt_boxes[use_range,:]
            gt_labels = gt_labels[use_range]
        
        if img_id not in detections or len(detections[img_id])==0:
            pred_boxes = np.zeros((0,4))
            pred_labels = np.zeros((0,))
            pred_scores = np.zeros((0,))
        else:
            objects  = [o for o in detections[img_id] if o["category"] in classnames]
            pred_boxes = [[o['bbox']['xmin'],o['bbox']['ymin'],o['bbox']['xmax'],o['bbox']['ymax']] for o in objects]
            pred_labels = [classnames.index(o['category'])+1 for o in objects]
            pred_scores = [o["score"] for o in objects]

            pred_boxes = np.array(pred_boxes)
            pred_labels = np.array(pred_labels)
            pred_scores = np.array(pred_scores)
        
        results.append((img_id,pred_boxes,pred_labels,pred_scores,gt_boxes,gt_labels))
        index = (pred_scores>0.5)
        if False and (index.sum()>0 or gt_labels.shape[0]>0):
            save_visualize_image(img_id,pred_boxes[index,:],pred_scores[index],pred_labels[index],gt_boxes,gt_labels,classnames)
    return results

def filter_range(results,areaRange,is_filter_preds=True):
    filter_results = []
    for img_id,pred_boxes,pred_labels,pred_scores,gt_boxes,gt_labels in results:
        if is_filter_preds:
            pred_w = pred_boxes[:,2]-pred_boxes[:,0]
            pred_h = pred_boxes[:,3]-pred_boxes[:,1]
            pred_area = pred_h*pred_w
            pred_used = (pred_area>=areaRange[0]) & (pred_area<areaRange[1])
            pred_boxes = pred_boxes[pred_used,:]
            pred_labels = pred_labels[pred_used]
            pred_scores = pred_scores[pred_used]

        gt_w = gt_boxes[:,2]-gt_boxes[:,0]
        gt_h = gt_boxes[:,3]-gt_boxes[:,1]
        gt_area = gt_w*gt_h
        gt_used = (gt_area>=areaRange[0]) & (gt_area<areaRange[1])
        gt_boxes = gt_boxes[gt_used,:]
        gt_labels = gt_labels[gt_used]

        filter_results.append((img_id,pred_boxes,pred_labels,pred_scores,gt_boxes,gt_labels))
    return filter_results

def evaluate(detection_f,gt_f,classnames=CLASSNAMES,areaRanges=[[16*16,32*32],[32*32,96*96],[96*96,2048*2048],[16*16,None]]):
    results = build_comparison(detection_f,gt_f,classnames)
    mAPs = []
    for areaRange in areaRanges:
        if areaRange[1] is None:
            areaRange[1] = 2048*2048
            f_res = filter_range(results,areaRange,is_filter_preds=False)
        else:
            f_res = filter_range(results,areaRange)
        mAP,aps = calculate_VOC_mAP(f_res,classnames,use_07_metric=False)
        mAPs.append(mAP)
    return mAPs

def plot_aps(aps_json):
    mAP,classnames,aps = json.load(open(aps_json))
    a = []
    b = []
    for c,ap in zip(classnames,aps):
        if ap is not None:
            a.append(c)
            b.append(ap)
    plt.figure(figsize=(100, 7))
    plt.bar(range(len(a)),b,width=0.3)
    plt.xticks(range(len(a)),a)
    plt.savefig("test.png")

def main():
    jt.flags.use_cuda=1
    data_dir = "/data/lxl/dataset/tt100k"
    epoch = 15
    img_dir = f"{data_dir}/TEST_A"
    classnames = CLASSNAMES
    checkpoint_path = f"/data/lxl/dataset/tt100k/tt100k_2021/checkpoints/checkpoint_{epoch}.pkl"
    save_path = f"{data_dir}/trafficsign"
    run_images(img_dir,classnames,checkpoint_path,save_path)

if __name__ == "__main__":
    main()

